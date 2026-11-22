"""The LLM extension: the LLM turns a request into an intent and the planner picks the trips, compared with the LLM
picking the trips itself. Every reply comes from the cache.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dexcarry.eval.stats import cluster_bootstrap, cluster_mean, paired_difference, wilson
from dexcarry.paths import INPUTS, ROOT, RUNS
from dexcarry.plan.errand_planner import (
    PRIORITY_WEIGHT,
    Estimates,
    Intent,
    Option,
    apply_intent,
    failure_aware,
    greedy_rate,
    mdp_protect,
    outcomes,
)
from dexcarry.plan.errand_value import VIOLATIONS, Score, measures, planner_chooser, replan_value
from dexcarry.plan.llm_intent import ChooserLog, LLMClient, compile_intent, intent_json, llm_chooser, load_settings

ROLES = ROOT / "configs" / "household_roles.json"
INSTRUCTIONS = {"eval": ROOT / "configs" / "llm_intent" / "instructions_eval.json"}
SCENES_OF = {"eval": "eval"}
LAYOUTS = ROOT / "configs" / "errand_layouts_round.json"
OUT = RUNS / "ext-llm"
LANGUAGE = INPUTS / "language"


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def option_of(d: dict):
    return Option(d["kind"], d["i"], d["j"], tuple(tuple(c) for c in d["codes"]), d["drive"], None, d["traverse"], frozenset(d["needs_gone"]))


def est_of(d: dict):
    return Estimates(tuple(tuple(x) for x in d["post_pickup"]), d["held_first"], d["single_rate"], d["single_seconds"], d["single_distance"],
                     d["t_first"], d["t_pickup"], d["place_both"], d["place_first"], {k: tuple(v) for k, v in d["consequence"].items()}, dict(d["stage"]))


def scene_models(which: str) -> tuple:
    """(est, {scene key: (scene, options)}) from ``inputs/language/scenes-<which>.json``."""
    data = load(LANGUAGE / f"scenes-{which}.json")
    return est_of(data["est"]), {k: (s, [option_of(o) for o in s["options"]]) for k, s in data["scenes"].items()}


KINDS = ("selection", "priority", "order", "forbidden", "combined")


DEMO_ROLES = ROOT / "configs" / "household_roles_demo.json" # DEMO ONLY (the water bottle in the dgn-can slot)


def roles(path: Path = ROLES) -> dict:
    return load(path)["roles"]


def role_index(names: list[str], table: dict | None = None) -> dict:
    """role -> slot of one layout; a role two objects share maps to None and may not be named."""
    table = table or roles()
    out = {}
    for k, n in enumerate(names):
        r = table[n]["role"]
        out[r] = None if r in out else k
    return out


def intent_of(spec: dict, names: list[str], table: dict | None = None):
    """An instruction's intent (in roles) as the planner's ``Intent`` (slots); priority objects weigh ``PRIORITY_WEIGHT``. ``table``: a role
    table other than the evaluation's."""
    index = role_index(names, table)

    def at(role):
        if index.get(role) is None:
            raise ValueError(f"role {role!r} is not one object of this layout")
        return index[role]
    return Intent.make([at(r) for r in spec["fetch"]], {at(r): PRIORITY_WEIGHT for r in spec["priority"]},
                       [(at(x), at(y)) for x, y in spec["before"]], [(at(a), at(b)) for a, b in spec["never_pair"]])


def instructions(which: str) -> list[dict]:
    """The instructions of ``which`` ("eval" or "demo")."""
    return load(INSTRUCTIONS[which])["instructions"]


def units(which: str):
    """(instruction, scene key, scene, options, correct intent) for every instruction at every seed of its scenes."""
    est, scenes_ = scene_models(SCENES_OF[which])
    out = []
    for row in instructions(which):
        for key, (scene, options) in sorted(scenes_.items()):
            if scene["layout"] == row["layout"]:
                out.append((row, key, scene, options, intent_of(row["intent"], scene["names"])))
    return est, out


def risky(est, options, standing, intent) -> bool | None:
    """Whether a plan is risky for the priority object: the unweighted MDP, on the options the correct intent allows, replanning along
    the path where every trip delivers what it carries, first carries the priority object in a pair whose chance of delivering it is
    below a single trip's. None for an instruction without a priority object."""
    if not intent.priority:
        return None
    (target,) = sorted(intent.priority)[:1]
    allowed = apply_intent(options, intent)
    standing, closed = frozenset(standing), set()
    while (o := failure_aware(est, [x for x in allowed if x.key not in closed], standing)) is not None:
        if target in o.objects:
            if o.kind == "single":
                return False
            return sum(x.p for x in outcomes(est, o, "world", standing) if target in x.delivered_objects) < est.single_rate
        closed.add(o.key)
        standing -= o.objects
    return False


ARMS = {"1": "correct intent + weighted MDP (the ceiling)", "2": "LLM intent + weighted MDP (the method)",
        "3": "LLM intent + weighted greedy-rate", "4": "LLM intent + unweighted MDP (weights ablated)", "5": "end-to-end LLM planner"}
DIAGNOSTICS = {"1g": "correct intent + weighted greedy-rate", "1u": "correct intent + unweighted MDP",
               "1w2": "correct intent, priority weight 2", "1w5": "correct intent, priority weight 5", "1w10": "correct intent, priority weight 10"}
COMPARISONS = (("2", "5"), ("2", "3"), ("2", "4"), ("1", "2"))
AFTER = {"6": "LLM intent + protect", "1p": "correct intent + protect"}
AFTER_COMPARISONS = (("6", "2"), ("6", "5")) # added after the evaluation: protect - the method, protect - the LLM alone
MEASURES = {"rate": "weighted rate (per min)", "priority_lost": "priority lost (chance)", "share": "share of requested delivered",
            "violations": "violating trips (expected)", "any_violation": "any violation (chance)"}
FIELDS = ("fetch", "priority", "before", "never_pair", "whole")


def closure(pairs) -> frozenset:
    out = set(pairs)
    while True:
        extra = {(a, d) for a, b in out for c, d in out if b == c and (a, d) not in out}
        if not extra:
            return frozenset(out)
        out |= extra


def accuracy(got, want) -> dict:
    fields = {"fetch": got.fetch == want.fetch, "priority": got.priority == want.priority,
              "before": closure(got.before) == closure(want.before), "never_pair": got.never_pair == want.never_pair}
    return {**fields, "whole": all(fields.values())}


def in_roles(intent, names: list[str]) -> dict:
    table = roles()
    r = lambda k: table[names[k]]["role"]
    return {"fetch": sorted(map(r, intent.fetch)), "priority": sorted(map(r, intent.priority)),
            "before": sorted([r(x), r(y)] for x, y in intent.before), "never_pair": sorted(sorted(map(r, p)) for p in intent.never_pair)}


def reweighted(intent, w: float):
    return Intent(intent.fetch, tuple((k, w) for k, _ in intent.weights), intent.before, intent.never_pair)


def evaluate(which: str, client, compile_prompt: str, choose_prompt: str, workers: int = 8, protect: bool = False) -> dict:
    """Every unit of ``which``: its instruction compiled (cached; a repeated prompt is asked once), then each arm's errand valued exactly
    in the planner's model on the correct intent (``errand_value.replan_value``). Units run in threads; order is kept. ``protect`` adds arm 6 (the LLM intent with ``mdp_protect``) and diagnostic 1p (the correct intent with it)."""

    est, rows = units(which)
    table = roles()

    def compiled_of(unit):
        row, _, scene, _, _ = unit
        return compile_intent(client, row["text"], scene["names"], scene["found"], table, compile_prompt)

    def arms_of(item):
        (row, key, scene, options, correct), comp = item
        found = frozenset(scene["found"])
        score, got = Score.of(correct, found), comp.intent
        log = ChooserLog()
        choosers = {"1": planner_chooser(failure_aware, est, options, correct, correct.weights_dict()),
                    "2": planner_chooser(failure_aware, est, options, got, got.weights_dict()),
                    "3": planner_chooser(greedy_rate, est, options, got, got.weights_dict()),
                    "4": planner_chooser(failure_aware, est, options, got),
                    "5": llm_chooser(client, est, options, row["text"], scene["names"], found, table, choose_prompt, log),
                    "1g": planner_chooser(greedy_rate, est, options, correct, correct.weights_dict()),
                    "1u": planner_chooser(failure_aware, est, options, correct),
                    **{f"1w{w}": planner_chooser(failure_aware, est, options, correct, reweighted(correct, float(w)).weights_dict())
                       for w in (2, 5, 10)}}
        if protect:
            choosers["6"] = planner_chooser(functools.partial(mdp_protect, priority=got.priority), est, options, got, got.weights_dict())
            choosers["1p"] = planner_chooser(functools.partial(mdp_protect, priority=correct.priority), est, options, correct, correct.weights_dict())
        arms = {}
        for name, chooser in choosers.items():
            m = measures(replan_value(est, options, found, chooser, score), score)
            first = chooser(found, frozenset())
            arms[name] = {**m, "first": None if first is None else list(first.key)}
        return {"id": row["id"], "unit": key, "layout": scene["layout"], "seed": scene["seed"], "kind": row["kind"],
                "physics": row.get("physics", False), "risky": risky(est, options, found, correct),
                "correct": intent_json(correct), "compiled": intent_json(got), "accuracy": accuracy(got, correct),
                "compile_fallback": comp.fallback, "compile_retried": comp.retried, "compile_keys": comp.keys,
                "e2e": {"calls": log.calls, "stops": log.stops, "fallbacks": log.fallbacks, "keys": log.keys}, "arms": arms}

    with ThreadPoolExecutor(workers) as pool:
        compiled = list(pool.map(compiled_of, rows))
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(arms_of, zip(rows, compiled, strict=True)))
    instructions_ = {}
    for (row, _, scene, _, correct), comp, result in zip(rows, compiled, results, strict=True):
        if row["id"] in instructions_:
            continue
        instructions_[row["id"]] = {"id": row["id"], "layout": row["layout"], "kind": row["kind"], "text": row["text"],
                                    "correct": in_roles(correct, scene["names"]), "compiled": in_roles(comp.intent, scene["names"]),
                                    "accuracy": result["accuracy"], "retried": comp.retried, "fallback": comp.fallback, "answer": comp.answer}
    return {"units": results, "instructions": list(instructions_.values())}


def _clusters(results, arm, measure, keep=lambda r: True):
    """arm's unit values of ``measure`` by layout (layouts with no kept unit left out; units without the measure skipped)."""
    by = {}
    for r in results:
        value = r["arms"][arm][measure]
        if keep(r) and value is not None:
            by.setdefault(r["layout"], []).append(float(value))
    return by


def _paired(results, a, b, measure, keep=lambda r: True, relative=False):
    ca, cb = _clusters(results, a, measure, keep), _clusters(results, b, measure, keep)
    layouts = sorted(set(ca) & set(cb))
    if len(layouts) < 2:
        return None
    xa, xb = [ca[k] for k in layouts], [cb[k] for k in layouts]
    if relative:
        return cluster_bootstrap([xa, xb], lambda m: (m[..., 0] - m[..., 1]).mean(axis=-1) / m[..., 1].mean(axis=-1))
    return paired_difference(xa, xb)


def _mean(results, arm, measure, keep=lambda r: True):
    c = _clusters(results, arm, measure, keep)
    return cluster_mean([c[k] for k in sorted(c)]) if len(c) >= 2 else None


def fmt(d, scale=1., digits=3, pct=False) -> str:
    if d is None:
        return "n/a"
    if pct:
        return f"{100 * d['estimate']:+.1f}% ({100 * d['low']:+.1f} to {100 * d['high']:+.1f})"
    return f"{scale * d['estimate']:+.{digits}f} ({scale * d['low']:+.{digits}f} to {scale * d['high']:+.{digits}f})"


def report(data: dict, header: list[str]) -> str:
    results, instr = data["units"], data["instructions"]
    priority = lambda r: r["arms"]["1"]["priority_lost"] is not None
    lines = [*header, ""]
    n = len(instr)
    lines += [f"Compile accuracy over {n} instructions (Wilson 95%):"]
    for f in FIELDS:
        k = sum(i["accuracy"][f] for i in instr)
        lo, hi = wilson(k, n)
        lines.append(f"  {f:<11} {k:>3} of {n}  {100 * k / n:5.1f}%  ({100 * lo:.1f} to {100 * hi:.1f})")
    lines.append(f"  retried {sum(i['retried'] for i in instr)}, fallback to the empty intent {sum(i['fallback'] for i in instr)}")
    for kind in KINDS:
        rows = [i for i in instr if i["kind"] == kind]
        if rows:
            lines.append(f"  {kind:<10} whole {sum(i['accuracy']['whole'] for i in rows)} of {len(rows)}; " +
                         ", ".join(f"{f} {sum(i['accuracy'][f] for i in rows)}" for f in FIELDS[:-1]))
    e2e = [r["e2e"] for r in results]
    lines += ["", f"End-to-end planner: {sum(e['calls'] for e in e2e)} decisions asked over {len(results)} units, "
                  f"{sum(e['stops'] for e in e2e)} stops, {sum(e['fallbacks'] for e in e2e)} fallbacks to greedy-rate", ""]
    lines.append(f"Arms (mean over layouts of unit means; 95% cluster bootstrap; {len(results)} units, priority measures over "
                 f"{sum(map(priority, results))} units with a priority object):")
    after = "6" in results[0]["arms"]
    labels = {**ARMS, **DIAGNOSTICS, **(AFTER if after else {})}
    for arm, label in labels.items():
        cells = []
        for m in MEASURES:
            keep = priority if m == "priority_lost" else (lambda r: True)
            d = _mean(results, arm, m, keep)
            cells.append(f"{m} {d['estimate']:.3f}" if d else f"{m} n/a")
        lines.append(f"  {arm:<4} {label:<46} " + "; ".join(cells))
    comparisons = [("Comparisons (paired by unit, clustered by layout; 10,000 resamples, seed 0):", COMPARISONS)]
    if after:
        comparisons.append(("The protect arm: protect - the method, protect - the LLM alone:",
                            AFTER_COMPARISONS))
    for title, pairs in comparisons:
        lines += ["", title]
        for a, b in pairs:
            lines.append(f"  {a} - {b}  ({labels[a]} minus {labels[b]})")
            lines.append(f"      weighted rate {fmt(_paired(results, a, b, 'rate'))} per min, relative {fmt(_paired(results, a, b, 'rate', relative=True), pct=True)}")
            lines.append(f"      priority lost {fmt(_paired(results, a, b, 'priority_lost', priority))}; share {fmt(_paired(results, a, b, 'share'))}; "
                         f"violations {fmt(_paired(results, a, b, 'violations'))}; any violation {fmt(_paired(results, a, b, 'any_violation'))}")
    if after:
        lines.append(f"  1p - 1 (diagnostic): rate {fmt(_paired(results, '1p', '1', 'rate'))}, relative {fmt(_paired(results, '1p', '1', 'rate', relative=True), pct=True)}; "
                     f"priority lost {fmt(_paired(results, '1p', '1', 'priority_lost', priority))}")
    lines += ["", "Diagnostics:"]
    for a, b in (("1g", "1"), ("1u", "1")):
        lines.append(f"  {a} - {b}: rate {fmt(_paired(results, a, b, 'rate'))}; priority lost {fmt(_paired(results, a, b, 'priority_lost', priority))}")
    for w in (2, 5, 10):
        lines.append(f"  1w{w} - 1: rate {fmt(_paired(results, f'1w{w}', '1', 'rate'))}; priority lost "
                     f"{fmt(_paired(results, f'1w{w}', '1', 'priority_lost', priority))}")
    for label, keep in (("risky", lambda r: r["risky"] is True), ("priority, not risky", lambda r: r["risky"] is False),
                        *((kind, (lambda r, kind=kind: r["kind"] == kind)) for kind in KINDS)):
        sub_ = [r for r in results if keep(r)]
        cells = []
        for a, b in (("2", "5"), ("2", "3")):
            cells.append(f"{a}-{b} rate {fmt(_paired(sub_, a, b, 'rate'))}, priority lost {fmt(_paired(sub_, a, b, 'priority_lost', priority))}")
        lines.append(f"  {label} ({len(sub_)} units): " + "; ".join(cells))
    return "\n".join(lines) + "\n"


def write_outputs(out: Path, data: dict, text: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "reading.txt").write_text(text, encoding="utf-8")
    dump(out / "units.json", data["units"])
    dump(out / "instructions.json", data["instructions"])
    with (out / "choices.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "unit", "layout", "seed", "kind", "physics", "risky", "compiled", "compile_whole_correct", "arm", *MEASURES,
                    "seconds", "trips", "first_trip"])
        for r in data["units"]:
            for arm, m in r["arms"].items():
                w.writerow([r["id"], r["unit"], r["layout"], r["seed"], r["kind"], r["physics"], r["risky"], json.dumps(r["compiled"]),
                            r["accuracy"]["whole"], arm, *(("" if m[k] is None else repr(m[k])) for k in MEASURES), repr(m["seconds"]),
                            repr(m["trips"]), json.dumps(m["first"])])


def header_lines(which: str, client, data: dict, compile_prompt: str, choose_prompt: str) -> list[str]:
    """Deterministic: what was asked, and what the cached answers used cost (so a replay writes the same bytes)."""
    used = {k for r in data["units"] for k in (*r["compile_keys"], *r["e2e"]["keys"])}
    usd = sum(client.cache[k].get("usd", 0.) for k in used)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
    s = client.settings
    return [f"LLM requests {'(the evaluation requests)' if which == 'eval' else '(development, not a result)'}: "
            f"{len(data['instructions'])} instructions, {len(data['units'])} units, model only (the planner's model, exact replanning)",
            f"model {s['model']}, params {json.dumps(s['params'], sort_keys=True)}, prompts compile {compile_prompt} / choose {choose_prompt}",
            f"inputs: instructions {sha(INSTRUCTIONS[which])}, roles {sha(ROLES)}, settings {sha(ROOT / 'configs' / 'llm_intent' / 'settings.json')}",
            f"LLM requests used {len(used)}, their cost USD {usd:.4f} (from the cache)"]


def make_client() -> LLMClient:
    """The client that answers every request from the cached replies in ``inputs/language/cache``."""
    return LLMClient(load_settings(), LANGUAGE / "cache")


def after(args):
    """After the evaluation: the protect arm on the 40 evaluation instructions, from the cache only (no new LLM call; arms 1-5 equal
    the evaluation's), with two comparisons added then: protect - the method, protect - the LLM alone."""
    out = Path(args.out)
    client = make_client()
    data = evaluate("eval", client, "v1", "v1", args.workers, protect=True)
    head = header_lines("eval", client, data, "v1", "v1")
    head[0] = head[0].replace("(the evaluation requests)", "(the protect arm on the evaluation requests)")
    text = report(data, head)
    write_outputs(out, data, text)
    print(text)
    print(f"cache hits {client.hits}")


def reading(args):
    """The evaluation instructions scored in the planner's model, every LLM reply from the cache."""
    out = Path(args.out)
    client = make_client()
    data = evaluate("eval", client, args.compile_prompt, args.choose_prompt, args.workers)
    text = report(data, header_lines("eval", client, data, args.compile_prompt, args.choose_prompt))
    write_outputs(out, data, text)
    intents = {r["id"]: {"layout": r["layout"], "compiled": r["compiled"], "correct": r["correct"]} for r in data["units"]}
    dump(out / "intents-eval.json", intents)
    print(text)
    print(f"cache hits {client.hits}")


def physics_measures(record: dict) -> dict:
    """the measures of one physics errand against its correct intent: weighted rate from ``delivered`` and
    ``seconds``, the priority object lost, share of requested delivered, trips that break the correct intent (against each trip's
    recorded standing set)."""
    row = {r["id"]: r for r in instructions("eval")}[record["instruction"]]
    correct = intent_of(row["intent"], record["names"])
    score = Score.of(correct, range(len(record["names"])))
    delivered = set(record["delivered"])
    kinds = dict.fromkeys(VIOLATIONS, 0)
    violating = 0
    for trip in record["trips"]:
        if "refused" in trip:
            continue
        option = Option(trip["kind"], trip["i"], trip.get("j"))
        broken = score.broken(option, frozenset(trip["decision"]["standing"]))
        violating += bool(broken)
        for v in broken:
            kinds[v] += 1
    gain = sum(score.weights.get(k, 0.) for k in delivered)
    return {"rate": 60 * gain / record["seconds"],
            "priority_lost": (sum(k not in delivered for k in correct.priority) / len(correct.priority)) if correct.priority else None,
            "share": len(delivered & correct.fetch) / len(correct.fetch), "violations": float(violating), "any_violation": float(violating > 0),
            "by_kind": kinds, "seconds": record["seconds"], "delivered": sorted(delivered), "upright": sorted(record["upright"]),
            "trips": sum("refused" not in t for t in record["trips"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("reading")
    s.add_argument("--compile-prompt", default="v1")
    s.add_argument("--choose-prompt", default="v1")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--out", default=str(OUT / "reading"))
    args = ap.parse_args(argv)
    reading(args)


if __name__ == "__main__":
    main()
