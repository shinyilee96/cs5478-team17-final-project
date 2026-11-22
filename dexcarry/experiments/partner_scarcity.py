"""Partner-scarcity layouts: an object that often fails alone has few partners, and a tempting pair lures the
greedy planner into using one of them first.
"""

from __future__ import annotations

import functools
import json

from dexcarry.envs.round_table import draw_positions
from dexcarry.experiments import errands, measured_odds
from dexcarry.paths import ROOT
from dexcarry.plan.errand_planner import greedy_rate, open_options, outcomes

SCARCITY_STREAM = 61
ATTEMPTS = 200
LURE_MARGIN = .05 # "lure 5%": the lure's one-step rate this far above every open option holding U
TARGET = 4.0 # m, the evaluation's distance
SCARCITY_LAYOUTS = ROOT / "configs" / "errand_layouts_scarcity.json"


@functools.lru_cache(maxsize=1)
def source():
    """the round table's round layouts, errand_plans.json's rows and the measured odds."""
    return errands.load(errands.LAYOUTS["round"]), errands.load(errands.ERRAND_PLANS)["options"], measured_odds.load_odds()


def scarcity_rows(entry: dict) -> list[dict]:
    """errand_plans.json's rows of the entry's source layout among its objects, with ``i`` and ``j`` in the partner scarcity layout's order (the source's
    kept as ``source_i``, ``source_j``)."""
    _, rows, _ = source()
    pos = {s: k for k, s in enumerate(entry["idx"])}
    return [{**r, "i": pos[r["i"]], "j": pos[r["j"]], "source_i": r["i"], "source_j": r["j"]} for r in rows
            if r["layout"] == entry["source"] and r["i"] in pos and r["j"] in pos]


def draw(entry: dict, seed: int, attempt: int):
    layouts, _, _ = source()
    return draw_positions(len(entry["idx"]), entry["source"], seed, layouts["spacing"],
                          key=[seed, SCARCITY_STREAM, entry["source"], *entry["idx"], attempt])


def measured_options(entry: dict, positions) -> list:
    return measured_odds.true_options(entry["names"], scarcity_rows(entry), positions, TARGET, source()[2], 1.)


def one_step(est, options, o) -> float:
    out = outcomes(est, o)
    return sum(x.p * x.delivered for x in out) / sum(x.p * x.seconds for x in out)


def lure_at(entry: dict, positions, est=None) -> tuple[bool, float, tuple]:
    """At the errand's start: does greedy-rate's best one-step option (measured odds) pair a partner of U with something other than U;
    its rate over the best open option holding U, less 1; and that option's key."""

    est = est or errands.estimates()[0]
    options = measured_options(entry, positions)
    n = len(entry["names"])
    u, partners = entry["names"].index(entry["U"]), {entry["names"].index(w) for w in entry["partners"]}
    best = greedy_rate(est, options, frozenset(range(n)))
    holds = best.kind == "pair" and best.i in partners and u not in best.objects
    with_u = max(one_step(est, options, o) for o in open_options(options, range(n)) if u in o.objects)
    return holds, one_step(est, options, best) / with_u - 1, best.key


def positions_for(entry: dict, seed: int, lure: str, est=None) -> tuple[list | None, int | None]:
    """The position rule: "drawn", attempt 0 with no condition; "lure", the first attempt (of ``ATTEMPTS``) at which the
    lure holds; "lure 5%", the first at which it holds by ``LURE_MARGIN``. (None, None) if no attempt qualifies."""
    if lure == "drawn":
        return draw(entry, seed, 0), 0
    for attempt in range(ATTEMPTS):
        positions = draw(entry, seed, attempt)
        if positions is None:
            continue
        holds, margin, _ = lure_at(entry, positions, est)
        if holds and (lure == "lure" or margin >= LURE_MARGIN - 1e-12):
            return positions, attempt
    return None, None


@functools.lru_cache(maxsize=1)
def scarcity_file() -> dict:
    return json.loads(SCARCITY_LAYOUTS.read_text(encoding="utf-8"))


def entry_of(layout: int) -> dict:
    """The partner scarcity layout ``layout`` of ``configs/errand_layouts_scarcity.json``: source, indices, names, ``U``, partners, lure rule, cell."""
    return scarcity_file()["entries"][layout]


def is_lure(entry: dict, key) -> bool:
    """A trip's option ``key`` is a lure: a pair whose wrapped object is a partner of ``U`` and which does not hold ``U``."""
    names = entry["names"]
    return key[0] == "pair" and names[key[1]] in entry["partners"] and names.index(entry["U"]) not in key[1:]


def errand_story(r: dict) -> dict:
    """What one partner scarcity errand did with its stranded-prone object: the first trip's option, whether it was a lure, how ``U`` travelled
    (in a pair, alone, or never tried) and whether it was delivered."""
    entry = entry_of(r["layout"])
    u = entry["names"].index(entry["U"])
    trips = [t for t in r["trips"] if "refused" not in t]
    first = trips[0] if trips else None
    first_key = (first["kind"], first["i"], *(() if first["kind"] == "single" else (first["j"],))) if first else None
    with_u = [t for t in trips if t["i"] == u or t.get("j") == u]
    how = "never" if not with_u else ("pair" if with_u[0]["kind"] == "pair" else "alone")
    alone = [t for t in with_u if t["kind"] == "single"]
    return {"first": first_key, "lure_first": bool(first_key) and is_lure(entry, first_key), "how": how,
            "alone_trips": len(alone), "alone_delivered": sum(bool(t["delivered"].get(str(u))) for t in alone),
            "u_delivered": u in r["delivered"] or str(u) in map(str, r["delivered"])}

