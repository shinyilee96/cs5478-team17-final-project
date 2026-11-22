"""LLM request handling: a cached client, the request-to-intent compiler, and the LLM-only trip chooser (the baseline)."""

from __future__ import annotations

import hashlib
import json
import string
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .errand_planner import PRIORITY_WEIGHT, Intent, greedy_rate, open_options, outcomes

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "configs" / "llm_intent" / "settings.json"
CACHE_DIR = ROOT / "inputs" / "language" / "cache"


class CacheMiss(RuntimeError):
    """A request the cache does not hold."""


def canonical(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(body: dict) -> str:
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def load_settings(path: Path = SETTINGS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class LLMClient:
    """Answers each request from the cached replies (``cache*.jsonl`` in ``cache_dir``), found by the SHA-256 of the request.
    A request that is not cached stops with ``CacheMiss``."""
    settings: dict
    cache_dir: Path = CACHE_DIR
    cache: dict = field(default_factory=dict)
    hits: int = 0

    def __post_init__(self):
        self._lock = threading.Lock() # threads share one client
        self.cache_dir = Path(self.cache_dir)
        for path in sorted(self.cache_dir.glob("cache*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache.setdefault(row["key"], row)

    def body(self, messages: list[dict], name: str, schema: dict) -> dict:
        s = self.settings
        if schema is None: # (v3): free text (thinking, then the answer); no schema
            return {"model": s["model"], "messages": messages, **s["params"]}
        return {"model": s["model"], "messages": messages, **s["params"],
                "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}

    def ask(self, messages: list[dict], name: str, schema: dict) -> dict:
        """The cached response to this exact request (``content``, ``key``, ``cached``)."""
        key = request_key(self.body(messages, name, schema))
        with self._lock:
            if key not in self.cache:
                raise CacheMiss(key)
            self.hits += 1
            return {**self.cache[key]["response"], "key": key, "cached": True}


def ids_for(found) -> dict:
    """Neutral ids: slot -> "A", "B",... in slot order."""
    return {k: string.ascii_uppercase[n] for n, k in enumerate(sorted(found))}


def object_lines(names: list[str], found, roles: dict) -> list[str]:
    ids = ids_for(found)
    out = []
    for k in sorted(found):
        r = roles[names[k]]
        size = " x ".join(f"{v:.1f}" for v in r["size_cm"])
        out.append(f"- {ids[k]}: {r['role']} ({r['shape']}, {size} cm)")
    return out


COMPILE_PROMPT_V1 = """You turn a person's spoken request to a home robot into a structured intent.
The robot fetches objects from a round table and brings them to the person. It carries one or two objects per trip.

Fill four fields, using only the ids of the objects listed:
- fetch: the objects the person wants brought. Leave out every object they did not ask for.
- priority: the objects the person says matter most. Empty if they say none matters more than the others.
- before: ordering rules the person states, each as {"first": x, "then": y}: x must be delivered before y is carried. Empty if no order is stated.
- never_pair: pairs the person says must not be carried together, each as {"a": x, "b": y}. Empty if none.

Every id in priority, before and never_pair must also be in fetch."""

COMPILE_EXAMPLES_V2 = """Examples (each has its own objects; ids belong to that example only):

Objects on the table:
- A: football (ball, 22.0 x 22.0 x 22.0 cm)
- B: bicycle pump (pump, 5.0 x 5.0 x 45.0 cm)
- C: ball needle (needle, 0.5 x 0.5 x 4.0 cm)
- D: skipping rope (rope, 20.0 x 15.0 x 5.0 cm)
- E: umbrella (umbrella, 8.0 x 8.0 x 60.0 cm)
Request: "The kids want to play football but the ball is flat."
Reasoning: a flat ball needs pumping, and the pump needs the needle; the rope and umbrella are not needed.
Answer: {"fetch":["A","B","C"],"priority":[],"before":[],"never_pair":[]}

Objects on the table:
- A: birthday candles (small box, 8.0 x 2.0 x 12.0 cm)
- B: matches (small box, 5.0 x 3.5 x 1.5 cm)
- C: cake knife (knife, 3.0 x 2.0 x 30.0 cm)
- D: paper plates (stack, 23.0 x 23.0 x 3.0 cm)
- E: scissors (scissors, 8.0 x 1.5 x 18.0 cm)
Request: "Let's light the birthday candles, and then we'll cut the cake."
Reasoning: lighting needs the candles and the matches, cutting needs the knife, and lighting comes first.
Answer: {"fetch":["A","B","C"],"priority":[],"before":[{"first":"A","then":"C"},{"first":"B","then":"C"}],"never_pair":[]}

Objects on the table:
- A: TV remote, takes AA batteries (remote, 5.0 x 2.5 x 20.0 cm)
- B: AA batteries (pack, 6.0 x 1.5 x 9.0 cm)
- C: screwdriver (tool, 2.5 x 2.5 x 20.0 cm)
- D: notebook (book, 15.0 x 21.0 x 1.0 cm)
Request: "The TV remote has stopped working. I really need it working before the match."
Reasoning: a dead remote needs new AA batteries; the remote is what really matters.
Answer: {"fetch":["A","B"],"priority":["A"],"before":[],"never_pair":[]}

Objects on the table:
- A: seedlings tray (tray, 30.0 x 20.0 x 8.0 cm)
- B: trowel (tool, 8.0 x 4.0 x 30.0 cm)
- C: watering can (can, 20.0 x 12.0 x 25.0 cm)
- D: apple (fruit, 8.0 x 8.0 x 8.0 cm)
- E: magazine (magazine, 21.0 x 28.0 x 0.5 cm)
Request: "I'm planting the seedlings this afternoon, and I'll want a snack when I'm done."
Reasoning: planting needs the seedlings, the trowel and the watering can; the snack is the apple, for after the planting.
Answer: {"fetch":["A","B","C","D"],"priority":[],"before":[{"first":"A","then":"D"},{"first":"B","then":"D"},{"first":"C","then":"D"}],"never_pair":[]}

Objects on the table:
- A: stapler (stapler, 4.0 x 15.0 x 6.0 cm)
- B: paper stack (stack, 21.0 x 29.7 x 3.0 cm)
- C: pen (pen, 1.0 x 1.0 x 14.0 cm)
- D: ruler (ruler, 3.0 x 0.3 x 30.0 cm)
Request: "Pass me the stapler."
Reasoning: only the stapler is asked for.
Answer: {"fetch":["A"],"priority":[],"before":[],"never_pair":[]}"""

# (added after the evaluation; five fixed examples): v1's instructions followed by worked examples, none with a role of our scenes
COMPILE_PROMPT_V2 = COMPILE_PROMPT_V1 + "\n\n" + COMPILE_EXAMPLES_V2

# (for the demo only): v3 = v2 with new fetch and priority definitions, a "Thinking:" line before the answer, and worked
# examples with thinking. Free text, so no JSON schema: only the JSON after "Answer:" is parsed (``parse_v3``).
COMPILE_PROMPT_V3 = """You turn a person's spoken request to a home robot into a structured intent.
The robot fetches objects from a round table and brings them to the person. It carries one or two objects per trip.

Fill four fields, using only the ids of the objects listed:
- fetch: the objects the person asks for, and the objects the purpose clearly needs. Leave out everything else.
- priority: the object the request is for (the point of it), unless the person says another matters most. Empty only when nothing stands out.
- before: ordering rules the person states, each as {"first": x, "then": y}: x must be delivered before y is carried. Empty if no order is stated.
- never_pair: pairs the person says must not be carried together, each as {"a": x, "b": y}. Empty if none.

Every id in priority, before and never_pair must also be in fetch.

Before the answer, think in three short steps and write them on one line starting "Thinking:":
1. What is the person trying to do?
2. What does that need? For example: swallowing a tablet or capsule needs a drink; lighting candles needs matches; a dead remote needs batteries.
3. Which object is the point of the request (the priority), and is any order stated or implied ("then", "after", "first")?
Then write a line starting "Answer:" followed by the JSON object with the keys fetch, priority, before and never_pair.

Examples (each has its own objects; ids belong to that example only):

Objects on the table:
- A: football (ball, 22.0 x 22.0 x 22.0 cm)
- B: bicycle pump (pump, 5.0 x 5.0 x 45.0 cm)
- C: ball needle (needle, 0.5 x 0.5 x 4.0 cm)
- D: skipping rope (rope, 20.0 x 15.0 x 5.0 cm)
- E: umbrella (umbrella, 8.0 x 8.0 x 60.0 cm)
Request: "The kids want to play football but the ball is flat."
Thinking: they want to play football; a flat ball needs the pump, and the pump needs the needle; the football is the point; no order.
Answer: {"fetch":["A","B","C"],"priority":["A"],"before":[],"never_pair":[]}

Objects on the table:
- A: birthday candles (small box, 8.0 x 2.0 x 12.0 cm)
- B: matches (small box, 5.0 x 3.5 x 1.5 cm)
- C: cake knife (knife, 3.0 x 2.0 x 30.0 cm)
- D: paper plates (stack, 23.0 x 23.0 x 3.0 cm)
- E: scissors (scissors, 8.0 x 1.5 x 18.0 cm)
Request: "Let's light the birthday candles, and then we'll cut the cake."
Thinking: they want to light the candles, then cut the cake; lighting needs the candles and the matches, cutting needs the knife; two tasks, nothing stands out; lighting comes first ("then").
Answer: {"fetch":["A","B","C"],"priority":[],"before":[{"first":"A","then":"C"},{"first":"B","then":"C"}],"never_pair":[]}

Objects on the table:
- A: TV remote, takes AA batteries (remote, 5.0 x 2.5 x 20.0 cm)
- B: AA batteries (pack, 6.0 x 1.5 x 9.0 cm)
- C: screwdriver (tool, 2.5 x 2.5 x 20.0 cm)
- D: notebook (book, 15.0 x 21.0 x 1.0 cm)
Request: "The TV remote has stopped working. I really need it working before the match."
Thinking: they want the remote working; a dead remote needs new AA batteries; the remote is the point; no order.
Answer: {"fetch":["A","B"],"priority":["A"],"before":[],"never_pair":[]}

Objects on the table:
- A: seedlings tray (tray, 30.0 x 20.0 x 8.0 cm)
- B: trowel (tool, 8.0 x 4.0 x 30.0 cm)
- C: watering can (can, 20.0 x 12.0 x 25.0 cm)
- D: apple (fruit, 8.0 x 8.0 x 8.0 cm)
- E: magazine (magazine, 21.0 x 28.0 x 0.5 cm)
Request: "I'm planting the seedlings this afternoon, and I'll want a snack when I'm done."
Thinking: planting, then a snack; planting needs the seedlings, the trowel and the watering can, and the snack is the apple; the seedlings are the point; the snack comes after ("when I'm done").
Answer: {"fetch":["A","B","C","D"],"priority":["A"],"before":[{"first":"A","then":"D"},{"first":"B","then":"D"},{"first":"C","then":"D"}],"never_pair":[]}

Objects on the table:
- A: allergy capsules (jar, 5.0 x 5.0 x 8.0 cm)
- B: water bottle (bottle, 7.0 x 7.0 x 22.0 cm)
- C: sunglasses (case, 16.0 x 6.0 x 4.0 cm)
- D: house keys (keys, 8.0 x 3.0 x 1.0 cm)
- E: newspaper (paper, 30.0 x 20.0 x 1.0 cm)
Request: "My dad has to take his allergy capsule."
Thinking: he wants to take a capsule; swallowing a capsule needs a drink, so the water bottle; the capsule is the point of the request.
Answer: {"fetch":["A","B"],"priority":["A"],"before":[],"never_pair":[]}

Objects on the table:
- A: throat lozenges (small box, 7.0 x 2.0 x 9.0 cm)
- B: novel (book, 13.0 x 20.0 x 3.0 cm)
- C: glasses case (case, 16.0 x 6.0 x 4.0 cm)
- D: mug (mug, 9.0 x 9.0 x 10.0 cm)
- E: pencil (pencil, 0.8 x 0.8 x 17.0 cm)
Request: "My throat is sore. Get my lozenges, and something to read once I'm settled."
Thinking: relief for a sore throat, then reading; lozenges need nothing else; the lozenges are the point; reading comes after.
Answer: {"fetch":["A","B"],"priority":["A"],"before":[{"first":"A","then":"B"}],"never_pair":[]}

Objects on the table:
- A: stapler (stapler, 4.0 x 15.0 x 6.0 cm)
- B: paper stack (stack, 21.0 x 29.7 x 3.0 cm)
- C: pen (pen, 1.0 x 1.0 x 14.0 cm)
- D: ruler (ruler, 3.0 x 0.3 x 30.0 cm)
Request: "Pass me the stapler."
Thinking: they want the stapler; only the stapler is asked for; the stapler is the point; no order.
Answer: {"fetch":["A"],"priority":["A"],"before":[],"never_pair":[]}"""

PROMPTS = {"v1": COMPILE_PROMPT_V1, "v2": COMPILE_PROMPT_V2, "v3": COMPILE_PROMPT_V3}


def compile_schema(ids: list[str]) -> dict:
    ref = {"type": "string", "enum": ids}
    return {"type": "object", "additionalProperties": False, "required": ["fetch", "priority", "before", "never_pair"],
            "properties": {"fetch": {"type": "array", "items": ref}, "priority": {"type": "array", "items": ref},
                           "before": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["first", "then"],
                                                                 "properties": {"first": ref, "then": ref}}},
                           "never_pair": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["a", "b"],
                                                                     "properties": {"a": ref, "b": ref}}}}}


def parse_intent(content: str | None, slot_of: dict) -> tuple[Intent | None, str | None]:
    """The answer as an ``Intent`` in slots, or (None, the reason it fails the check)."""
    try:
        data = json.loads(content or "")
        fetch = [slot_of[x] for x in data["fetch"]]
        priority = [slot_of[x] for x in data["priority"]]
        before = [(slot_of[r["first"]], slot_of[r["then"]]) for r in data["before"]]
        never = [(slot_of[r["a"]], slot_of[r["b"]]) for r in data["never_pair"]]
    except (ValueError, KeyError, TypeError) as error:
        return None, f"the answer is not valid JSON of the schema ({type(error).__name__})"
    if not fetch:
        return None, "fetch is empty"
    fs = set(fetch)
    if not set(priority) <= fs:
        return None, "every priority object must be in fetch"
    if any(x == y or x not in fs or y not in fs for x, y in before + never):
        return None, "before and never_pair must name two different objects that are both in fetch"
    return Intent.make(fetch, dict.fromkeys(priority, PRIORITY_WEIGHT), before, never), None


@dataclass
class Compiled:
    intent: Intent
    fallback: bool # the empty intent after two failed checks
    retried: bool
    keys: list
    answer: str | None


def empty_intent(found) -> Intent:
    """The fallback: no intent (every object, unit weights, no constraint), i.e. the plain errand planner."""
    return Intent.make(found)


def compile_intent(client: LLMClient, instruction: str, names: list[str], found, roles: dict, prompt: str = "v1") -> Compiled:
    ids = ids_for(found)
    slot_of = {v: k for k, v in ids.items()}
    schema = compile_schema([ids[k] for k in sorted(found)])
    user = "Objects on the table:\n" + "\n".join(object_lines(names, found, roles)) + f'\n\nRequest: "{instruction}"'
    messages = [{"role": "system", "content": PROMPTS[prompt]}, {"role": "user", "content": user}]
    first = client.ask(messages, "intent", schema)
    intent, error = parse_intent(first.get("content"), slot_of)
    if intent is not None:
        return Compiled(intent, False, False, [first["key"]], first.get("content"))
    retry = [*messages, {"role": "assistant", "content": first.get("content") or ""},
             {"role": "user", "content": f"That answer fails the check: {error}. Answer again."}]
    second = client.ask(retry, "intent", schema)
    intent, error = parse_intent(second.get("content"), slot_of)
    if intent is not None:
        return Compiled(intent, False, True, [first["key"], second["key"]], second.get("content"))
    return Compiled(empty_intent(found), True, True, [first["key"], second["key"]], second.get("content"))


def intent_json(intent: Intent) -> dict:
    return {"fetch": sorted(intent.fetch), "priority": sorted(intent.priority), "weights": [list(w) for w in intent.weights],
            "before": sorted(map(list, intent.before)), "never_pair": sorted(sorted(p) for p in intent.never_pair)}


CHOOSE_PROMPT_V1 = """You plan a home robot's errand. The robot fetches objects from a round table and brings them to a person about 4 m away.
It carries one or two objects per trip. Each trip listed is one option, and each option can be tried only once.
A trip can fail: an object may be knocked over or dropped (lost for this errand) or left standing on the table (another trip may take it).
The odds and times given are the robot's own estimates.

Choose the next trip that best does what the person asked, as quickly as possible, or "stop" when nothing more should be fetched.
Bring only what the person asked for, respect any order or pairing they asked for, and protect what they said matters most."""

CHOOSE_PROMPTS = {"v1": CHOOSE_PROMPT_V1}


def trip_lines(est, options, standing, ids: dict) -> tuple[list[str], dict]:
    """Each open trip as the LLM sees it (from the planner's own outcomes), and trip id -> option."""
    lines, by_id = [], {}
    for n, o in enumerate(options, 1):
        tid = f"T{n}"
        by_id[tid] = o
        out = outcomes(est, o, "world", standing)
        seconds = sum(x.p * x.seconds for x in out)
        parts = []
        for k, how in ((o.i, "wrapped in the palm"), *(((o.j, "held by three fingers"),) if o.kind == "pair" else ())):
            if o.kind == "single":
                how = "alone"
            delivered = sum(x.p for x in out if k in x.delivered_objects)
            lost = sum(x.p for x in out if k in x.gone and k not in x.delivered_objects)
            parts.append(f"{ids[k]} ({how}): delivered {100 * delivered:.0f}%, lost {100 * lost:.0f}%, left standing {100 * (1 - delivered - lost):.0f}%")
        lines.append(f"- {tid}: {'; '.join(parts)}. Expected time {seconds:.0f} s.")
    return lines, by_id


@dataclass
class ChooserLog:
    calls: int = 0
    fallbacks: int = 0
    stops: int = 0
    keys: list = field(default_factory=list)


def llm_chooser(client: LLMClient, est, options, instruction: str, names: list[str], found, roles: dict, prompt: str = "v1",
                log: ChooserLog | None = None):
    """A chooser (``errand_value``'s interface) that asks the LLM at every decision; memoised per node, so a node is asked once."""
    ids = ids_for(found)
    memo: dict = {}
    log = log if log is not None else ChooserLog()

    def choose(standing, closed):
        key = (frozenset(standing), frozenset(closed))
        if key in memo:
            return memo[key]
        live = open_options([o for o in options if o.key not in closed], standing)
        if not live:
            memo[key] = None
            return None
        lines, by_id = trip_lines(est, live, standing, ids)
        on_table = [ids[k] for k in sorted(found) if k in standing]
        gone = [ids[k] for k in sorted(found) if k not in standing]
        user = ("Objects:\n" + "\n".join(object_lines(names, found, roles)) + f'\n\nRequest: "{instruction}"\n\n'
                f"Still on the table: {', '.join(on_table) or 'none'}. No longer on the table (delivered or lost): {', '.join(gone) or 'none'}.\n\n"
                "Open trips:\n" + "\n".join(lines))
        schema = {"type": "object", "additionalProperties": False, "required": ["trip"],
                  "properties": {"trip": {"type": "string", "enum": [*by_id, "stop"]}}}
        answer = client.ask([{"role": "system", "content": CHOOSE_PROMPTS[prompt]}, {"role": "user", "content": user}], "next_trip", schema)
        log.calls += 1
        log.keys.append(answer["key"])
        try:
            trip = json.loads(answer.get("content") or "")["trip"]
            if trip != "stop" and trip not in by_id:
                raise KeyError(trip)
        except (ValueError, KeyError, TypeError):
            log.fallbacks += 1
            choice = greedy_rate(est, live, standing)
        else:
            choice = None if trip == "stop" else by_id[trip]
            log.stops += trip == "stop"
        memo[key] = choice
        return choice
    choose.log = log
    return choose
