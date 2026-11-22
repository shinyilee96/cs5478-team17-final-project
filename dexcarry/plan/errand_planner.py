"""The errand planners: ``greedy_rate`` picks the best trip for now; ``failure_aware`` plans the whole errand,
including what each failure leaves for later trips.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
from dataclasses import dataclass

from ..sim.transport import MotionProfile, _segment


@functools.cache
def drive_seconds(distance: float) -> float:
    """Duration of a base drive of ``distance`` metres on the default profile."""
    m = MotionProfile(distance=distance)
    t, _ = _segment(distance, m.speed, m.acceleration, m.braking, 1e-3)
    return float(t[-1])


@dataclass(frozen=True)
class Outcome:
    p: float
    delivered: int
    gone: frozenset
    seconds: float
    delivered_objects: frozenset | None = None # which objects arrived. Left None, it is derived when that is unambiguous (none
                                               # delivered, or every gone object delivered: a delivered object is always gone)

    def __post_init__(self):
        if self.delivered_objects is None and (self.delivered == 0 or self.delivered == len(self.gone)):
            object.__setattr__(self, "delivered_objects", frozenset(self.gone) if self.delivered else frozenset())


def _gain(o: Outcome, weights) -> float:
    """What an outcome is worth: its count of delivered objects (``weights`` None: every earlier run's arithmetic, unchanged), or the
    sum of the delivered objects' weights (an object not in ``weights`` weighs 1)."""
    if weights is None:
        return o.delivered
    if o.delivered_objects is None:
        raise ValueError("a weighted objective needs the outcome's delivered objects")
    return sum(weights.get(k, 1.) for k in o.delivered_objects)


@dataclass(frozen=True)
class Option:
    kind: str # "pair" or "single"
    i: int # the wrapped object's slot, or the single object's
    j: int | None = None # the three-finger object's slot
    codes: tuple = () # ((code or None, probability),...) of a pair; empty for a single
    drive: float = 1. # metres from the stance to the target table
    post: tuple | None = None # what follows code None, if not ``Estimates.post_pickup``: whole-trip practice's estimate
                              # already covers the set-down, so its option passes ((1., True, True),)
    traverse: float = 0. # (the any-pair counter): seconds between the grasps that ``Estimates.t_pickup`` does not hold (the
                         # drive to a second object beyond the long table's reach); paid by a pickup that gets past the first grasp
    needs_gone: frozenset = frozenset() # (the round table): objects that must be gone before the option opens (a standing blocker);
                                        # empty, every earlier run's, leaves every choice as it was
    rate: float | None = None # a single trip's own chance of delivering its object (measured odds); None is
                              # ``Estimates.single_rate``, every earlier run's

    @property
    def key(self) -> tuple:
        return ("single", self.i) if self.kind == "single" else ("pair", self.i, self.j)

    @property
    def objects(self) -> frozenset:
        return frozenset({self.i} if self.kind == "single" else {self.i, self.j})


@dataclass(frozen=True)
class Estimates:
    post_pickup: tuple # ((p, first delivered, second delivered),...) after a pickup that held both
    held_first: float # chance that a first object still held after a failed pickup is delivered
    single_rate: float # chance that a single trip delivers its object
    single_seconds: float # a single trip from the counter to its release, over ``single_distance``
    single_distance: float
    t_first: float # pickup time to the first lift
    t_pickup: float # pickup time to the end of the hold
    place_both: float # set-down of both, from arrival
    place_first: float # set-down of the wrapped object alone, from arrival
    consequence: dict # code -> what it leaves of (first, second): "standing", "fallen" or "held"
    stage: dict # code -> where it ends the pickup: "first", "between" or "pickup"
    recovery: dict | None = None # errand recovery's outcomes (``recovery_outcomes``); None is the long table's model
    knocks: Knocks | None = None # the injected knock model, in the world model only; None is every earlier run's


@dataclass(frozen=True)
class Knocks:
    """an injected knock model, an assumption (this simulator's table gives almost none). After a trip
    that fails at the table, each object still standing within the radius of the trip's objects is toppled with chance ``p``,
    independently. Severity: a toppled object is lost to the errand (``lost``), or it is stood up again at a cost of
    ``seconds`` each and stays (not ``lost``). Which outcomes failed at the table: a pair's pickup failure whose code is in
    ``pair_codes``; of a single trip's failures, the share ``single_table``."""
    p: float
    near: tuple # slot -> frozenset of the other slots within the radius (from the positions the planner has)
    lost: bool = True
    seconds: float = 0.
    pair_codes: frozenset = frozenset()
    single_table: float = 0.


def pair_codes(success: float, failures: dict) -> tuple:
    """A pair's outcome distribution: ``success`` from E or practice, what a failure leaves from pooled counts."""
    total = sum(failures.values())
    return ((None, float(success)), *((code, (1 - success) * n / total) for code, n in sorted(failures.items())))


def outcomes(est: Estimates, option: Option, kind: str = "world", standing=None) -> list[Outcome]:
    """``outcome_model.py._outcomes``. ``kind`` "world" is the consequence model, "success" MDP-success's. With ``est.knocks`` and
    ``standing`` given, the world model also holds what a failure at the table knocks over (``knocked``); without, as before."""
    out = _outcomes(est, option, kind)
    if est.knocks is None or kind != "world" or standing is None:
        return [o for o, _ in out]
    return knocked(est.knocks, out, option, frozenset(standing))


def _outcomes(est: Estimates, option: Option, kind: str) -> list[tuple[Outcome, bool]]:
    """``outcomes`` before any knock, each with whether it failed at the table (read only when ``est.knocks`` is set)."""
    drive = drive_seconds(option.drive)
    k = est.knocks
    if option.kind == "single":
        i, p = option.i, est.single_rate if option.rate is None else option.rate
        t = est.single_seconds - drive_seconds(est.single_distance) + drive + drive
        failed = frozenset({i}) if kind == "world" else frozenset()
        if k is None or not k.single_table:
            return [(Outcome(p, 1, frozenset({i}), t, frozenset({i})), False), (Outcome(1 - p, 0, failed, t), False)]
        return [(Outcome(p, 1, frozenset({i}), t, frozenset({i})), False), (Outcome((1 - p) * k.single_table, 0, failed, t), True),
                (Outcome((1 - p) * (1 - k.single_table), 0, failed, t), False)]
    i, j = option.i, option.j
    success_t = est.t_pickup + option.traverse + drive + est.place_both + drive
    out = []
    for code, p in option.codes:
        if code is None:
            out += [(Outcome(p * q, int(a) + int(b), frozenset({i, j}), success_t, _arrived(i, j, a, b)), False)
                    for q, a, b in (option.post or est.post_pickup)]
            continue
        t = {"first": est.t_first, "between": (est.t_first + est.t_pickup) / 2 + option.traverse,
             "pickup": est.t_pickup + option.traverse}[est.stage[code]]
        table = k is not None and code in k.pair_codes
        if est.recovery is not None and kind == "world":
            out += [(o, table) for o in recovery_outcomes(est, option, code, p, t)]
            continue
        first, second = est.consequence[code]
        gone = frozenset(([i] if first != "standing" else []) + ([j] if second != "standing" else []))
        if first == "held":
            held = t + drive + est.place_first + drive
            out += [(Outcome(p * est.held_first, 1, gone, held, frozenset({i})), table),
                    (Outcome(p * (1 - est.held_first), 0, gone, held), table)]
        else:
            out.append((Outcome(p, 0, gone, t), table))
    if kind == "success":
        win = sum(o.p for o, _ in out if o.delivered == 2)
        lose = [o for o, _ in out if o.delivered < 2]
        t_lose = sum(o.p * o.seconds for o in lose) / max(1e-12, sum(o.p for o in lose))
        return [(Outcome(win, 2, frozenset({i, j}), success_t, frozenset({i, j})), False),
                (Outcome(1 - win, 0, frozenset(), t_lose), False)]
    return out


def _arrived(i, j, a, b) -> frozenset:
    """The objects a pair delivered: the wrapped one ``i`` if ``a``, the second ``j`` if ``b``."""
    return frozenset(k for k, d in ((i, a), (j, b)) if d)


def knocked(k: Knocks, out: list[tuple[Outcome, bool]], option: Option, standing: frozenset) -> list[Outcome]:
    """each outcome that failed at the table, split over which of the standing objects near the trip's objects it topples
    (each with chance ``k.p``): lost ones join ``gone``; with severity "stood up again" they stay and cost ``k.seconds`` each."""
    near = frozenset().union(*(k.near[x] for x in option.objects)) & standing - option.objects
    result = []
    for o, table in out:
        if not table or not near or not k.p:
            result.append(o)
            continue
        for size in range(len(near) + 1):
            for hit in itertools.combinations(sorted(near), size):
                q = o.p * k.p ** size * (1 - k.p) ** (len(near) - size)
                if q > 0:
                    result.append(Outcome(q, o.delivered, o.gone | frozenset(hit), o.seconds, o.delivered_objects) if k.lost
                                  else Outcome(q, o.delivered, o.gone, o.seconds + size * k.seconds, o.delivered_objects))
    return result


def entry_states(est: Estimates, code: str) -> dict:
    """What a pickup failure ``code`` leaves for errand recovery: ``"retry"`` (step 3 runs) or ``"F+S"`` (first, second, each
    one of ``RECOVERY_STATES``), to probability. From the development runs (``est.recovery["entry"]``); a code they never saw
    takes the consequence table's (held stays held, fallen is gone, standing stands; held with standing is a retry once the first
    phase has ended, so a code of the second grasp)."""
    seen = est.recovery["entry"].get(code)
    if seen:
        return seen
    first, second = ({"fallen": "gone"}.get(s, s) for s in est.consequence[code])
    retry = (first, second) == ("held", "standing") and code.startswith("SECOND_")
    return {"retry": 1.} if retry else {f"{first}+{second}": 1.}


def recovery_outcomes(est: Estimates, option: Option, code: str, p: float, t: float) -> list[Outcome]:
    """a pair's pickup failure ``code`` (probability ``p``, ``t`` seconds into the trip) followed by errand recovery, as
    its development runs measured it (``est.recovery``):
    - ``retry``: the second grasp again (``retry.p``, ``retry.seconds``); held both, the trip ends as a pair trip that held both
      (``post_pickup``); otherwise it leaves ``retry.after``'s states, which steps 1 and 4 take on;
    - step 1: each held object delivered with ``deliver.p``, one carry of ``deliver.seconds`` plus ``deliver.legs`` drives;
    - step 4: each standing object retaken alone, delivered with ``retake.p``, ``retake.seconds`` plus ``retake.legs`` drives;
      a failed retake leaves it standing with ``retake.stays``, else it is gone.
    Drives are of the option's own length, as every other outcome's. Objects delivered or no longer standing are gone."""
    rec, drive, i, j = est.recovery, drive_seconds(option.drive), option.i, option.j
    out = []
    for entry, q in entry_states(est, code).items():
        if entry != "retry":
            out += _recovery_steps(rec, entry, p * q, t, i, j, drive)
            continue
        s, t3 = rec["retry"]["p"], t + rec["retry"]["seconds"]
        out += [Outcome(p * q * s * r, int(a) + int(b), frozenset({i, j}), t3 + drive + est.place_both + drive, _arrived(i, j, a, b))
                for r, a, b in est.post_pickup]
        for after, r in rec["retry"]["after"].items():
            out += _recovery_steps(rec, after, p * q * (1 - s) * r, t3, i, j, drive)
    return out


def _recovery_steps(rec: dict, entry: str, p: float, t: float, i: int, j: int, drive: float) -> list[Outcome]:
    """Steps 1 and 4 of errand recovery from ``entry`` ("F+S"): branches of (probability, delivered, gone, seconds, delivered objects)."""
    states = dict(zip((i, j), entry.split("+"), strict=True))
    held = [k for k, s in states.items() if s == "held"]
    gone = frozenset(k for k, s in states.items() if s != "standing")
    deliver, retake = rec["deliver"], rec["retake"]
    branches = [(1., 0, gone, t + (deliver["seconds"] + deliver["legs"] * drive if held else 0.), frozenset())]
    for h in held:
        branches = [(bp * x, bd + d, bg, bt, bo | {h} if d else bo) for bp, bd, bg, bt, bo in branches
                    for x, d in ((deliver["p"], 1), (1 - deliver["p"], 0))]
    for k in (k for k, s in states.items() if s == "standing"):
        cost = retake["seconds"] + retake["legs"] * drive
        fail = 1 - retake["p"]
        branches = [b for bp, bd, bg, bt, bo in branches for b in ((bp * retake["p"], bd + 1, bg | {k}, bt + cost, bo | {k}),
                                                                   (bp * fail * retake["stays"], bd, bg, bt + cost, bo),
                                                                   (bp * fail * (1 - retake["stays"]), bd, bg | {k}, bt + cost, bo))]
    return [Outcome(p * bp, bd, bg, bt, bo) for bp, bd, bg, bt, bo in branches if p * bp > 0]


def open_options(options: list[Option], standing) -> list[Option]:
    """The options whose objects all stand and whose blockers (``needs_gone``) have all gone, pairs first, each kind in its sort
    order."""
    standing = frozenset(standing)
    live = [o for o in options if o.objects <= standing and not (o.needs_gone & standing)]
    return sorted((o for o in live if o.kind == "pair"), key=lambda o: (o.i, o.j)) + sorted((o for o in live if o.kind == "single"), key=lambda o: o.i)


def two_trips(est, options, standing) -> Option | None:
    singles = [o for o in open_options(options, standing) if o.kind == "single"]
    return singles[0] if singles else None


def greedy_success(est, options, standing) -> Option | None:
    live = open_options(options, standing)
    pairs = [o for o in live if o.kind == "pair"]
    if pairs:
        return max(pairs, key=lambda o: (sum(x.p for x in outcomes(est, o) if x.delivered == 2), -live.index(o)))
    return live[0] if live else None # every single has the same estimate; ties to the lowest slot, as the outcome model's


def greedy_rate(est, options, standing, weights=None) -> Option | None:
    """Each open option's one-trip rate: E[delivered] over E[seconds]."""
    live = open_options(options, standing)

    def rate(o):
        out = outcomes(est, o)
        return sum(x.p * _gain(x, weights) for x in out) / sum(x.p * x.seconds for x in out), -live.index(o)
    return max(live, key=rate) if live else None


def greedy_blind(est, options, standing) -> Option | None:
    """a failure-blind greedy. Each open option's nominal rate, as if the trip succeeds: its objects over the seconds of
    a trip that succeeds (``_outcomes``' success times); no odds are read. Ties to the outcome model's order, as ``greedy_rate``."""
    live = open_options(options, standing)

    def rate(o):
        drive = drive_seconds(o.drive)
        if o.kind == "single":
            return 1 / (est.single_seconds - drive_seconds(est.single_distance) + drive + drive), -live.index(o)
        return 2 / (est.t_pickup + o.traverse + drive + est.place_both + drive), -live.index(o)
    return max(live, key=rate) if live else None


def subsets(standing) -> list[frozenset]:
    """Every set of standing objects, smallest first."""
    items = sorted(standing)
    return [frozenset(c) for size in range(len(items) + 1) for c in itertools.combinations(items, size)]


def _values(est, options, policy, states, kind, weights=None):
    """Expected delivered and seconds from every state under ``policy``, self-loops summed
    as a geometric series."""
    ed, et = {}, {}
    for s in states:
        if s not in policy:
            ed[s] = et[s] = 0.
            continue
        out = outcomes(est, policy[s], kind, s)
        stay = sum(o.p for o in out if not o.gone)
        ed[s] = sum(o.p * (_gain(o, weights) + (ed[s - o.gone] if o.gone else 0)) for o in out) / (1 - stay)
        et[s] = sum(o.p * (o.seconds + (et[s - o.gone] if o.gone else 0)) for o in out) / (1 - stay)
    return ed, et


def solve(est, options, standing, kind, iterations=100, weights=None, allow=None) -> dict:
    """``outcome_model.py.mdp_ratio`` over the given options: the policy (state to option) with the best E[delivered] /
    E[seconds] from ``standing``, by Dinkelbach's method. A state with no option left ends the errand. with ``weights``, the
    best E[sum of the weights of the objects delivered] / E[seconds], by the same method. ``allow(state, open options)`` narrows each
    state's choice (``protected``) before the ratio decides; None (every earlier run) offers every open option."""
    states = subsets(standing)
    start = states[-1]
    lam, policy = 0., None
    offered = {s: allow(s, open_options(options, s)) for s in states} if allow is not None else None
    for _ in range(iterations):
        value, new = {}, {}
        for s in states:
            best = None
            for o in (open_options(options, s) if offered is None else offered[s]):
                out = outcomes(est, o, kind, s)
                stay = sum(x.p for x in out if not x.gone)
                if stay >= 1 - 1e-12:
                    continue
                v = sum(x.p * (_gain(x, weights) - lam * x.seconds + (value[s - x.gone] if x.gone else 0)) for x in out) / (1 - stay)
                if best is None or v > best[0] + 1e-12:
                    best = (v, o)
            value[s] = best[0] if best else 0.
            if best:
                new[s] = best[1]
        ed, et = _values(est, options, new, states, kind, weights)
        ratio = ed[start] / et[start] if et[start] else 0.
        if policy == new or abs(ratio - lam) < 1e-12:
            return new
        lam, policy = ratio, new
    return policy


def mdp_success(est, options, standing, weights=None) -> Option | None:
    return solve(est, options, frozenset(standing), "success", weights=weights).get(frozenset(standing))


def failure_aware(est, options, standing, weights=None) -> Option | None:
    return solve(est, options, frozenset(standing), "world", weights=weights).get(frozenset(standing))


PROTECT_DELTA = .02 # (added after the evaluation): an option is "safe enough" for a priority object within 0.02 of its best chance


def delivery_chance(est, option: Option, k, standing) -> float:
    """the chance that one trip on ``option`` delivers object ``k`` (the world model, from ``standing``)."""
    return sum(x.p for x in outcomes(est, option, "world", standing) if k in x.delivered_objects)


def protected(est, live, standing, priority, delta: float = PROTECT_DELTA) -> list[Option]:
    """the lexicographic rule: while a priority object stands, only the open options whose chance of delivering it is within ``delta``
    of the best open option for it (alone or in its safest pair). With several priority objects standing, an option safe enough for any
    of them is kept. A priority object no open option holds sets no rule; with none standing, every open option is kept."""
    standing = frozenset(standing)
    live = open_options(live, standing)
    keep = None
    for p in sorted(frozenset(priority) & standing):
        chance = {o.key: delivery_chance(est, o, p, standing) for o in live if p in o.objects}
        if not chance:
            continue
        best = max(chance.values())
        near = {key for key, c in chance.items() if c >= best - delta - 1e-12}
        keep = near if keep is None else keep | near
    return live if keep is None else [o for o in live if o.key in keep]


def mdp_protect(est, options, standing, weights=None, priority=frozenset(), delta: float = PROTECT_DELTA) -> Option | None:
    """the "protect" arm: priority first, then the weighted rate. Each state's choice is narrowed by ``protected``, then solved
    exactly as ``failure_aware`` (the world model, Dinkelbach), so later states follow the same rule."""
    def allow(s, live):
        return protected(est, live, s, priority, delta)
    return solve(est, options, frozenset(standing), "world", weights=weights, allow=allow).get(frozenset(standing))


def option_values(est, options, standing, kind="world", weights=None) -> list[dict]:
    """For the trip's record only, never for a choice: what each open option is worth now, if the errand then goes on by the
    policy solved from this state: expected objects, seconds and their ratio, and its odds of delivering both. with
    ``weights`` the policy is the weighted one, and each row adds ``value`` (expected weight delivered) and ``value_rate``."""
    standing = frozenset(standing)
    policy = solve(est, options, standing, kind, weights=weights) or {}
    ed, et = _values(est, options, policy, subsets(standing), kind)
    if weights is not None:
        ew, _ = _values(est, options, policy, subsets(standing), kind, weights)
    rows = []
    for o in open_options(options, standing):
        out = outcomes(est, o, kind, standing)
        stay = sum(x.p for x in out if not x.gone)
        if stay >= 1 - 1e-12:
            continue
        objects = sum(x.p * (x.delivered + (ed[standing - x.gone] if x.gone else 0)) for x in out) / (1 - stay)
        seconds = sum(x.p * (x.seconds + (et[standing - x.gone] if x.gone else 0)) for x in out) / (1 - stay)
        rows.append({"option": list(o.key), "objects": objects, "seconds": seconds, "rate": objects / seconds if seconds else 0.,
                     "both": sum(x.p for x in out if x.delivered == 2)})
        if weights is not None:
            value = sum(x.p * (_gain(x, weights) + (ew[standing - x.gone] if x.gone else 0)) for x in out) / (1 - stay)
            rows[-1].update(value=value, value_rate=value / seconds if seconds else 0.)
    return rows


PRIORITY_WEIGHT = 3. # a priority object is worth three ordinary ones; fixed before any instruction


@dataclass(frozen=True)
class Intent:
    """what an instruction asks of the errand, in the errand's slot indices.
    - ``fetch``: the objects to bring; every other object is left where it stands;
    - ``weights``: ((slot, weight),...), sorted; an object not listed weighs 1 (``PRIORITY_WEIGHT`` for a priority object);
    - ``before``: (x, y): x must be gone (delivered or fallen) before any trip holding y starts, so x and y never share a trip;
    - ``never_pair``: frozensets {a, b} that are never carried together."""
    fetch: frozenset
    weights: tuple = ()
    before: frozenset = frozenset()
    never_pair: frozenset = frozenset()

    @classmethod
    def make(cls, fetch, weights=None, before=(), never_pair=()) -> Intent:
        return cls(frozenset(int(k) for k in fetch), tuple(sorted((int(k), float(w)) for k, w in dict(weights or {}).items())),
                   frozenset((int(x), int(y)) for x, y in before), frozenset(frozenset(int(k) for k in p) for p in never_pair))

    def weight(self, k) -> float:
        return dict(self.weights).get(k, 1.)

    def weights_dict(self) -> dict:
        """The planner's ``weights``: every fetched object's weight."""
        return {k: self.weight(k) for k in sorted(self.fetch)}

    @property
    def priority(self) -> frozenset:
        """The fetched objects of the highest weight, if that weight is above 1; else none."""
        top = max((self.weight(k) for k in self.fetch), default=1.)
        return frozenset(k for k in self.fetch if self.weight(k) == top) if top > 1. else frozenset()


def apply_intent(options, intent: Intent) -> list[Option]:
    """the options an intent allows: every object in ``fetch``, not a forbidden pair, and each "x before y" added to the
    ``needs_gone`` of every option holding y (an x not fetched is ignored; a pair holding both never opens)."""
    kept = []
    for o in options:
        if not o.objects <= intent.fetch or (o.kind == "pair" and o.objects in intent.never_pair):
            continue
        first = frozenset(x for x, y in intent.before if y in o.objects and x in intent.fetch and x != y)
        kept.append(dataclasses.replace(o, needs_gone=o.needs_gone | first) if first else o)
    return kept


PLANNERS = {"two_trips": two_trips, "greedy_success": greedy_success, "greedy_rate": greedy_rate,
            "mdp_success": mdp_success, "failure_aware": failure_aware}
# the outcome model's planners above; the errand also runs these
ALL_PLANNERS = {**PLANNERS, "greedy_blind": greedy_blind}
