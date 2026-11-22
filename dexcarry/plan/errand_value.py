"""The exact value of an errand in the planner's model, and the measures used for LLM requests."""

from __future__ import annotations

from dataclasses import dataclass, field

from .errand_planner import Intent, Option, apply_intent, outcomes

VIOLATIONS = ("forbidden", "order", "unrequested")


@dataclass(frozen=True)
class Score:
    """The correct intent, as the scorer reads it. ``weights``: object -> weight, 0 for an object not asked for."""
    weights: dict
    requested: frozenset
    priority: frozenset
    before: frozenset = frozenset()
    never_pair: frozenset = frozenset()

    @classmethod
    def of(cls, intent: Intent, objects) -> Score:
        return cls({k: (intent.weight(k) if k in intent.fetch else 0.) for k in objects}, intent.fetch, intent.priority,
                   frozenset((x, y) for x, y in intent.before if x in intent.fetch and x != y), intent.never_pair)

    def broken(self, option: Option, standing: frozenset) -> tuple:
        """The kinds of violation a trip on ``option`` makes from ``standing``."""
        out = []
        if option.kind == "pair" and option.objects in self.never_pair:
            out.append("forbidden")
        if any(y in option.objects and x in standing for x, y in self.before):
            out.append("order")
        if not option.objects <= self.requested:
            out.append("unrequested")
        return tuple(out)


@dataclass
class Value:
    gain: float = 0. # E[sum of the correct weights delivered]
    seconds: float = 0. # E[seconds]
    delivered: dict = field(default_factory=dict) # object -> P(delivered)
    violations: dict = field(default_factory=lambda: dict.fromkeys(VIOLATIONS, 0.)) # kind -> expected count of such trips
    any_violation: float = 0. # P(at least one violating trip)
    trips: float = 0. # E[trips]


def replan_value(est, options, standing, choose, score: Score, kind: str = "world") -> Value:
    """The exact expectations of the errand that ``choose(standing, closed)`` (an option or None) runs from ``standing``, with
    ``options`` its start options (every option the errand has; ``choose`` decides which it may use)."""
    objects = sorted(frozenset(standing) | frozenset(k for o in options for k in o.objects))
    memo: dict = {}

    def node(standing: frozenset, closed: frozenset) -> Value:
        key = (standing, closed)
        if key in memo:
            return memo[key]
        option = choose(standing, closed)
        value = Value(delivered=dict.fromkeys(objects, 0.))
        if option is not None:
            if option.key in closed:
                raise ValueError(f"the chooser chose {option.key}, which this errand has already used")
            broken = score.broken(option, standing)
            closed2 = closed | {option.key}
            value.trips = 1.
            for v in broken:
                value.violations[v] += 1.
            later_violation = 0.
            for x in outcomes(est, option, kind, standing):
                child = node(standing - x.gone, closed2)
                value.gain += x.p * (sum(score.weights.get(k, 0.) for k in x.delivered_objects) + child.gain)
                value.seconds += x.p * (x.seconds + child.seconds)
                value.trips += x.p * child.trips
                for k in objects:
                    value.delivered[k] += x.p * ((k in x.delivered_objects) + child.delivered[k])
                for v in VIOLATIONS:
                    value.violations[v] += x.p * child.violations[v]
                later_violation += x.p * child.any_violation
            value.any_violation = 1. if broken else later_violation
        memo[key] = value
        return value

    return node(frozenset(standing), frozenset())


def measures(value: Value, score: Score) -> dict:
    """the measures of one errand from its exact value:
    - ``rate``: the weighted rate, 60 E[weight delivered] / E[seconds], per minute;
    - ``priority_lost``: the chance a priority object is not delivered (the mean over the priority objects; None without one);
    - ``share``: E[requested objects delivered] / requested;
    - ``violations``: expected violating trips (all kinds), and ``any_violation``."""
    requested = sorted(score.requested)
    return {"rate": 60 * value.gain / value.seconds if value.seconds else 0.,
            "priority_lost": (sum(1 - value.delivered[k] for k in score.priority) / len(score.priority)) if score.priority else None,
            "share": sum(value.delivered[k] for k in requested) / len(requested) if requested else None,
            "violations": sum(value.violations.values()), "any_violation": value.any_violation,
            "by_kind": dict(value.violations), "seconds": value.seconds, "gain": value.gain, "trips": value.trips}


def planner_chooser(planner, est, options, intent: Intent | None = None, weights: dict | None = None):
    """A chooser that runs ``planner`` as the errand does: on the options the intent allows (``apply_intent``; all with None), minus
    the closed ones, with ``weights`` (None: the planner's own, unweighted objective)."""
    allowed = apply_intent(options, intent) if intent is not None else list(options)
    extra = {} if weights is None else {"weights": weights}

    def choose(standing, closed):
        return planner(est, [o for o in allowed if o.key not in closed], standing, **extra)
    return choose

