"""Which catalog object a cluster is (reported only; the experiments take identity from the simulator)."""

from __future__ import annotations

import numpy as np

from .estimate import estimate

CLASS_TOLERANCE = .005


def extents_of(spec) -> np.ndarray:
    """Full sizes of a standing catalogue object: long side, short side, height."""
    hx, hy, hz = (2 * float(v) for v in spec.half_extents)
    return np.array([max(hx, hy), min(hx, hy), hz])


def visual_classes(catalog: dict, tolerance: float = CLASS_TOLERANCE) -> dict[str, str]:
    """Each object's class name. In name order, an object joins the first class all of whose members agree with it
    within ``tolerance`` on every extent, else
    starts its own; a class is named after its first member. Single linkage would chain objects that differ by more."""
    sizes = {n: extents_of(catalog[n]) for n in sorted(catalog)}
    classes: list[list[str]] = []
    for n in sizes:
        home = next((c for c in classes if all(np.max(np.abs(sizes[n] - sizes[m])) <= tolerance + 1e-12 for m in c)), None)
        if home is None:
            classes.append([n])
        else:
            home.append(n)
    return {n: c[0] for c in classes for n in c}


SHORTLIST = .012 # m: catalogue objects within this extent distance of a cluster are fitted


def identify_by_fit(cluster, catalog: dict, table_top: float, classes: dict[str, str] | None = None,
                    shortlist: float = SHORTLIST) -> tuple[str, float, int]:
    """The visual class of the catalogue model that best explains a cluster (``segment.Cluster``): objects within
    ``shortlist`` of its extents (at least the nearest) are each fitted by ``estimate.estimate``, and the lowest mean
    point-to-plane residual wins. Extents alone confuse objects closer than their measurement error, such as a round bottle and a box of its size. Returns (class, residual, number fitted)."""
    classes = classes or visual_classes(catalog)
    e = np.asarray(cluster.extents, float)
    distance = {n: float(np.max(np.abs(extents_of(catalog[n]) - e))) for n in catalog}
    nearest = min(distance, key=lambda n: (distance[n], n))
    candidates = sorted({n for n, d in distance.items() if d <= shortlist} | {nearest})
    fits = {n: estimate(cluster.points, cluster.centre, cluster.axes, catalog[n], table_top,
                        rng=np.random.default_rng(0)).residual for n in candidates}
    best = min(fits, key=lambda n: (fits[n], n))
    return classes[best], fits[best], len(candidates)
