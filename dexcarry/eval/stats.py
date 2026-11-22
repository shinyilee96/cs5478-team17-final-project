"""Confidence intervals used for every result: the Wilson interval for a rate, and a two-level cluster bootstrap
(10,000 resamples) for means and paired differences.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

RESAMPLES = 10_000
LEVEL = .95


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval of ``successes`` out of ``trials`` (95% at the default ``z``)."""
    if trials <= 0:
        return float("nan"), float("nan")
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return float(max(0., centre - half)), float(min(1., centre + half))


def _as_clusters(arms: Sequence[Sequence[Sequence[float]]], paired_units: bool) -> list[list[np.ndarray]]:
    """``arms[a][c]`` (arm, cluster) → ``clusters[c][a]``, as float arrays, checked."""
    if not arms or any(len(arm) != len(arms[0]) for arm in arms):
        raise ValueError("every arm needs the same clusters, in the same order")
    clusters = [[np.asarray(arm[c], dtype=float).ravel() for arm in arms] for c in range(len(arms[0]))]
    if not clusters:
        raise ValueError("no clusters")
    for c, units in enumerate(clusters):
        if any(len(u) == 0 for u in units):
            raise ValueError(f"cluster {c} has an arm without units")
        if paired_units and len({len(u) for u in units}) > 1:
            raise ValueError(f"cluster {c}: paired units need the same number in every arm")
    return clusters


def _mean_difference(means: np.ndarray) -> np.ndarray:
    return (means[..., 0] - means[..., 1]).mean(axis=-1)


def cluster_bootstrap(arms: Sequence[Sequence[Sequence[float]]], statistic: Callable[[np.ndarray], np.ndarray], *,
                      paired_units: bool = True, resamples: int = RESAMPLES, level: float = LEVEL, seed: int = 0) -> dict:
    """Two-level cluster bootstrap of ``statistic``.

    ``arms[a][c]`` holds the unit outcomes (e.g. 0/1 per pickup) of arm ``a`` in cluster ``c``.
    ``statistic`` maps per-cluster arm means of shape (..., clusters, arms) to shape (...); it is applied
    to the data as observed for the estimate and to every resample for the interval.

    Returns the estimate, the percentile interval at ``level``, the bootstrap standard error and the sizes.
    """
    clusters = _as_clusters(arms, paired_units)
    n_clusters, n_arms = len(clusters), len(arms)
    observed = np.array([[u.mean() for u in units] for units in clusters])
    rng = np.random.default_rng(seed)
    outer = rng.integers(n_clusters, size=(resamples, n_clusters))
    means = np.empty((resamples, n_clusters, n_arms))
    # Every slot of every resample that drew cluster c gets its own draw of c's units.
    for c, units in enumerate(clusters):
        slots = np.nonzero(outer == c)
        k = len(slots[0])
        if k == 0:
            continue
        if paired_units:
            index = rng.integers(len(units[0]), size=(k, len(units[0])))
            for a, u in enumerate(units):
                means[slots[0], slots[1], a] = u[index].mean(axis=1)
        else:
            for a, u in enumerate(units):
                means[slots[0], slots[1], a] = u[rng.integers(len(u), size=(k, len(u)))].mean(axis=1)
    values = np.asarray(statistic(means), dtype=float)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    return {"estimate": float(statistic(observed)), "low": float(low), "high": float(high), "se": float(values.std(ddof=1)),
            "level": level, "resamples": resamples, "clusters": n_clusters,
            "units": int(sum(len(u) for units in clusters for u in units))}


def paired_difference(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], **kwargs) -> dict:
    """Arm A minus arm B, the mean over clusters of the per-cluster difference of unit means, with the
    two-level cluster bootstrap. ``a[c]`` and ``b[c]`` are the unit outcomes of cluster ``c``."""
    return cluster_bootstrap([a, b], _mean_difference, **kwargs)


def cluster_mean(a: Sequence[Sequence[float]], **kwargs) -> dict:
    """One arm: the mean over clusters of the per-cluster unit mean, with the same bootstrap."""
    return cluster_bootstrap([a], lambda means: means[..., 0].mean(axis=-1), **kwargs)
