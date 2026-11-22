"""Object positions on the round table, drawn per layout and seed at least a set spacing apart."""

from __future__ import annotations

import numpy as np

BAND = ((.42, .68), (-.45, .45))
DRAW_STREAM = 31
RESTARTS = 2000
CANDIDATES = 500 # points drawn per restart, in order; a restart ends when they run out


def draw_positions(n: int, layout: int, seed: int, spacing: float, restarts: int = RESTARTS, key: list[int] | None = None
                   ) -> list[tuple[float, float]] | None:
    """``n`` points in ``BAND`` at least ``spacing`` apart, or None if ``restarts`` restarts never place all ``n``. ``key`` replaces the generator's key; None is the round table's ``[seed, layout, DRAW_STREAM]``."""
    rng = np.random.default_rng([seed, layout, DRAW_STREAM] if key is None else key)
    (x0, x1), (y0, y1) = BAND
    for _ in range(restarts):
        points = np.c_[rng.uniform(x0, x1, CANDIDATES), rng.uniform(y0, y1, CANDIDATES)]
        kept: list[np.ndarray] = []
        for q in points:
            if not kept or np.min(np.hypot(*(np.asarray(kept) - q).T)) >= spacing:
                kept.append(q)
                if len(kept) == n:
                    return [(float(a), float(b)) for a, b in kept]
    return None

