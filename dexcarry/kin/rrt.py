"""RRT-Connect in joint space with shortcut smoothing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

CollisionFn = Callable[[np.ndarray], bool]


def interpolate(q1: np.ndarray, q2: np.ndarray, step: float) -> np.ndarray:
    """Points from q1 to q2 (inclusive) spaced at most ``step`` apart in the infinity norm."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    n = max(int(np.ceil(np.abs(q2 - q1).max() / step)), 1)
    return np.linspace(q1, q2, n + 1)


def edge_free(q1: np.ndarray, q2: np.ndarray, collision_fn: CollisionFn, step: float) -> bool:
    for q in interpolate(q1, q2, step)[1:]:
        if collision_fn(q):
            return False
    return True


class _Tree:
    def __init__(self, root: np.ndarray):
        self.nodes = [np.asarray(root, dtype=float)]
        self.parents = [-1]

    def nearest(self, q: np.ndarray) -> int:
        arr = np.stack(self.nodes)
        return int(np.argmin(np.linalg.norm(arr - q, axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(np.asarray(q, dtype=float))
        self.parents.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, idx: int) -> list[np.ndarray]:
        out = []
        while idx != -1:
            out.append(self.nodes[idx])
            idx = self.parents[idx]
        return out


def _extend(tree: _Tree, q_target: np.ndarray, collision_fn: CollisionFn, step: float,
            max_extension: float) -> tuple[str, int]:
    """Grow the tree toward q_target. Returns ('reached'|'advanced'|'trapped', node index)."""
    idx = tree.nearest(q_target)
    q_near = tree.nodes[idx]
    delta = q_target - q_near
    dist = np.linalg.norm(delta)
    if dist < 1e-9:
        return "reached", idx
    n_steps = max(int(np.ceil(min(dist, max_extension) / step)), 1)
    direction = delta / dist
    last = idx
    q_cur = q_near
    for _ in range(n_steps):
        remaining = np.linalg.norm(q_target - q_cur)
        q_new = q_target if remaining <= step else q_cur + direction * step
        if collision_fn(q_new):
            return ("advanced" if last != idx else "trapped"), last
        last = tree.add(q_new, last)
        q_cur = q_new
        if np.linalg.norm(q_target - q_cur) < 1e-9:
            return "reached", last
    return "advanced", last


def rrt_connect(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    collision_fn: CollisionFn,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    step: float = 0.05,
    max_extension: float = 0.5,
    max_iters: int = 3000,
    goal_bias: float = 0.1,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray] | None:
    """Bidirectional RRT-Connect (Kuffner and LaValle, 2000).

    Returns a list of configurations from start to goal, or None. The start and goal are checked
    for collision first; a direct straight-line connection is tried before growing any tree.
    """
    rng = rng or np.random.default_rng(0)
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if collision_fn(q_start) or collision_fn(q_goal):
        return None
    if edge_free(q_start, q_goal, collision_fn, step):
        return [q_start, q_goal]

    ta, tb = _Tree(q_start), _Tree(q_goal)
    swapped = False
    for _ in range(max_iters):
        q_rand = q_goal if rng.random() < goal_bias and not swapped else rng.uniform(lower, upper)
        status, idx_a = _extend(ta, q_rand, collision_fn, step, max_extension)
        if status != "trapped":
            q_new = ta.nodes[idx_a]
            status_b, idx_b = _extend(tb, q_new, collision_fn, step, max_extension)
            if status_b == "reached":
                path_a = ta.path_to_root(idx_a)[::-1]
                path_b = tb.path_to_root(idx_b)
                path = path_a + path_b[1:]
                if swapped:
                    path = path[::-1]
                return [np.asarray(q) for q in path]
        ta, tb = tb, ta
        swapped = not swapped
    return None


def shortcut(path: list[np.ndarray], collision_fn: CollisionFn, *, step: float = 0.05,
             n_iters: int = 100, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Random shortcutting: repeatedly try to replace a sub-path by a straight segment."""
    rng = rng or np.random.default_rng(0)
    path = [np.asarray(q, dtype=float) for q in path]
    if len(path) < 3:
        return path
    for _ in range(n_iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.integers(0, len(path), size=2))
        if j - i < 2:
            continue
        if edge_free(path[i], path[j], collision_fn, step):
            path = path[: i + 1] + path[j:]
    return path


def densify_path(path: list[np.ndarray], step: float) -> list[np.ndarray]:
    """Resample a path so consecutive configurations are at most ``step`` apart."""
    out = [np.asarray(path[0], dtype=float)]
    for q1, q2 in zip(path[:-1], path[1:]):
        out.extend(interpolate(q1, q2, step)[1:])
    return out
