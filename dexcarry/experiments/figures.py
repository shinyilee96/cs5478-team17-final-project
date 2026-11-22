"""Figure 2: (a) the delivery-rate gain over the single-object baseline by distance, (b) failure-aware against
greedy planning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MUTED, ROUND, LONG, GRID, ACCENT = "#6b7280", "#2563eb", "#b45309", "#d1d5db", "#2563eb"
DISTANCES = (1., 2., 4., 8., 12., 16.)


def distance_figure(points: dict, out: Path) -> Path:
    """``points[table][distance] = (effect %, low %, high %)`` for table "Round table" and "Long table"."""
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for (name, colour), shift in zip((("Round table", ROUND), ("Long table", LONG)), (1.05, 0.95)):
        data = points[name]
        xs = [d * shift for d in DISTANCES]
        ys = [data[d][0] for d in DISTANCES]
        ax.plot(xs, ys, color=colour, lw=1.2, alpha=0.55, zorder=2)
        for x, d in zip(xs, DISTANCES):
            ax.plot([x, x], data[d][1:3], color=colour, lw=1.6, solid_capstyle="round", zorder=3)
        ax.scatter(xs, ys, s=18, color=colour, edgecolor="white", lw=0.8, zorder=4, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(DISTANCES), [f"{d:g}" for d in DISTANCES])
    ax.minorticks_off()
    ax.set_xlim(0.75, 21)
    ax.set_xlabel("Drive distance (m, log scale)")
    ax.set_ylim(-20, 58)
    ax.set_yticks([-20, 0, 20, 40])
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_ylabel("DexCarry vs. baseline (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_distance.pdf")
    fig.savefig(out / "fig_distance_preview.png", dpi=200)
    plt.close(fig)
    return out / "fig_distance.pdf"


def planner_figure(bars: list[tuple[float, float, float]], out: Path) -> Path:
    """``bars``: (effect %, low %, high %) for random layouts, then partner scarcity (privileged success probabilities)."""
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    ys = list(range(len(bars)))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for y, (e, lo, hi) in zip(ys, bars):
        ax.plot([lo, hi], [y, y], color=ACCENT, lw=1.8, solid_capstyle="round", zorder=2)
        ax.scatter([e], [y], s=22, color=ACCENT, edgecolor="white", lw=0.8, zorder=3)
    ax.set_yticks(ys, ["Set-up 1:\nrandom scenes", "Set-up 2:\nbuilt to trap\ngreedy*"], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-12, 16)
    ax.set_ylim(-0.6, len(bars) - 0.4)
    ax.set_xticks([-10, 0, 10], ["−10", "0", "+10"])
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Failure-aware vs. greedy (%)")
    fig.tight_layout(pad=0.3)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_lookahead.pdf")
    fig.savefig(out / "fig_lookahead_preview.png", dpi=200)
    plt.close(fig)
    return out / "fig_lookahead.pdf"
