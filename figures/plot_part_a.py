"""Render the redesigned Part A board-level figure from archived stats.

    python plot_part_a.py /path/to/prior_cog_sci
"""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


TERRA = "#C85A32"
INDIGO = "#275684"
SLATE = "#949CA6"
LIGHT = "#DCE0E5"
INK = "#1E232A"
MUTED = "#5A636E"
GRID = "#ECEEF2"


def swarm(vals, radius=0.024, pitch=0.068):
    offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4]
    ys = [0.0] * len(vals)
    assigned = []
    for i, v in sorted(enumerate(vals), key=lambda t: t[1]):
        shift = next(
            x * pitch for x in offsets
            if not any(abs(v - p) < radius and abs(x * pitch - y) < pitch * 0.98 for p, y in assigned)
        )
        ys[i] = 0.64 + shift
        assigned.append((v, shift))
    return ys


def fmt_pp(v):
    return f"{v * 100:+.1f}".replace("-", "\u2212")


def style_panel(ax, letter, title, subtitle):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#C5CBD3")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["left"].set_color("#C5CBD3")
    ax.spines["left"].set_linewidth(0.7)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.15, f"{letter}", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=TERRA, va="baseline")
    ax.text(0.075, 1.15, title, transform=ax.transAxes,
            fontsize=9.6, fontweight="bold", color=INK, va="baseline")
    ax.text(0.0, 1.045, subtitle, transform=ax.transAxes,
            fontsize=7.9, color=MUTED, va="baseline")


def plot_contrast(ax, stats, letter, title, subtitle):
    style_panel(ax, letter, title, subtitle)
    ax.spines["left"].set_visible(False)
    ax.set(xlim=(-0.40, 0.88), ylim=(0.0, 1.0), yticks=[])
    ax.set_xticks([-0.25, 0.0, 0.25, 0.50, 0.75])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}".replace("-", "\u2212")))
    ax.tick_params(axis="x", length=0, pad=4, labelsize=8.0, colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.axvline(0.0, color="#B2BAC4", linewidth=0.95, zorder=2)

    vals = stats["board_differences"]
    ax.scatter(vals, swarm(vals), s=28, color=INDIGO, alpha=0.55, edgecolors="white", linewidths=0.4, zorder=3)
    x = stats["mean_difference"]
    lo, hi = stats["bootstrap_ci95"]
    ax.errorbar(x, 0.29, xerr=[[x - lo], [hi - x]], fmt="D", color=INDIGO,
                ms=5.2, lw=1.6, capsize=3.0, capthick=1.2, zorder=4)
    ax.text(0.50, 0.095, f"{fmt_pp(x)} pp  [{fmt_pp(lo)}, {fmt_pp(hi)}]",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.8})
    ax.set_xlabel("Board difference (pp)", fontsize=8.0, color=MUTED, labelpad=5)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("part_a_board_level.png"))
    args = parser.parse_args()

    stats_path = (
        args.repo / "runs" / "sokoban-action-only-batch5-2026-11-25"
        / "question-tests" / "paper-results" / "board-level-stats.json"
    )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    a = stats["opening"]
    c = stats["first_push"]
    b_counts = [10, 2, 1]

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, (ax, bx, cx) = plt.subplots(
        1, 3, figsize=(8.8, 2.75), gridspec_kw={"width_ratios": [1.08, 0.92, 1.08]}
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.24, top=0.77, wspace=0.32)

    plot_contrast(
        ax, a, "a", "Opening plan vs. shortest solve",
        f"Complete vs. other ({a['left_mean']:.1%} vs. {a['right_mean']:.1%}; 7 boards)"
    )

    style_panel(bx, "b", "Opening plan at first deviation", "13 episodes across 5 boards")
    bx.set(xlim=(-0.55, 2.55), ylim=(0, 12.6), yticks=[0, 4, 8, 12])
    bx.set_ylabel("Episodes", fontsize=8.0, color=MUTED, labelpad=3)
    bx.grid(axis="y", color=GRID, lw=0.7)
    bx.tick_params(axis="x", length=0, pad=5, labelsize=7.9, colors=INK)
    bx.tick_params(axis="y", length=0, pad=4, labelsize=7.9, colors=MUTED)
    bx.set_xticks(range(3), ["Optimal\nnext move", "Too short\nor blocked", "Earlier\ndivergence"])
    bars = bx.bar(range(3), b_counts, width=0.52, color=[TERRA, LIGHT, LIGHT], zorder=3)
    for idx, (bar, n) in enumerate(zip(bars, b_counts)):
        bx.text(bar.get_x() + bar.get_width() / 2, n + 0.35, f"{n}/13",
                ha="center", va="bottom", fontsize=8.2,
                fontweight="bold" if idx == 0 else "normal",
                color=TERRA if idx == 0 else MUTED)

    plot_contrast(
        cx, c, "c", "First push at choice points",
        f"Reported vs. uniform ({c['left_mean']:.1%} vs. {c['right_mean']:.1%}; 12 boards)"
    )

    fig.legend(
        handles=[
            Line2D([], [], ls="none", marker="o", color=INDIGO, alpha=0.6, markersize=5.2, label="Individual board"),
            Line2D([], [], marker="D", color=INDIGO, markersize=5.0, lw=1.5, label="Board mean and 95% bootstrap CI"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False,
        fontsize=8.0, labelcolor=MUTED, handlelength=1.8, columnspacing=1.8,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400, facecolor="white")
    plt.close(fig)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
