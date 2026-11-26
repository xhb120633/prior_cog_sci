"""Render the Part B Sokoban held-out figure with per-board scatter + summary dumbbells.

    python plot_sokoban.py /path/to/prior_cog_sci
"""

from argparse import ArgumentParser
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TERRA = "#C85A32"
INDIGO = "#275684"
SLATE = "#949CA6"
INK = "#1E232A"
MUTED = "#5A636E"
GRID = "#ECEEF2"


def aggregates(repo):
    run = repo / "runs" / "sokoban-heldout-final-2026-11-25" / "run"
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    board_path = repo / "sokoban_part_a" / "data" / "actor_boards.json"
    boards = json.loads(board_path.read_text(encoding="utf-8"))["boards"]
    feasible = {str(board["id"]): board["original"]["optimal_cost"] >= 0 for board in boards}
    comp = scores["comparison"]
    by_board = comp["per_board"]
    solvable = [v for k, v in by_board.items() if feasible[k]]
    unsolvable = [v for k, v in by_board.items() if not feasible[k]]
    all_rows = list(by_board.values())
    assert len(solvable) == len(unsolvable) == 4

    arms = ("baseline", "champion")
    keys = ("trajectory_distance", "rt_spearman")
    summary = {
        key: [
            [comp["macro"][a][key] for a in arms],
            [mean(row[a][key] for row in solvable) for a in arms],
            [mean(row[a][key] for row in unsolvable) for a in arms],
        ]
        for key in keys
    }
    per_board = {
        key: [
            ([r["baseline"][key] for r in grp], [r["champion"][key] for r in grp])
            for grp in (all_rows, solvable, unsolvable)
        ]
        for key in keys
    }
    arm_pb = scores["arms"]["pilot-unit-cost"]["per_board"]
    hh = [
        comp["macro"]["human_human_distance_at_matched_reference_n"],
        mean(v["human_human_distance"] for k, v in arm_pb.items() if feasible[k]),
        mean(v["human_human_distance"] for k, v in arm_pb.items() if not feasible[k]),
    ]
    return summary, per_board, hh


def plot(summary, per_board, hh, output):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.4,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.28))
    fig.subplots_adjust(left=0.155, right=0.98, top=0.735, bottom=0.165, wspace=0.22)

    legend = [
        Line2D([], [], marker="o", color=SLATE, lw=0, markersize=5.0, label="Local sA-LH"),
        Line2D([], [], marker="o", color=INDIGO, lw=0, markersize=5.0, label="Discovered model"),
        Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor=TERRA,
               markeredgewidth=1.25, lw=0, markersize=4.8, label="Human\u2013human yardstick"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.54, 1.015),
               frameon=False, ncol=3, handletextpad=0.35, columnspacing=1.35,
               fontsize=8.0, labelcolor=INK)

    titles = (
        ("a", "Trajectory edit distance", "Lower is closer to people; faint dots are boards"),
        ("b", r"Response time correlation", r"Within-board Spearman $\rho$; higher is better"),
    )
    xlims = ((0.0, 0.85), (-0.16, 0.27))
    ticks = ((0.0, 0.2, 0.4, 0.6, 0.8), (-0.1, 0.0, 0.1, 0.2))

    for j, (ax, (key, points)) in enumerate(zip(axes, summary.items())):
        letter, title, sub = titles[j]
        ax.text(0.0, 1.15, letter, transform=ax.transAxes, fontsize=10.2, fontweight="bold", color=TERRA)
        ax.text(0.065, 1.15, title, transform=ax.transAxes, fontsize=9.2, fontweight="bold", color=INK)
        ax.text(0.0, 1.045, sub, transform=ax.transAxes, fontsize=7.6, color=MUTED)

        ax.set_xlim(xlims[j])
        ax.set_ylim(-0.52, 2.58)
        ax.set_xticks(ticks[j], [f"{v:.1f}".replace("-", "\u2212") for v in ticks[j]])
        ax.set_yticks([2, 1, 0], ["All (8 boards)", "Solvable (4)", "Unsolvable (4)"] if j == 0 else [])
        ax.tick_params(axis="x", length=0, pad=4, labelsize=7.8, colors=MUTED)
        ax.tick_params(axis="y", length=0, pad=7, labelsize=8.3, colors=INK)
        ax.grid(axis="x", c=GRID, lw=0.65, zorder=0)
        if j == 1:
            ax.axvline(0.0, lw=0.9, color="#B2BAC4", zorder=1)
        for name, spine in ax.spines.items():
            spine.set_visible(name == "bottom")
            spine.set_color("#C5CBD3")
            spine.set_linewidth(0.7)

        for idx, (y, (base, champ)) in enumerate(zip((2, 1, 0), points)):
            b_pts, c_pts = per_board[key][idx]
            # Faint per-board points on a subtle offset band
            n_b = len(b_pts)
            jit = [((k - (n_b - 1) / 2.0) / max(n_b, 1)) * 0.20 for k in range(n_b)]
            ax.scatter(b_pts, [y + d for d in jit], s=14, color=SLATE, alpha=0.26, lw=0, zorder=2)
            ax.scatter(c_pts, [y + d for d in jit], s=14, color=INDIGO, alpha=0.26, lw=0, zorder=2)

            ax.plot([base, champ], [y, y], color="#C5CBD3", lw=1.8, zorder=3)
            ax.scatter([base, champ], [y, y], c=[SLATE, INDIGO], s=32,
                       linewidths=0.6, edgecolors="white", zorder=4)
            for val, c, off, weight in [(base, MUTED, 4.6, "normal"), (champ, INDIGO, -4.6, "bold")]:
                ax.annotate(f"{val:.3f}".replace("-", "\u2212"), xy=(val, y), xytext=(0, off),
                            textcoords="offset points", ha="center",
                            va="bottom" if off > 0 else "top",
                            color=c, fontsize=7.6, fontweight=weight, zorder=5)
            if j == 0:
                hval = hh[idx]
                ax.scatter([hval], [y], marker="D", s=24, facecolors="white",
                           edgecolors=TERRA, linewidths=1.25, zorder=5)
                ax.annotate(f"{hval:.3f}", xy=(hval, y), xytext=(0, 4.8),
                            textcoords="offset points", ha="center", va="bottom",
                            color=TERRA, fontsize=7.4, fontweight="bold", zorder=5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("sokoban_heldout.png"))
    args = parser.parse_args()
    summ, pb, hh_vals = aggregates(args.repo)
    plot(summ, pb, hh_vals, args.output)
    print("Saved", args.output)
