"""Plot pooled three-category tactical choices on 409 single-threat positions.

Run with the held-out test-32-paired.json as the positional argument.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TERRA = "#C85A32"
INDIGO = "#275684"
SLATE = "#949CA6"
INK = "#1E232A"
MUTED = "#5A636E"
GRID = "#ECEEF2"


def plot(results: Path, output: Path) -> None:
    scores = json.loads(results.read_text(encoding="utf-8"))
    baseline = scores["aggregate"]["baseline"]["tactics"]["all"]
    candidate = scores["aggregate"]["r3-c1"]["tactics"]["all"]
    assert baseline["positions"] == candidate["positions"] == scores["n"]["tactics"]

    hum = baseline["observed_rate"]
    bas = baseline["raw_model_rate"]
    can = candidate["raw_model_rate"]
    for p in (hum, bas, can):
        assert math.isclose(sum(p.values()), 1.0, rel_tol=1e-12)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.4,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.10), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )
    fig.subplots_adjust(left=0.17, right=0.975, top=0.76, bottom=0.19, wspace=0.34)

    # Panel a: Block rate on single threats
    ax1.text(0.0, 1.15, "a", transform=ax1.transAxes, fontsize=10.2, fontweight="bold", color=TERRA)
    ax1.text(0.06, 1.15, "Defending a single-move threat", transform=ax1.transAxes,
             fontsize=9.2, fontweight="bold", color=INK)
    ax1.text(0.0, 1.03, "Share of 409 positions where player blocked", transform=ax1.transAxes,
             fontsize=7.6, color=MUTED)

    labels = ["Human", "Baseline", "Discovered (r3-c1)"]
    vals = [hum["block"] * 100, bas["block"] * 100, can["block"] * 100]
    cols = [TERRA, SLATE, INDIGO]
    ys = [2, 1, 0]

    ax1.barh(ys, vals, height=0.38, color=cols, zorder=3)
    for y, v, c in zip(ys, vals, cols):
        ax1.text(v + 1.8, y, f"{v:.1f}%", va="center", ha="left",
                 fontsize=8.3, fontweight="bold" if y != 1 else "normal",
                 color=c if y != 1 else MUTED)

    # Gap callout between Human (81.7) and Models (59.1)
    gap = vals[2] - vals[0]
    ax1.plot([vals[2], vals[0]], [1.44, 1.44], color=INK, lw=0.85, ls="--", zorder=4)
    ax1.text((vals[2] + vals[0]) / 2, 1.49, f"{gap:.1f} pp".replace("-", "\u2212"),
             ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=INK)

    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.55, 2.55)
    ax1.set_yticks(ys, labels)
    ax1.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax1.tick_params(axis="x", length=0, pad=4, labelsize=7.7, colors=MUTED)
    ax1.tick_params(axis="y", length=0, pad=6, labelsize=8.3, colors=INK)
    ax1.grid(axis="x", color=GRID, lw=0.65, zorder=0)
    for name, spine in ax1.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color("#C5CBD3")
        spine.set_linewidth(0.7)

    # Panel b: Breakdown of missed blocks
    ax2.text(0.0, 1.15, "b", transform=ax2.transAxes, fontsize=10.2, fontweight="bold", color=TERRA)
    ax2.text(0.07, 1.15, "Where missed blocks go", transform=ax2.transAxes,
             fontsize=9.2, fontweight="bold", color=INK)
    ax2.text(0.0, 1.03, "Models chase own threats 3\u00d7 as often", transform=ax2.transAxes,
             fontsize=7.6, color=MUTED)

    cats = ["Offensive miss\n(own threat)", "Other miss"]
    x = np.arange(len(cats))
    w = 0.275
    h_miss = [hum["offensive_miss"] * 100, hum["other_miss"] * 100]
    b_miss = [bas["offensive_miss"] * 100, bas["other_miss"] * 100]
    c_miss = [can["offensive_miss"] * 100, can["other_miss"] * 100]

    for offset, data, col, weight in [
        (-w, h_miss, TERRA, "bold"),
        (0.0, b_miss, SLATE, "normal"),
        (w, c_miss, INDIGO, "bold"),
    ]:
        bars = ax2.bar(x + offset, data, width=w * 0.80, color=col, zorder=3)
        for bar, val in zip(bars, data):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.7, f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=7.0, fontweight=weight,
                     color=col if col != SLATE else MUTED)

    ax2.set_ylim(0, 27.5)
    ax2.set_yticks([0, 10, 20], ["0%", "10%", "20%"])
    ax2.set_xticks(x, cats)
    ax2.tick_params(axis="x", length=0, pad=5, labelsize=8.0, colors=INK)
    ax2.tick_params(axis="y", length=0, pad=4, labelsize=7.7, colors=MUTED)
    ax2.grid(axis="y", color=GRID, lw=0.65, zorder=0)
    for name, spine in ax2.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color("#C5CBD3")
        spine.set_linewidth(0.7)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Held-out test-32-paired.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("four_tactics.png"))
    args = parser.parse_args()
    plot(args.results, args.output)
    print("Saved", args.output)
