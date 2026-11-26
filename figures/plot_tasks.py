"""Render Figure 1: Task overview for Sokoban (real Board 30002) and Four-in-a-row (4x9 threat).

    python plot_tasks.py /path/to/prior_cog_sci
"""

from argparse import ArgumentParser
from collections import deque
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


TERRA = "#C85A32"
TERRA_SOFT = "#F6E4DD"
INDIGO = "#275684"
INDIGO_SOFT = "#E1EAF2"
SLATE = "#6E7781"
WALL_FILL = "#DCE0E6"
WALL_EDGE = "#C4CBD3"
FLOOR_EDGE = "#ECEEF2"
INK = "#1E232A"
MUTED = "#5A636E"


def bfs_path(board):
    start = (board.player_start, board.box_start)
    q = deque([(start, [start])])
    seen = {start}
    while q:
        (pr, pc), (br, bc) = q[0][0]
        path = q[0][1]
        q.popleft()
        if (br, bc) == board.goal:
            return path
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            npr, npc = pr + dr, pc + dc
            if not board.walkable((npr, npc)):
                continue
            nbr, nbc = br, bc
            if (npr, npc) == (br, bc):
                nbr, nbc = br + dr, bc + dc
                if not board.walkable((nbr, nbc)):
                    continue
            nxt = ((npr, npc), (nbr, nbc))
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
    return None


def interior_cells(board):
    seen = {board.player_start}
    q = deque([board.player_start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if board.walkable((nr, nc)) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def draw_sokoban(ax, board):
    interior = interior_cells(board)
    path = bfs_path(board)
    box_path = []
    for _, bpos in path:
        if not box_path or box_path[-1] != bpos:
            box_path.append(bpos)

    H, W = board.height, board.width
    for r in range(H):
        for c in range(W):
            y = H - 1 - r
            if (r, c) in board.walls:
                ax.add_patch(Rectangle((c, y), 1, 1, facecolor=WALL_FILL, edgecolor=WALL_EDGE, lw=0.6, zorder=2))
            elif (r, c) in interior:
                ax.add_patch(Rectangle((c, y), 1, 1, facecolor="#FAF9F7", edgecolor=FLOOR_EDGE, lw=0.5, zorder=1))

    # Draw optimal box push route
    xs = [c + 0.5 for _, c in box_path]
    ys = [H - 1 - r + 0.5 for r, _ in box_path]
    ax.plot(xs, ys, color=INDIGO, lw=1.6, ls="--", alpha=0.75, zorder=3)
    # Arrowhead entering goal
    if len(xs) >= 2:
        ax.annotate(
            "", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
            arrowprops=dict(arrowstyle="-|>", color=INDIGO, lw=1.4, shrinkA=6, shrinkB=9, mutation_scale=10),
            zorder=4,
        )

    # Goal
    gr, gc = board.goal
    gy = H - 1 - gr
    ax.add_patch(Circle((gc + 0.5, gy + 0.5), 0.36, facecolor=TERRA_SOFT, edgecolor=TERRA, lw=1.3, zorder=4))
    ax.text(gc + 0.5, gy + 0.5, "G", ha="center", va="center", fontsize=7.0, fontweight="bold", color=TERRA, zorder=5)

    # Box
    br, bc = board.box_start
    by = H - 1 - br
    ax.add_patch(FancyBboxPatch((bc + 0.14, by + 0.14), 0.72, 0.72, boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=INDIGO, edgecolor="white", lw=0.8, zorder=5))
    ax.text(bc + 0.5, by + 0.5, "B", ha="center", va="center", fontsize=7.0, fontweight="bold", color="white", zorder=6)

    # Player
    pr, pc = board.player_start
    py = H - 1 - pr
    ax.add_patch(Circle((pc + 0.5, py + 0.5), 0.36, facecolor=TERRA, edgecolor="white", lw=0.8, zorder=5))
    ax.text(pc + 0.5, py + 0.5, "P", ha="center", va="center", fontsize=6.9, fontweight="bold", color="white", zorder=6)

    ax.set_xlim(-0.15, W + 0.15)
    ax.set_ylim(-0.35, H + 0.25)
    ax.set_aspect("equal")
    ax.set_anchor("W")
    ax.axis("off")

    ax.text(0.0, 1.15, "a", transform=ax.transAxes, fontsize=10.2, fontweight="bold", color=TERRA)
    ax.text(0.06, 1.15, "Sokoban (single-agent planning)", transform=ax.transAxes,
            fontsize=9.1, fontweight="bold", color=INK)
    ax.text(0.0, 1.03, f"Board 30002 ({len(path)-1} moves optimal); dashed line = optimal box path",
            transform=ax.transAxes, fontsize=7.4, color=MUTED)
    ax.text(0.0, -0.06, "P = player   B = box   G = goal   (14 solvable, 9 unsolvable)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.1, color=MUTED)


def draw_four_in_a_row(ax):
    H, W = 4, 9
    # Canonical single-threat midgame position (10 stones: 5 Indigo to move, 5 Slate opponent)
    # Opponent (Slate) has 3-in-a-row horizontally at row 1, cols 3,4,5 -> must block at (1, 6) [col 2 blocked by indigo]
    slate_stones = [(1, 3), (1, 4), (1, 5), (0, 4), (2, 2)]
    indigo_stones = [(1, 2), (2, 4), (2, 5), (3, 3), (0, 6)]
    block_sq = (1, 6)     # completes opponent's 4-in-a-row -> unique winning square
    off_miss_sq = (2, 6)  # creates Indigo's own 3-in-a-row (row 2, cols 4,5,6) while leaving (1,6) open

    for r in range(H):
        for c in range(W):
            y = H - 1 - r
            ax.add_patch(Rectangle((c, y), 1, 1, facecolor="#FAF9F7", edgecolor="#D8DCE2", lw=0.75, zorder=1))

    # Highlight opponent's 3-in-a-row threat line
    ax.plot([3.5, 6.5], [H - 1 - 1 + 0.5] * 2, color=TERRA, lw=1.2, alpha=0.45, zorder=2)

    for r, c in slate_stones:
        y = H - 1 - r
        ax.add_patch(Circle((c + 0.5, y + 0.5), 0.34, facecolor=SLATE, edgecolor="white", lw=0.9, zorder=3))
    for r, c in indigo_stones:
        y = H - 1 - r
        ax.add_patch(Circle((c + 0.5, y + 0.5), 0.34, facecolor=INDIGO, edgecolor="white", lw=0.9, zorder=3))

    # Unique block square
    br, bc = block_sq
    by = H - 1 - br
    ax.add_patch(FancyBboxPatch((bc + 0.10, by + 0.10), 0.80, 0.80, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=TERRA_SOFT, edgecolor=TERRA, lw=1.4, ls="--", zorder=4))
    ax.text(bc + 0.5, by + 0.5, "!", ha="center", va="center", fontsize=9.0, fontweight="bold", color=TERRA, zorder=5)

    # Offensive miss square
    orr, oc = off_miss_sq
    oy = H - 1 - orr
    ax.add_patch(FancyBboxPatch((oc + 0.10, oy + 0.10), 0.80, 0.80, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=INDIGO_SOFT, edgecolor=INDIGO, lw=1.2, ls=":", zorder=4))
    ax.text(oc + 0.5, oy + 0.5, "+", ha="center", va="center", fontsize=8.8, fontweight="bold", color=INDIGO, zorder=5)

    ax.set_xlim(-0.2, W + 0.2)
    ax.set_ylim(-0.35, H + 0.35)
    ax.set_aspect("equal")
    ax.set_anchor("W")
    ax.axis("off")

    ax.text(0.0, 1.15, "b", transform=ax.transAxes, fontsize=10.2, fontweight="bold", color=TERRA)
    ax.text(0.06, 1.15, "Four-in-a-row (4\u00d79 adversarial play)", transform=ax.transAxes,
            fontsize=9.1, fontweight="bold", color=INK)
    ax.text(0.0, 1.03, "New tactical metric on single-threat positions (blue to move)",
            transform=ax.transAxes, fontsize=7.4, color=MUTED)
    ax.text(0.0, -0.06, "! = Block opponent's win    + = Offensive miss    other = Other miss",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.1, color=MUTED)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("task_overview.png"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from sokoban_part_a.env import load_boards

    board = load_boards()["30002"]

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.2,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.20), gridspec_kw={"width_ratios": [1.0, 1.18]})
    fig.subplots_adjust(left=0.02, right=0.985, top=0.77, bottom=0.14, wspace=0.12)

    draw_sokoban(ax1, board)
    draw_four_in_a_row(ax2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400, facecolor="white")
    plt.close(fig)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
