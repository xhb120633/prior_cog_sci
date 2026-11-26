"""Offline ground-truth audit. Not an input to an A actor or browser player."""

from collections import deque

from .env import Action, Board, Outcome, SokobanEnv, load_boards


def shortest_path(board: Board) -> tuple[Action, ...] | None:
    """Enumerate (worker, box) states independently of env.step, one keypress at a time."""
    initial = (board.player_start, board.box_start)
    frontier = deque([(initial, ())])
    seen = {initial}
    floors = {(r, c) for r in range(board.height) for c in range(board.width)} - board.walls
    while frontier:
        ((row, col), box), path = frontier.popleft()
        if box == board.goal:
            return path
        for action, dr, dc in ((Action.UP, -1, 0), (Action.DOWN, 1, 0),
                               (Action.LEFT, 0, -1), (Action.RIGHT, 0, 1)):
            next_player = row + dr, col + dc
            if next_player not in floors:
                continue
            if next_player == box:
                next_box = box[0] + dr, box[1] + dc
                if next_box not in floors:
                    continue
            else:
                next_box = box
            candidate = next_player, next_box
            if candidate not in seen:
                seen.add(candidate)
                frontier.append((candidate, path + (action,)))
    return None


def main() -> None:
    solved = unsolvable = 0
    differences = []
    for board in load_boards().values():
        path = shortest_path(board)
        cost = len(path) if path is not None else -1
        if path is None:
            unsolvable += 1
        else:
            env = SokobanEnv(board)
            for action in path:
                result = env.step(action)
                if result.event == "blocked":
                    raise AssertionError(f"Oracle and env disagree on move legality: {board.id}")
            if env.observation.outcome != Outcome.SOLVED:
                raise AssertionError(f"Oracle and env disagree on goal: {board.id}")
            solved += 1
        if board.author_optimal_cost is not None and cost != board.author_optimal_cost:
            differences.append((board.id, board.author_optimal_cost, cost))
    print(f"Verified {solved} solvable boards and {unsolvable} unsolvable boards.")
    for board_id, author_cost, computed in differences:
        print(f"Metadata discrepancy: {board_id}, author cost {author_cost}, computed cost {computed}.")


if __name__ == "__main__":
    main()
