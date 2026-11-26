"""Board coordinates are (row, column). Author metadata is not an observation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
from pathlib import Path


Position = tuple[int, int]
_BOARD_FILE = Path(__file__).with_name("data") / "actor_boards.json"


class Action(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    RESET = "reset"
    GIVE_UP = "give_up"


class Outcome(str, Enum):
    ACTIVE = "active"
    SOLVED = "solved"
    GAVE_UP = "gave_up"


DELTA = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}


def _add(a: Position, b: Position) -> Position:
    return (a[0] + b[0], a[1] + b[1])


@dataclass(frozen=True)
class Board:
    id: str
    height: int
    width: int
    walls: frozenset[Position]
    player_start: Position
    box_start: Position
    goal: Position
    author_optimal_cost: int | None = None

    @classmethod
    def from_author_json(cls, board_id: str, original: dict) -> Board:
        grid = original["map"]
        height, width = len(grid), len(grid[0])
        if any(len(row) != width for row in grid):
            raise ValueError(f"Board {board_id} is not rectangular")
        player, box, goal, walls = [], [], [], set()
        for r, row in enumerate(grid):
            for c, tile in enumerate(row):
                if tile not in {1, 2, 4, 5, 6, 7, 8}:
                    raise ValueError(f"Unknown actor tile {tile} at {(r, c)} on {board_id}")
                if tile in (5, 6):
                    player.append((r, c))
                if tile in (1, 2):
                    box.append((r, c))
                if tile in (2, 6, 7):
                    goal.append((r, c))
                if tile == 8:
                    walls.add((r, c))
        if (len(player), len(box), len(goal)) != (1, 1, 1):
            raise ValueError(f"Expected one player, box, and goal on {board_id}")
        if "pacman" in original and player[0] != (original["pacman"]["y"], original["pacman"]["x"]):
            raise ValueError(f"Player tile and metadata disagree on {board_id}")
        return cls(
            board_id, height, width, frozenset(walls), player[0], box[0], goal[0],
            original.get("optimal_cost"),
        )

    def walkable(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width and pos not in self.walls


def load_boards() -> dict[str, Board]:
    """Load the 23 original actor boards in canonical orientation."""
    data = json.loads(_BOARD_FILE.read_text(encoding="utf-8"))
    boards = {row["id"]: Board.from_author_json(row["id"], row["original"]) for row in data["boards"]}
    if len(boards) != len(data["boards"]):
        raise ValueError("Duplicate board ID")
    return boards


@dataclass(frozen=True)
class Observation:
    """Only visible game state and action history counters, without ground truth."""

    height: int
    width: int
    walls: frozenset[Position]
    player: Position
    box: Position
    goal: Position
    outcome: Outcome
    moves_since_reset: int = 0
    total_moves: int = 0
    resets: int = 0

    def render(self) -> str:
        """Render a lossless ASCII grid. See README for symbols."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for r, c in self.walls:
            grid[r][c] = "#"
        for (r, c), symbol in (
            (self.goal, "G"),
            (self.box, "*" if self.box == self.goal else "B"),
            (self.player, "+" if self.player == self.goal else "P"),
        ):
            grid[r][c] = symbol
        return "\n".join("".join(row) for row in grid)


@dataclass(frozen=True)
class StepResult:
    action: Action
    event: str
    observation: Observation
    reason: str | None = None


class SokobanEnv:
    """One episode, with explicit reset and give-up actions and no I/O."""

    def __init__(self, board: Board):
        self._board = board
        self._observation = Observation(
            board.height, board.width, board.walls, board.player_start, board.box_start, board.goal,
            Outcome.SOLVED if board.box_start == board.goal else Outcome.ACTIVE,
        )

    @property
    def observation(self) -> Observation:
        return self._observation

    def step(self, action: Action | str) -> StepResult:
        action = Action(action)
        state = self.observation
        if state.outcome != Outcome.ACTIVE:
            raise RuntimeError(f"Episode has ended ({state.outcome.value})")

        if action == Action.RESET:
            state = replace(
                state, player=self._board.player_start, box=self._board.box_start,
                moves_since_reset=0, resets=state.resets + 1,
            )
            event = "reset"
        elif action == Action.GIVE_UP:
            state = replace(state, outcome=Outcome.GAVE_UP)
            event = "gave_up"
        else:
            delta = DELTA[action]
            player = _add(state.player, delta)
            if not self._board.walkable(player):
                return StepResult(action, "blocked", state, "player_path_blocked")
            pushing = player == state.box
            box = _add(state.box, delta) if pushing else state.box
            if pushing and not self._board.walkable(box):
                return StepResult(action, "blocked", state, "box_path_blocked")
            state = replace(
                state, player=player, box=box,
                moves_since_reset=state.moves_since_reset + 1, total_moves=state.total_moves + 1,
                outcome=Outcome.SOLVED if box == state.goal else Outcome.ACTIVE,
            )
            event = "solved" if state.outcome == Outcome.SOLVED else "push" if pushing else "move"

        self._observation = state
        return StepResult(action, event, state)
