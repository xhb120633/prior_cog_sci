"""Deterministic task adapters for the offline portable-harness smoke test."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

from sokoban_part_a.env import Action as SokobanAction, Board, DELTA, Outcome, SokobanEnv

from .ports import Action, Frame, PublicEvent


_SOKOBAN_RULES = (
    "Move the single box onto the goal. Each direction walks one cell. Walking into "
    "the box pushes it one cell if the next cell is free. You cannot pull it or move "
    "through walls. Reset restores the initial board. Give up ends the episode. "
    "Grid: # wall, . floor, P player, B box, G goal, + player on goal, * box on goal."
)
_DIRECTIONS = (
    (SokobanAction.UP, "Move up"),
    (SokobanAction.DOWN, "Move down"),
    (SokobanAction.LEFT, "Move left"),
    (SokobanAction.RIGHT, "Move right"),
)


@dataclass(frozen=True)
class _SokobanSnapshot:
    owner: object
    env: SokobanEnv


class SokobanTask:
    """Adapt the existing Sokoban environment without modifying its rules."""

    def __init__(self, board: Board):
        self._board = board
        self._owner = object()
        self._env = SokobanEnv(board)

    def reset(self) -> Frame:
        self._env = SokobanEnv(self._board)
        self._owner = object()
        return self.frame()

    def _legal(self, env: SokobanEnv) -> tuple[Action, ...]:
        obs = env.observation
        if obs.outcome != Outcome.ACTIVE:
            return ()
        choices = []
        for direction, description in _DIRECTIONS:
            dr, dc = DELTA[direction]
            row, col = obs.player[0] + dr, obs.player[1] + dc
            if not self._board.walkable((row, col)):
                continue
            if (row, col) == obs.box and not self._board.walkable((row + dr, col + dc)):
                continue
            choices.append(Action(direction.value, description))
        choices.extend((
            Action(SokobanAction.RESET.value, "Restart from the initial board"),
            Action(SokobanAction.GIVE_UP.value, "End without solving"),
        ))
        return tuple(choices)

    def _frame(self, env: SokobanEnv) -> Frame:
        obs = env.observation
        terminal = None if obs.outcome == Outcome.ACTIVE else obs.outcome.value
        return Frame(
            instructions=_SOKOBAN_RULES,
            observation=(
                f"{obs.render()}\nMoves: {obs.total_moves}. "
                f"Moves since reset: {obs.moves_since_reset}. Resets: {obs.resets}."
            ),
            actions=self._legal(env),
            actor="player",
            terminal=terminal,
        )

    def frame(self) -> Frame:
        return self._frame(self._env)

    def scratch(self) -> object:
        return _SokobanSnapshot(self._owner, copy(self._env))

    def preview(self, snapshot: object, action_id: str) -> tuple[object, Frame]:
        if not isinstance(snapshot, _SokobanSnapshot) or snapshot.owner is not self._owner:
            raise ValueError("Preview requires a snapshot from this task episode")
        if action_id not in (action.id for action in self._legal(snapshot.env)):
            raise ValueError("Illegal action on scratch state")
        # Board and observations are immutable. Stepping the copy replaces only its observation.
        new_env = copy(snapshot.env)
        new_env.step(action_id)
        return _SokobanSnapshot(self._owner, new_env), self._frame(new_env)

    def commit(self, action_id: str) -> PublicEvent:
        before = self.frame()
        if action_id not in (action.id for action in before.actions):
            raise ValueError("Illegal action on live state")
        step = self._env.step(action_id)
        return PublicEvent(before=before, action_id=action_id, after=self.frame(), label=step.event)


def sokoban_fixture() -> SokobanTask:
    """A synthetic fixture with no archived or human data dependencies."""
    rows, cols = 5, 7
    walls = frozenset(
        (r, c) for r in range(rows) for c in range(cols)
        if r in (0, rows - 1) or c in (0, cols - 1)
    )
    return SokobanTask(Board("fixture", rows, cols, walls, (2, 2), (2, 3), (2, 5)))


_ROWS, _COLS = 4, 9
_CELLS = _ROWS * _COLS
_FOUR_RULES = (
    "On a 4 by 9 board, X and O alternate placing one piece in any empty cell. "
    "There is no gravity. Four contiguous pieces of one color in a horizontal, "
    "vertical, or diagonal line win. A full board without a winner draws."
)
_CELL_IDS = tuple(f"p{i:02}" for i in range(_CELLS))
_CELL_INDEX = {action_id: i for i, action_id in enumerate(_CELL_IDS)}
_WIN_LINES = tuple(
    tuple((row + k * dr) * _COLS + col + k * dc for k in range(4))
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(_ROWS)
    for col in range(_COLS)
    if 0 <= row + 3 * dr < _ROWS and 0 <= col + 3 * dc < _COLS
)


@dataclass(frozen=True)
class _FourState:
    owner: object
    cells: tuple[str, ...]
    actor: str
    terminal: str | None = None


class FourInARowTask:
    """A rules-only 4 by 9 board, with no opponent policy or tactical oracle."""

    def __init__(self):
        self.reset()

    def reset(self) -> Frame:
        self._owner = object()
        self._state = _FourState(self._owner, (".",) * _CELLS, "X")
        return self.frame()

    @staticmethod
    def _frame(state: _FourState) -> Frame:
        actions = () if state.terminal else tuple(
            Action(_CELL_IDS[i], f"Place {state.actor} at row {i // _COLS + 1}, column {i % _COLS + 1}")
            for i, mark in enumerate(state.cells) if mark == "."
        )
        board = "\n".join(" ".join(state.cells[i:i + _COLS]) for i in range(0, _CELLS, _COLS))
        status = f"To move: {state.actor}." if state.terminal is None else f"Result: {state.terminal}."
        return Frame(
            instructions=_FOUR_RULES,
            observation=f"{board}\n{status}",
            actions=actions,
            actor=state.actor,
            terminal=state.terminal,
        )

    @staticmethod
    def _advance(state: _FourState, action_id: str) -> _FourState:
        index = _CELL_INDEX.get(action_id) if isinstance(action_id, str) else None
        if state.terminal is not None or index is None or state.cells[index] != ".":
            raise ValueError("Illegal placement")
        mark = state.actor
        cells = state.cells[:index] + (mark,) + state.cells[index + 1:]
        if any(index in line and all(cells[i] == mark for i in line) for line in _WIN_LINES):
            return _FourState(state.owner, cells, mark, f"win:{mark}")
        if "." not in cells:
            return _FourState(state.owner, cells, mark, "draw")
        return _FourState(state.owner, cells, "O" if mark == "X" else "X")

    def frame(self) -> Frame:
        return self._frame(self._state)

    def scratch(self) -> object:
        return self._state

    def preview(self, snapshot: object, action_id: str) -> tuple[object, Frame]:
        if not isinstance(snapshot, _FourState) or snapshot.owner is not self._owner:
            raise ValueError("Preview requires a snapshot from this task episode")
        successor = self._advance(snapshot, action_id)
        return successor, self._frame(successor)

    def commit(self, action_id: str) -> PublicEvent:
        before = self.frame()
        self._state = self._advance(self._state, action_id)
        return PublicEvent(
            before=before,
            action_id=action_id,
            after=self.frame(),
            label=self._state.terminal or "place",
        )
