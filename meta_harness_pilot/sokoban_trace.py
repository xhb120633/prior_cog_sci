"""Convert pilot runner records to the locked Part B Sokoban schema."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real

from .runner import DecisionResult, EpisodeResult


_DIRECTIONS = {"up": "U", "down": "D", "left": "L", "right": "R"}
_TERMINALS = {"solved": "SOLVED", "gave_up": "GIVE_UP", "cap": "CAP"}


def episode_record(episode: EpisodeResult) -> dict:
    """Score committed actions, not hypothetical preview turns or failed runs."""
    if episode.terminal not in _TERMINALS or episode.error is not None:
        raise ValueError(f"Cannot score unsuccessful or incomplete episode: {episode.terminal}")
    if episode.initial.terminal is not None or not episode.events:
        raise ValueError("A scored episode must start in play and commit at least one action")
    if len(episode.events) != len(episode.decisions):
        raise ValueError("A scored episode requires one decision trace per committed event")

    tokens: list[str] = []
    moves = resets = 0
    frame = episode.initial
    for event, decision in zip(episode.events, episode.decisions, strict=True):
        if (
            frame.terminal is not None or event.before != frame
            or decision.frame != frame or decision.event != event or decision.status != "committed"
            or decision.error is not None or decision.action_id != event.action_id
            or event.action_id not in {action.id for action in frame.actions}
        ):
            raise ValueError("Episode does not contain a consistent chain of legal committed decisions")
        action = event.action_id
        after = event.after.terminal
        if action in _DIRECTIONS:
            if (after is None and event.label not in {"move", "push"}) or (
                after is not None and (after != "solved" or event.label != "solved")
            ):
                raise ValueError("Unexpected outcome for a committed direction")
            tokens.append(_DIRECTIONS[action])
            moves += 1
        elif action == "reset" and event.label == "reset" and after is None:
            tokens.append("RESET")
            resets += 1
        elif action != "give_up" or event.label != "gave_up" or after != "gave_up":
            raise ValueError(f"Unexpected Sokoban transition: {action!r}")
        frame = event.after

    expected_terminal = None if episode.terminal == "cap" else episode.terminal
    if frame != episode.final or frame.terminal != expected_terminal:
        raise ValueError("Episode's last committed state disagrees with the declared outcome")
    tokens.append(_TERMINALS[episode.terminal])
    return {"tokens": tokens, "outcome": episode.terminal, "moves": moves, "resets": resets}


def replay_record(rt_ms: float, repeats: Sequence[DecisionResult]) -> dict:
    """Pair an observed RT with three scratch-work counts on the same frame.

    The caller must rebuild the same public prefix for every repeat and keep
    the human action and RT out of every model decision. This adapter never
    commits a move or treats model latency or token counts as observed work.
    """
    if isinstance(rt_ms, bool) or not isinstance(rt_ms, Real):
        raise ValueError("Human RT must be a finite positive number in milliseconds")
    if not isfinite(rt_ms) or rt_ms <= 0:
        raise ValueError("Human RT must be a finite positive number in milliseconds")
    if len(repeats) != 3:
        raise ValueError("Part B RT matching requires exactly three replay decisions")

    frame = repeats[0].trace.frame
    counts: list[int] = []
    legal = {action.id for action in frame.actions}
    for repeat in repeats:
        trace = repeat.trace
        if (
            trace.frame != frame or frame.terminal is not None or trace.status != "ready"
            or trace.error is not None or trace.event is not None or repeat.decision is None
            or trace.action_id not in legal or repeat.decision.action_id != trace.action_id
            or any(p.error is not None or p.result is None for p in trace.previews)
        ):
            raise ValueError("Replay decisions must be valid, uncommitted, and share a pre-move frame")
        counts.append(trace.preview_attempts)
    return {"rt_ms": float(rt_ms), "expansions": counts}
