"""Bounded route rehearsal and episodic recognition."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json

from .ports import DecisionServices, Frame, InvalidModelOutput, Message, ModelRequest, PublicEvent


PROPOSE = (
    'Inspect the current decision frame. Return {"lines": [["action_id", ...], ...]}. '
    'Use IDs from the list for the first action of each line. A proposed line is hypothetical. '
    'Rehearse the likely route to a useful working position. Put a plausible continuation first. Respect max_lines and max_depth. Return an empty list when no check is useful.'
)
CHOOSE = (
    'Choose exactly one legal action ID for the actual state. Consider scratch preview results '
    'if useful. Use the previous movement as a cue if it still serves the intended route. Return {"action_id": "...", "plan_note": "..."}. The note is a short provisional '
    'plan to retain for the next decision, or the empty string.'
)


@dataclass(frozen=True)
class H0Config:
    recent_events: int = 4
    max_lines: int = 2
    max_depth: int = 2

    def __post_init__(self) -> None:
        for key, lower, upper in (("recent_events", 0, 8), ("max_lines", 1, 3), ("max_depth", 1, 3)):
            value = getattr(self, key)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{key} must be an integer in [{lower}, {upper}]")


@dataclass(frozen=True)
class Memory:
    events: tuple[PublicEvent, ...] = ()
    plan_note: str = ""
    departures: tuple[tuple[tuple[str, str], str], ...] = ()


@dataclass(frozen=True)
class Decision:
    action_id: str
    memory: Memory
    notes: tuple[str, ...] = ()


def _state_key(frame: Frame) -> tuple[str, str]:
    """Recognize rectangular text displays even when surrounding counters change."""
    observation = frame.observation
    try:
        document = json.loads(observation)
    except ValueError:
        document = None
    if isinstance(document, dict):
        rows = document.get("grid")
        if (isinstance(rows, list) and len(rows) >= 2 and all(isinstance(row, str) for row in rows)
                and len(rows[0]) >= 2 and all(len(row) == len(rows[0]) for row in rows)):
            return frame.actor, "grid\n" + "\n".join(rows)

    best = []
    run = []
    for row in observation.splitlines():
        row = row.strip()
        if not run or len(row) != len(run[-1]):
            run = []
        if len(row) >= 2 and not any(c.isspace() for c in row) and any(not c.isalnum() for c in row):
            run.append(row)
            if len(run) > len(best):
                best = run.copy()
        else:
            run = []
    if len(best) >= 2:
        return frame.actor, "grid\n" + "\n".join(best)
    return frame.actor, "text\n" + observation


def _frame_payload(frame: Frame) -> dict:
    return {
        "observation": frame.observation,
        "actor": frame.actor,
        "legal_actions": [{"id": a.id, "description": a.description} for a in frame.actions],
        "terminal": frame.terminal,
    }


@dataclass(frozen=True)
class H0:
    config: H0Config = H0Config()

    def start(self) -> Memory:
        return Memory()

    def observe(self, event: PublicEvent, memory: Memory) -> Memory:
        n = self.config.recent_events
        return replace(memory, departures=(*memory.departures, (_state_key(event.before), event.action_id)), events=(*memory.events, event)[-n:] if n else ())

    def _context(self, frame: Frame, memory: Memory) -> dict:
        key = _state_key(frame)
        return {
            "task_instructions": frame.instructions,
            "current": _frame_payload(frame),
            "recent_public_events": [
                {"actor": event.before.actor, "action_id": event.action_id,
                 "event": event.label, "resulting_observation": event.after.observation}
                for event in memory.events
            ],
            "earlier_departures_from_here": [a for obs, a in memory.departures if obs == key][-4:],
            "active_plan": memory.plan_note,
            "max_lines": self.config.max_lines,
            "max_depth": self.config.max_depth,
        }

    def _ask(self, stage: str, prompt: str, context: dict, frame: Frame, ports: DecisionServices) -> object:
        return ports.ask(ModelRequest(
            stage, (Message("system", prompt), Message("user", json.dumps(context, ensure_ascii=False))),
            tuple(a.id for a in frame.actions), ports.max_output_tokens,
            self.config.max_lines, self.config.max_depth,
        ))

    def _lines(self, reply: object) -> list[list[str]]:
        if not isinstance(reply, dict) or not isinstance(reply.get("lines"), list):
            raise InvalidModelOutput("propose must return a list of lines")
        lines = reply["lines"]
        cleaned = []
        for line in lines:
            if not isinstance(line, list) or any(not isinstance(a, str) or not a for a in line):
                raise InvalidModelOutput("proposed actions must be strings")
            trimmed = line[:self.config.max_depth]
            if trimmed and trimmed not in cleaned:
                cleaned.append(trimmed)
            if len(cleaned) == self.config.max_lines:
                break
        return cleaned

    def decide(self, frame: Frame, memory: Memory, ports: DecisionServices) -> Decision:
        if frame.terminal is not None or not frame.actions:
            raise ValueError("cannot ask a model to act at a terminal frame")
        context = self._context(frame, memory)
        lines = self._lines(self._ask("propose", PROPOSE, context, frame, ports))
        previews = []
        notes = []
        # Reopen deliberation on revisiting a position. Continuing an available
        # movement in the proposed route draws on motor memory.
        if memory.events and not context["earlier_departures_from_here"]:
            previous = memory.events[-1].action_id
            if lines and lines[0][0] == previous:
                lines = []
            else:
                lines = [line[:1] for line in lines[:1]]
        for line_index, line in enumerate(lines):
            view = ports.root()
            for depth, action_id in enumerate(line, start=1):
                if not ports.remaining_previews:
                    notes.append("preview_budget_reached")
                    break
                if action_id not in {action.id for action in view.frame.actions}:
                    notes.append(f"unavailable_action_in_line:{action_id}")
                    break
                view = ports.preview(view, action_id)
                previews.append({"line": line_index, "depth": depth, "action_id": action_id,
                                 "result": _frame_payload(view.frame)})
                if view.frame.terminal is not None:
                    break
        choose_prompt = CHOOSE
        if context["earlier_departures_from_here"]:
            choose_prompt += (
                " You have departed from this position before and returned. If those "
                "attempts were unsuccessful, consider another approach or stopping."
            )
        reply = self._ask("choose", choose_prompt, {**context, "scratch_previews": previews}, frame, ports)
        if not isinstance(reply, dict):
            raise InvalidModelOutput("choose must return an action and plan note")
        action_id, plan = reply.get("action_id"), reply.get("plan_note")
        if not isinstance(action_id, str) or action_id not in {a.id for a in frame.actions}:
            raise InvalidModelOutput("choice did not name a legal action ID")
        if not isinstance(plan, str):
            raise InvalidModelOutput("plan_note must be text")
        return Decision(action_id, replace(memory, plan_note=plan), tuple(notes))

