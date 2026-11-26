"""Fixed seed candidate. No knowledge of task rules, adapters or providers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json

from .ports import DecisionServices, Frame, InvalidModelOutput, Message, ModelRequest, PublicEvent


PROPOSE = (
    'Inspect the current decision frame. Return {"lines": [["action_id", ...], ...]}. '
    'Use IDs from the list for the first action of each line. A proposed line is hypothetical. '
    'Respect max_lines and max_depth. Return an empty list when no check is useful.'
)
CHOOSE = (
    'Choose exactly one legal action ID for the actual state. Consider scratch preview results '
    'if useful. Return {"action_id": "...", "plan_note": "..."}. The note is a short provisional '
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


@dataclass(frozen=True)
class Decision:
    action_id: str
    memory: Memory
    notes: tuple[str, ...] = ()


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
        return replace(memory, events=(*memory.events, event)[-n:] if n else ())

    def _context(self, frame: Frame, memory: Memory) -> dict:
        return {
            "task_instructions": frame.instructions,
            "current": _frame_payload(frame),
            "recent_public_events": [
                {"actor": event.before.actor, "action_id": event.action_id,
                 "event": event.label, "resulting_observation": event.after.observation}
                for event in memory.events
            ],
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
        if len(lines) > self.config.max_lines or any(
            not isinstance(line, list) or not 1 <= len(line) <= self.config.max_depth
            or any(not isinstance(action, str) or not action for action in line)
            for line in lines
        ):
            raise InvalidModelOutput("a proposed line violates the action/depth schema")
        return lines

    def decide(self, frame: Frame, memory: Memory, ports: DecisionServices) -> Decision:
        if frame.terminal is not None or not frame.actions:
            raise ValueError("cannot ask a model to act at a terminal frame")
        context = self._context(frame, memory)
        lines = self._lines(self._ask("propose", PROPOSE, context, frame, ports))
        previews = []
        notes = []
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
        reply = self._ask("choose", CHOOSE, {**context, "scratch_previews": previews}, frame, ports)
        if not isinstance(reply, dict):
            raise InvalidModelOutput("choose must return an action and plan note")
        action_id, plan = reply.get("action_id"), reply.get("plan_note")
        if not isinstance(action_id, str) or action_id not in {a.id for a in frame.actions}:
            raise InvalidModelOutput("choice did not name a legal action ID")
        if not isinstance(plan, str):
            raise InvalidModelOutput("plan_note must be text")
        return Decision(action_id, replace(memory, plan_note=plan), tuple(notes))
