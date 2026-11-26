"""Continue established action runs and test interrupted intentions against experience."""

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
    visited: frozenset[tuple[str, str]] = frozenset()


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
        return replace(memory, events=(*memory.events, event)[-n:] if n else (),
                       visited=memory.visited | {_state_key(event.before)})

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
        result = []
        for line in reply["lines"]:
            if not isinstance(line, list) or any(not isinstance(a, str) or not a for a in line):
                raise InvalidModelOutput("a proposed line contains malformed action IDs")
            clipped = line[:self.config.max_depth]
            if clipped and clipped not in result:
                result.append(clipped)
            if len(result) == self.config.max_lines:
                break
        return result
    def decide(self, frame: Frame, memory: Memory, ports: DecisionServices) -> Decision:
        if frame.terminal is not None or not frame.actions:
            raise ValueError("cannot ask a model to act at a terminal frame")
        context = self._context(frame, memory)
        # A small change in available controls can accompany ordinary progress
        # (e.g. a return action becomes available after leaving a starting state).
        last = memory.events[-1] if memory.events else None
        current_ids = {a.id for a in frame.actions}
        established_run = len(memory.events) >= 2 and memory.events[-2].after == last.before and memory.events[-2].action_id == last.action_id and memory.events[-2].before.actor == frame.actor
        continuation = (
            last is not None
            and last.before.actor == frame.actor
            and last.after == frame
            and last.action_id in current_ids
            and (established_run or len(current_ids - {a.id for a in last.before.actions}) <= 1)
            and _state_key(frame) not in memory.visited
        )
        if continuation:
            context["ongoing_action"] = last.action_id
            choose_prompt = CHOOSE + (
                " The last action is still available after observable progress. Continue it "
                "if it still advances the immediate intention. Change when the current "
                "observation gives a concrete reason to do so."
            )
            lines = []
        else:
            choose_prompt = CHOOSE
            propose_prompt = PROPOSE + (
                " Picture a small, testable change you could make from here. "
                "Include another possible first move if uncertain."
            )
            interrupted = last is not None and last.after == frame and last.before.actor == frame.actor
            single_test = interrupted and _state_key(frame) not in memory.visited
            if single_test:
                propose_prompt = PROPOSE + (
                    " The movement has been interrupted. Name the immediate move that best "
                    "tests your intention, or leave the lines empty if its effect is clear."
                )
                choose_prompt += " Judge that immediate effect against your intended change."
            lines = self._lines(self._ask("propose", propose_prompt, context, frame, ports))
            if single_test:
                lines = [lines[0][:1]] if lines else []

        # Recognize returns to actual experience separately from loops inside a proposed line.
        previews = []
        notes = []
        cache = {}
        seen = memory.visited | {_state_key(frame)}
        for line_index, line in enumerate(lines):
            view = ports.root()
            path = ()
            for depth, action_id in enumerate(line, 1):
                if action_id not in {a.id for a in view.frame.actions}:
                    break
                path += (action_id,)
                if path not in cache:
                    if len(cache) >= 8 or not ports.remaining_previews:
                        break
                    cache[path] = ports.preview(view, action_id)
                view = cache[path]
                familiar = _state_key(view.frame) in seen
                previews.append({"line": line_index, "depth": depth, "action_id": action_id,
                                 "seen_in_play": familiar, "result": _frame_payload(view.frame)})
                if familiar or view.frame.terminal is not None:
                    break
        if previews:
            choose_prompt += (
                " When the overall solution is unclear, try a concrete local change whose "
                "consequences you have not yet experienced. A return to a familiar state "
                "can be worthwhile to reach a different exit. Weigh ending the attempt "
                "when the remaining possibilities seem exhausted."
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
