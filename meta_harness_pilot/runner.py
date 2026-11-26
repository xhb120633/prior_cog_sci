"""Fixed task/model orchestration and observable per-decision budgets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from .ports import (
    BudgetExceeded,
    Frame,
    InvalidModelOutput,
    ModelPort,
    ModelReply,
    ModelRequest,
    PublicEvent,
    ScratchView,
    TaskPort,
)


@dataclass(frozen=True)
class Limits:
    max_requests: int = 2
    max_preview_attempts: int = 8
    max_output_tokens: int = 4096
    max_decisions: int = 100

    def __post_init__(self) -> None:
        for name in ("max_requests", "max_preview_attempts", "max_output_tokens", "max_decisions"):
            value = getattr(self, name)
            minimum = 0 if name in {"max_requests", "max_preview_attempts"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ModelCallTrace:
    request: ModelRequest
    reply: ModelReply | None = None
    error: str | None = None


@dataclass(frozen=True)
class PreviewTrace:
    source: ScratchView
    action_id: str
    result: ScratchView | None = None
    error: str | None = None


@dataclass(frozen=True)
class DecisionTrace:
    frame: Frame
    action_id: str | None
    notes: tuple[str, ...]
    model_calls: tuple[ModelCallTrace, ...]
    previews: tuple[PreviewTrace, ...]
    event: PublicEvent | None
    status: str
    error: str | None = None

    @property
    def preview_attempts(self) -> int:
        return len(self.previews)

    @property
    def request_count(self) -> int:
        return len(self.model_calls)


@dataclass(frozen=True)
class DecisionResult:
    decision: Any | None
    trace: DecisionTrace


@dataclass(frozen=True)
class EpisodeResult:
    initial: Frame
    final: Frame
    events: tuple[PublicEvent, ...]
    decisions: tuple[DecisionTrace, ...]
    terminal: str
    error: str | None = None

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(event.action_id for event in self.events)


class _ModelExecutionError(RuntimeError):
    pass


class _DecisionServices:
    """Expose metered previews and model calls, but not task.commit or snapshots."""

    def __init__(self, task: TaskPort, model: ModelPort, frame: Frame, limits: Limits) -> None:
        self._task = task
        self._model = model
        self._limits = limits
        self._frame = frame
        self._root = ScratchView(handle=0, frame=frame)
        self._scratch = {0: (self._root, task.scratch())}
        self.model_calls: list[ModelCallTrace] = []
        self.previews: list[PreviewTrace] = []
        self.assert_live_unchanged()

    @property
    def remaining_requests(self) -> int:
        return self._limits.max_requests - len(self.model_calls)

    @property
    def remaining_previews(self) -> int:
        return self._limits.max_preview_attempts - len(self.previews)

    @property
    def max_output_tokens(self) -> int:
        return self._limits.max_output_tokens

    def assert_live_unchanged(self) -> None:
        if self._task.frame() != self._frame:
            raise RuntimeError("Task adapter changed live state before commit")

    def root(self) -> ScratchView:
        return self._root

    def preview(self, view: ScratchView, action_id: str) -> ScratchView:
        if self.remaining_previews <= 0:
            raise BudgetExceeded("Scratch transition limit reached")
        stored = self._scratch.get(view.handle)
        if stored is None or stored[0] is not view:
            raise ValueError("Unknown scratch view for this decision")
        if action_id not in {action.id for action in view.frame.actions}:
            raise ValueError(f"Illegal scratch action: {action_id!r}")
        try:
            snapshot, frame = self._task.preview(stored[1], action_id)
        except Exception as exc:
            self.previews.append(PreviewTrace(view, action_id, error=str(exc)))
            raise
        result = ScratchView(handle=len(self._scratch), frame=frame)
        self._scratch[result.handle] = (result, snapshot)
        self.previews.append(PreviewTrace(view, action_id, result))
        self.assert_live_unchanged()
        return result

    def ask(self, request: ModelRequest) -> object:
        if self.remaining_requests <= 0:
            raise BudgetExceeded("Model request limit reached")
        requested = request.max_output_tokens
        if type(requested) is not int or not 0 < requested <= self.max_output_tokens:
            raise BudgetExceeded("Model request exceeds the output-token ceiling")
        try:
            reply = self._model.respond(request)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.model_calls.append(ModelCallTrace(request, error=message))
            raise _ModelExecutionError(message) from exc
        if not isinstance(reply, ModelReply):
            message = "Model adapter must return ModelReply"
            self.model_calls.append(ModelCallTrace(request, error=message))
            raise InvalidModelOutput(message)

        # Preserve the evidence even if a policy subsequently modifies the payload.
        recorded_reply = deepcopy(reply)
        if any(type(count) is not int or count < 0 for count in (reply.input_tokens, reply.output_tokens)):
            message = "Model adapter reported invalid token counts"
            self.model_calls.append(ModelCallTrace(request, recorded_reply, message))
            raise InvalidModelOutput(message)
        if reply.output_tokens > requested:
            message = "Model response exceeded the output-token ceiling"
            self.model_calls.append(ModelCallTrace(request, recorded_reply, message))
            raise BudgetExceeded(message)
        self.model_calls.append(ModelCallTrace(request, recorded_reply))
        self.assert_live_unchanged()
        return reply.payload


def run_decision(
    task: TaskPort,
    model: ModelPort,
    harness: Any,
    memory: Any,
    limits: Limits = Limits(),
) -> DecisionResult:
    """Evaluate one decision without committing it, including in replay mode."""
    frame = task.frame()
    if frame.terminal is not None or not frame.actions:
        raise ValueError("Cannot decide at a terminal frame or one with no legal actions")
    services = _DecisionServices(task, model, frame, limits)
    decision, action_id, notes, error = None, None, (), None
    status = "ready"
    try:
        proposed = harness.decide(frame, memory, services)
        action_id = proposed.action_id
        notes = tuple(proposed.notes)
        if not isinstance(action_id, str) or action_id not in {action.id for action in frame.actions}:
            action_id = None
            raise InvalidModelOutput("Harness did not choose a legal action ID")
        decision = proposed
    except InvalidModelOutput as exc:
        status, error = "invalid_model_output", str(exc)
    except BudgetExceeded as exc:
        status, error = "budget_exhausted", str(exc)
    except _ModelExecutionError as exc:
        status, error = "model_error", str(exc)
    services.assert_live_unchanged()
    return DecisionResult(decision, DecisionTrace(
        frame, action_id, notes, tuple(services.model_calls), tuple(services.previews),
        None, status, error,
    ))


def run_episode(
    task: TaskPort,
    model: ModelPort,
    harness: Any,
    limits: Limits = Limits(),
) -> EpisodeResult:
    """Reset once, then commit at most one validated real action per decision."""
    initial = task.reset()
    events: list[PublicEvent] = []
    traces: list[DecisionTrace] = []
    # An actor's decision can alter only its own private memory. All actors
    # receive the same public committed events, including those preceding
    # their first turn. Nothing carries over across episodes.
    memories: dict[str, Any] = {}
    status = initial.terminal
    error = None
    for _ in range(limits.max_decisions):
        if status is not None:
            break
        actor = task.frame().actor
        if actor not in memories:
            memory = harness.start()
            for event in events:
                memory = harness.observe(event, memory)
            memories[actor] = memory
        result = run_decision(task, model, harness, memories[actor], limits)
        if result.decision is None:
            traces.append(result.trace)
            status, error = result.trace.status, result.trace.error
            break
        decision = result.decision
        event = task.commit(decision.action_id)
        if event.before != result.trace.frame or event.action_id != decision.action_id or event.after != task.frame():
            raise RuntimeError("Task commit returned an inconsistent public event")
        events.append(event)
        traces.append(replace(result.trace, event=event, status="committed"))
        memories[actor] = decision.memory
        for key, memory in memories.items():
            memories[key] = harness.observe(event, memory)
        status = event.after.terminal
    return EpisodeResult(initial, task.frame(), tuple(events), tuple(traces), status or "cap", error)
