"""Narrow interfaces shared by the runner, candidate harness, and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Action:
    id: str
    description: str


@dataclass(frozen=True)
class Frame:
    instructions: str
    observation: str
    actions: tuple[Action, ...]
    actor: str
    terminal: str | None = None


@dataclass(frozen=True)
class PublicEvent:
    before: Frame
    action_id: str
    after: Frame
    label: str


@dataclass(frozen=True)
class ScratchView:
    # The runner owns the actual snapshot. The candidate sees only a frame and handle.
    handle: int
    frame: Frame


@dataclass(frozen=True)
class Message:
    role: str
    text: str


@dataclass(frozen=True)
class ModelRequest:
    stage: str
    messages: tuple[Message, ...]
    action_ids: tuple[str, ...]
    max_output_tokens: int
    max_lines: int
    max_depth: int


@dataclass(frozen=True)
class ModelReply:
    payload: object
    input_tokens: int = 0
    output_tokens: int = 0


class InvalidModelOutput(ValueError):
    """The model response does not satisfy the protocol. Never a game action."""


class BudgetExceeded(RuntimeError):
    """The runner refused to start more work under the shared limits."""


class TaskPort(Protocol):
    def reset(self) -> Frame: ...
    def frame(self) -> Frame: ...
    def scratch(self) -> object: ...
    def preview(self, snapshot: object, action_id: str) -> tuple[object, Frame]: ...
    def commit(self, action_id: str) -> PublicEvent: ...


class ModelPort(Protocol):
    def respond(self, request: ModelRequest) -> ModelReply: ...


class DecisionServices(Protocol):
    @property
    def remaining_previews(self) -> int: ...

    @property
    def max_output_tokens(self) -> int: ...

    def ask(self, request: ModelRequest) -> object: ...
    def root(self) -> ScratchView: ...
    def preview(self, parent: ScratchView, action_id: str) -> ScratchView: ...
