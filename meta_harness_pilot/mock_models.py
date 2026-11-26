"""Deterministic offline ModelPort fixtures. No credentials or network access."""

from __future__ import annotations

import json

from .ports import ModelReply, ModelRequest


def _reply(request: ModelRequest, payload: object) -> ModelReply:
    # Reproducible mock usage for the trace, not estimates of provider tokens.
    return ModelReply(
        payload=payload,
        input_tokens=sum(len(message.text.split()) for message in request.messages),
        output_tokens=len(json.dumps(payload, sort_keys=True).split()),
    )


def _payload(request: ModelRequest, *, prefer_second: bool) -> dict:
    actions = request.action_ids
    if prefer_second and len(actions) > 1:
        actions = (actions[1], actions[0], *actions[2:])
    if request.stage == "propose":
        # Only the root action IDs are known. A one-step line is deliberate.
        return {"lines": [[action] for action in actions[:request.max_lines]]
                if request.max_depth > 0 else []}
    if request.stage == "choose":
        return {"action_id": actions[0] if actions else "", "plan_note": "mock plan"}
    raise ValueError(f"Unknown model request stage: {request.stage!r}")


class MappingMock:
    """Mapping-native backend. Prefers the first available action."""

    def respond(self, request: ModelRequest) -> ModelReply:
        return _reply(request, _payload(request, prefer_second=False))


class JsonTextMock:
    """Text-returning backend with JSON parsing contained in the adapter."""

    def respond(self, request: ModelRequest) -> ModelReply:
        raw = json.dumps(_payload(request, prefer_second=True), ensure_ascii=False)
        return _reply(request, json.loads(raw))


class MalformedMock:
    """Emits a malformed payload or an unavailable action at the chosen stage."""

    def __init__(self, kind: str = "shape", stage: str = "choose") -> None:
        if kind not in {"shape", "illegal"} or stage not in {"propose", "choose"}:
            raise ValueError("Expected shape/illegal and propose/choose")
        self.kind = kind
        self.stage = stage

    def respond(self, request: ModelRequest) -> ModelReply:
        if request.stage != self.stage:
            return _reply(request, _payload(request, prefer_second=False))
        unavailable = "mock-unavailable-action"
        while unavailable in request.action_ids:
            unavailable += "-"
        if self.kind == "shape":
            payload: object = {"lines": 9} if self.stage == "propose" else {"action_id": 9}
        elif self.stage == "propose":
            payload = {"lines": [[unavailable]]}
        else:
            payload = {"action_id": unavailable, "plan_note": "mock illegal selection"}
        return _reply(request, payload)
