"""Offline 2 x 2 compatibility check. No API code or credentials are needed."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from .h0 import H0
from .mock_models import JsonTextMock, MalformedMock, MappingMock
from .runner import Limits, run_episode
from .tasks import FourInARowTask, sokoban_fixture


def _fingerprints(harness: H0) -> tuple[str, str]:
    source = Path(__file__).with_name("h0.py").read_bytes()
    config = json.dumps(asdict(harness.config), sort_keys=True, separators=(",", ":")).encode()
    return sha256(source).hexdigest(), sha256(config).hexdigest()


def run_smoke() -> dict:
    harness = H0()
    fingerprint = _fingerprints(harness)
    cases = []
    limits = Limits(max_decisions=1)
    with patch("socket.socket", side_effect=AssertionError("Network blocked in mock smoke")):
        for task_name, factory in (("sokoban", sokoban_fixture), ("four_in_a_row", FourInARowTask)):
            for backend in (MappingMock, JsonTextMock):
                run = run_episode(factory(), backend(), harness, limits)
                if run.error or len(run.actions) != 1 or run.terminal != "cap":
                    raise AssertionError(f"Expected exactly one legal commit: {task_name}, {backend.__name__}")
                trace = run.decisions[0]
                if trace.request_count != 2 or trace.preview_attempts < 1:
                    raise AssertionError("Expected propose, scratch preview, and choose")
                if _fingerprints(harness) != fingerprint:
                    raise AssertionError("Candidate or configuration changed during adapter swap")
                cases.append({"task": task_name, "model": backend.__name__, "action": run.actions[0],
                              "mock_requests": trace.request_count, "scratch_previews": trace.preview_attempts,
                              "terminal": run.terminal})
            if cases[-1]["action"] == cases[-2]["action"]:
                raise AssertionError("Mock model swap did not affect the selected action")
        bad = run_episode(sokoban_fixture(), MalformedMock(), harness, limits)
        if bad.terminal != "invalid_model_output" or bad.events or bad.final != bad.initial:
            raise AssertionError("A malformed mock response changed the live state")
    return {
        "scope": "offline_interface_smoke_only",
        "h0_sha256": fingerprint[0], "h0_config_sha256": fingerprint[1],
        "cases": cases, "malformed_response": {"status": bad.terminal, "commits": len(bad.events)},
        "api_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
