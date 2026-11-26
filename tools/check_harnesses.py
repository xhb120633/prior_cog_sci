"""Verify the frozen harnesses on small synthetic games, entirely offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import socket
import sys
from types import ModuleType
from unittest.mock import patch


EXPECTED_CONFIG = {"recent_events": 4, "max_lines": 2, "max_depth": 2}
SOURCE_ROOT = Path("meta_harness_pilot")
FROZEN_ROOT = SOURCE_ROOT / "trainval_restart_2026_11_26" / "candidates"
EXPECTED_HASHES = {
    "h0": "b33402e1a55b6f16efe0368bf37cedb961ff7801aaa5a61f8f5044e0073e4bd0",
    "r4_a": "c8a0e4eb1a25a3a3d24ab2c7a0c938e85d1af68a5d781caccb0e0b99821f2007",
    "r8_b": "493fe3ffcad584e74fb7c74a8a792fc2173afde44c9e54b56b60be302f606e46",
}


def load_frozen(root: Path, role: str) -> ModuleType:
    path = root / SOURCE_ROOT / "h0.py" if role == "h0" else root / FROZEN_ROOT / role / "harness.py"
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED_HASHES[role]:
        raise ValueError(f"{role}: source digest does not match the study")
    if role != "h0":
        metadata = json.loads(path.with_name("candidate.json").read_bytes())
        if (metadata["id"] != role or metadata["config"] != EXPECTED_CONFIG
                or metadata["reviewed_input_sha256"]["source_sha256"] != digest):
            raise ValueError(f"{role}: reviewed configuration does not match the study")

    # Execute the hash-checked bytes directly, not a potentially stale .pyc.
    name = f"meta_harness_pilot._review_check_{role}"
    module = ModuleType(name)
    module.__package__ = "meta_harness_pilot"
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        del sys.modules[name]
    return module


def verify(root: Path) -> dict:
    sys.path.insert(0, str(root))
    try:
        with patch.object(socket, "socket", side_effect=AssertionError("Offline check: network blocked")):
            from meta_harness_pilot.mock_models import JsonTextMock, MappingMock
            from meta_harness_pilot.runner import Limits, run_episode
            from meta_harness_pilot.tasks import FourInARowTask, sokoban_fixture

            checked = {}
            for role in EXPECTED_HASHES:
                module = load_frozen(root, role)
                spec = asdict(module.H0Config())
                if spec != EXPECTED_CONFIG:
                    raise ValueError(f"{role}: executable configuration does not match the study")
                cells = []
                for task_name, task_factory in (("sokoban", sokoban_fixture), ("four_in_a_row", FourInARowTask)):
                    for backend in (MappingMock, JsonTextMock):
                        trial = run_episode(
                            task_factory(), backend(), module.H0(config=module.H0Config(**spec)),
                            Limits(max_decisions=3),
                        )
                        if trial.error or not trial.actions or len(trial.actions) != len(trial.events):
                            raise AssertionError(f"{role}: invalid synthetic run for {task_name} / {backend.__name__}")
                        cells.append({"task": task_name, "mock_backend": backend.__name__, "commits": len(trial.actions)})
                checked[role] = {"source_sha256": EXPECTED_HASHES[role], "config": spec, "cells": cells}
    finally:
        sys.path.pop(0)
    return {"scope": "synthetic_interface_check_only", "api_calls": 0, "reran_reported_benchmarks": False, "verified": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.repo.resolve()), indent=2))

