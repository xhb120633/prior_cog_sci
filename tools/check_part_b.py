"""Run offline smoke checks for the two frozen Part B winners."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import exp, isfinite
from pathlib import Path
from random import Random
import socket
import sys
from types import ModuleType
from unittest.mock import patch


PINS = {
    "runs/sokoban-sol-continuation-2026-11-25/export/incumbent-program.py.txt":
        "12f9188f164c502fef7c503f4474ee0dbad916f4b1c475765a905697953c4f22",
    "runs/fourinarow-baseline-eval-2026-11-25/model.py":
        "1b7799b99c991ef544a1e6ff04b0f841ba1e7f789f982dd8754d000b5e7aa7f6",
    "runs/fourinarow-baseline-eval-2026-11-25/frozen-params.json":
        "b14ed70309c2913b523f9338d9994ac79511fb27a3f3d24a4147d32f02d7ddc4",
    "runs/fourinarow-discovery-2026-11-25/candidate_model.py":
        "0f6cd24a3f4a3e1be53c56e03e784b6f35fd8fc257c82805d304b0d2f764a91c",
    "runs/fourinarow-discovery-2026-11-25/champion-spec.json":
        "152f86edc1fb51968473dc9c01b3d20cfb0d0de3438c5d0dfcd8e730e079f038",
}
SOKOBAN, FOUR_BASE, FOUR_PARAMS, FOUR_MODEL, FOUR_SPEC = tuple(PINS)


def pinned(root: Path, name: str) -> bytes:
    data = (root / name).read_bytes()
    if sha256(data).hexdigest() != PINS[name]:
        raise ValueError(f"Frozen input changed: {name}")
    return data


def sokoban_check(root: Path) -> dict:
    from sokoban_part_a.env import DELTA, Outcome, SokobanEnv, load_boards

    board = load_boards()["30002"]
    game = SokobanEnv(board)
    source = pinned(root, SOKOBAN)
    rng = Random(0)
    history, actions = [], []
    work = 0

    def successors(state):
        player, box = state
        if box == board.goal:
            return {}
        result = {}
        for action, (dr, dc) in DELTA.items():
            new_player = (player[0] + dr, player[1] + dc)
            new_box = (box[0] + dr, box[1] + dc) if new_player == box else box
            if board.walkable(new_player) and board.walkable(new_box):
                result[action.value] = (new_player, new_box)
        return result

    def unused_baseline(*_):
        raise AssertionError("The frozen winner does not call the baseline")

    for _ in range(40):
        if game.observation.outcome is not Outcome.ACTIVE:
            break
        state = game.observation.player, game.observation.box
        # The evaluation namespace supplies exp. Run the exact source afresh
        # for every decision, with public state and official move semantics.
        scope = {"exp": exp}
        exec(compile(source, SOKOBAN, "exec"), scope)
        proposal = scope["plan"](state, tuple(history), board.goal,
                                 board.walkable, successors, unused_baseline, rng)
        weights = proposal["policy"]
        legal = set(successors(state)) | {"reset", "give_up"}
        if (not weights or any(not isfinite(weight) or weight < 0 for weight in weights.values())
                or any(weight > 0 and action not in legal for action, weight in weights.items())
                or max(weights.values()) <= 0):
            raise AssertionError("The Sokoban winner returned an invalid policy")
        expansion_count = proposal["expansions"]
        if type(expansion_count) is not int or expansion_count < 0:
            raise AssertionError("The Sokoban winner returned an invalid expansion count")
        # A deterministic argmax illustrates execution. The benchmark sampled
        # the policy rather than taking its maximum.
        action = max(weights, key=lambda key: (weights[key], key))
        if game.step(action).event == "blocked":
            raise AssertionError("A selected action was blocked by the environment")
        work += expansion_count
        history.append((state, action))
        actions.append(action)
    if not actions:
        raise AssertionError("No Sokoban decision was executed")
    return {"candidate": "g21-candidate-3", "public_board": "30002",
            "actions": actions, "outcome": game.observation.outcome.value, "search_work": work}


def four_check(root: Path) -> dict:
    loaded = {}
    for name, relpath in (("model", FOUR_BASE), ("_review_four_winner", FOUR_MODEL)):
        module = ModuleType(name)
        module.__file__ = str(root / relpath)
        sys.modules[name] = module
        exec(compile(pinned(root, relpath), relpath, "exec"), module.__dict__)
        loaded[name] = module
    base, candidate = loaded["model"], loaded["_review_four_winner"]
    params = base.Params(**json.loads(pinned(root, FOUR_PARAMS))["parameters"][0])
    simulate = candidate.build_simulator(json.loads(pinned(root, FOUR_SPEC)))
    # A synthetic legal position with an open four-in-a-row threat.
    black = (1 << 10) | (1 << 11) | (1 << 12)
    white = (1 << 18) | (1 << 20) | (1 << 27)
    decisions = {}
    for mode, rt_mode in (("choice", False), ("rt", True)):
        move, work = simulate(black, white, 1, params, Random(0), rt_mode=rt_mode)
        if not (0 < move <= base.FULL and move & (move - 1) == 0
                and move & (black | white) == 0 and type(work) is int and work >= 0):
            raise AssertionError(f"The four-in-a-row winner returned an invalid {mode} result")
        decisions[mode] = {"square_zero_based": move.bit_length() - 1, "search_work": work}
    return {"candidate": "r3-c1", "synthetic_position": True, "decisions": decisions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().repo.resolve()
    sys.path.insert(0, str(root))
    with patch.object(socket, "socket", side_effect=AssertionError("Offline check: network blocked")):
        result = {"scope": "smoke_check_only", "api_calls": 0, "reran_reported_benchmarks": False,
                  "sokoban": sokoban_check(root), "four_in_a_row": four_check(root)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
