"""Reuse the frozen Part B scoring kernel on archived or pilot-format records.

This module reads data and computes scores on CPU. It does not import a model,
run a game, or regenerate any trajectory or replay observation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


SCORER = Path("runs/sokoban-val-eval-2026-11-25/metrics.py")
SCORER_SHA = "64796d4c602485e8203cf0a664105ce7475a0a642af581194bbfee350e23d28f"
REPO = Path(__file__).resolve().parents[1]
OUTCOMES = frozenset(("solved", "gave_up", "cap"))


def _metrics(repo: Path):
    path = repo / SCORER
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SCORER_SHA:
        raise ValueError(f"Frozen Part B scorer hash mismatch: {path}")
    spec = importlib.util.spec_from_file_location("_sokoban_part_b_frozen_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Part B scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_sokoban(human_by_board: dict, rollout_by_board: dict, replay_by_board: dict,
                  families: dict[str, list[str]], *, seed: int = 20261125, repo: Path = REPO) -> dict:
    """Apply Part B's trajectory and timing scores to canonical record dictionaries.

    Like the original runner, exclude incomplete human attempts from trajectory
    matching and the matching diagnostics. Keep replay rows independently. Refuse
    unsupported model outcomes rather than treating generation failure as give-up.
    """
    complete = {}
    for board, rows in human_by_board.items():
        complete[board] = []
        for row in rows:
            outcome = row.get("outcome")
            if outcome in OUTCOMES:
                complete[board].append(row)
            elif outcome != "incomplete":
                raise ValueError(f"Unsupported human outcome on {board}")
    for board, rows in rollout_by_board.items():
        if any(row.get("outcome") not in OUTCOMES for row in rows):
            raise ValueError(f"Invalid model outcome on {board}, cannot score silently")
    return _metrics(repo).score(complete, rollout_by_board, replay_by_board, families, seed=seed)


def _read_rows(path: Path, board: str, hashes: dict[str, str]) -> list[dict]:
    data = path.read_bytes()
    hashes[path.name] = hashlib.sha256(data).hexdigest()
    rows = []
    for number, line in enumerate(data.splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            raise ValueError(f"Unreadable JSONL at {path.name}, row {number}") from None
        if not isinstance(row, dict) or row.get("board") != board:
            raise ValueError(f"Unexpected board at {path.name}, row {number}")
        rows.append(row)
    return rows


def _compare(actual, expected, at: str = "scores") -> tuple[int, float]:
    """Assert parity across all recorded score/count fields, not just the macro."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Field mismatch at {at}")
        parts = [_compare(actual[k], expected[k], f"{at}.{k}") for k in expected]
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"List length mismatch at {at}")
        parts = [_compare(a, e, f"{at}[{i}]") for i, (a, e) in enumerate(zip(actual, expected))]
    elif type(expected) is float:
        if type(actual) not in (float, int) or not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Score mismatch at {at}: {actual} vs {expected}")
        return 1, abs(actual - expected)
    else:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Value mismatch at {at}: {actual!r} vs {expected!r}")
        return 1, 0.0
    return sum(n for n, _ in parts), max((e for _, e in parts), default=0.0)


def check_archive(run: Path, *, repo: Path = REPO) -> dict:
    """Rescore a complete Part B archival run without any rollout or replay calls."""
    manifest_file = run / "manifest.json"
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["status"] != "complete":
        raise ValueError("Archive is not marked complete")
    families = manifest["families"]
    boards = [b for group in families.values() for b in group]
    if len(set(boards)) != len(boards) or not boards or set(manifest["complete_boards"]) != set(boards):
        raise ValueError("Manifest family/board coverage is inconsistent")
    if any(not isinstance(b, str) or len(b) != 5 or not b.isascii() or not b.isdigit() for b in boards):
        raise ValueError("Invalid board ID in archive manifest")
    groups = {kind: {} for kind in ("human", "rollout", "replay")}
    hashes: dict[str, str] = {}
    for board in boards:
        for kind in groups:
            groups[kind][board] = _read_rows(run / "by-board" / f"{board}-{kind}.jsonl", board, hashes)
    n_model = sum(map(len, groups["rollout"].values()))
    n_replay = sum(map(len, groups["replay"].values()))
    if (n_model != manifest["total_model_rollouts"] or n_replay != manifest["total_matched_rt_decisions"]
            or any(len(groups["human"][b]) != manifest["available_human_trials"][b] for b in boards)
            or any(sum(row["outcome"] in OUTCOMES for row in groups["human"][b])
                   != manifest["scored_full_human_trajectories"][b] for b in boards)):
        raise ValueError("Archived record counts differ from the manifest")

    got = score_sokoban(groups["human"], groups["rollout"], groups["replay"], families,
                        seed=manifest["master_seed"], repo=repo)
    expected_bytes = (run / "scores.json").read_bytes()
    expected = json.loads(expected_bytes)
    n_fields, max_error = _compare(got, expected)
    keys = ("trajectory_distance", "rt_spearman")
    return {
        "scope": "archived_sokoban_part_b_offline_rescore",
        "archive_scope": manifest["scope"],
        "baseline": manifest["baseline"],
        "seed": manifest["master_seed"],
        "rollout_horizon": manifest["rollout_horizon"],
        "boards": len(boards), "families": len(families),
        "human_trajectory_count": sum(manifest["scored_full_human_trajectories"].values()),
        "model_trajectory_count": n_model,
        "replay_row_count": n_replay, "eligible_rt_count": got["macro"]["n_rt"],
        "macro": {key: got["macro"][key] for key in keys},
        "reference_macro": {key: expected["macro"][key] for key in keys},
        "per_family": {f: {key: block[key] for key in keys} for f, block in got["per_family"].items()},
        "fields_compared": n_fields, "max_absolute_error": max_error, "exact_match": max_error == 0.0,
        "scorer_sha256": SCORER_SHA, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reference_scores_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "record_file_count": len(hashes),
        "record_set_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "replicated_new_rollouts": False, "api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-run", type=Path, required=True, help="Existing run with by-board JSONL and reference scores")
    parser.add_argument("--repo", type=Path, default=REPO, help="Path to the repo containing the frozen scorer")
    args = parser.parse_args()
    print(json.dumps(check_archive(args.archive_run, repo=args.repo), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
