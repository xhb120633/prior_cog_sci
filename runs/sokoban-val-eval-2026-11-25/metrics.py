"""Score the locked Sokoban validation protocol; no file or model access."""

from __future__ import annotations

from functools import lru_cache
from hashlib import blake2b
from statistics import mean, median

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


TERMINAL = {"solved": "SOLVED", "gave_up": "GIVE_UP", "cap": "CAP"}
NONTERMINAL = {"U", "D", "L", "R", "RESET"}
AGGREGATE_FIELDS = ("trajectory_distance", "human_human_distance",
                    "model_human_at_reference_n", "rt_spearman")


@lru_cache(maxsize=4096)
def _mask_by_token(sequence: tuple[str, ...]) -> dict[str, int]:
    masks: dict[str, int] = {}
    for i, tok in enumerate(sequence):
        masks[tok] = masks.get(tok, 0) | (1 << i)
    return masks


@lru_cache(maxsize=131072)
def _levenshtein(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Ordinary unit-cost Levenshtein via integer bit-vectors over token positions."""
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return len(b)
    masks = _mask_by_token(a)
    full = (1 << len(a)) - 1
    top = 1 << (len(a) - 1)
    pv, mv, score = full, 0, len(a)
    for token in b:
        eq = masks.get(token, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        score += bool(ph & top) - bool(mh & top)
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = (mh | ~(xv | ph)) & full
        mv = ph & xv
    return score


def _distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if b < a:  # Share symmetric cached entries.
        a, b = b, a
    return _levenshtein(a, b) / max(1, len(a), len(b))


def _matrix(a: list[tuple], b: list[tuple], symmetric: bool = False) -> np.ndarray:
    if symmetric:
        costs = np.zeros((len(a), len(a)), dtype=np.float64)
        for i, x in enumerate(a):
            for j in range(i):
                costs[i, j] = costs[j, i] = _distance(x, a[j])
        return costs
    return np.array([[_distance(x, y) for y in b] for x in a], dtype=np.float64)


def _assignment_mean(costs: np.ndarray) -> float:
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1] or costs.shape[0] == 0:
        raise ValueError(f"Matching requires a nonempty square cost matrix, got {costs.shape}")
    row, col = linear_sum_assignment(costs)
    return float(costs[row, col].mean())


def _tokens(trial: dict) -> tuple[str, ...]:
    seq = tuple(trial["tokens"])
    tail = TERMINAL[trial["outcome"]]
    if not seq or seq[-1] != tail or any(tok not in NONTERMINAL for tok in seq[:-1]):
        raise ValueError(f"Invalid {trial['outcome']} trajectory: {seq[:3]}...{seq[-3:]}")
    return seq


def _rng(seed: int, *names: str) -> np.random.Generator:
    key = ":".join((str(seed), *names)).encode()
    return np.random.default_rng(int.from_bytes(blake2b(key, digest_size=8).digest(), "big"))


def _mean_or_none(values):
    v = [float(x) for x in values if x is not None]
    return mean(v) if v else None


def _trajectory_scores(human: list[dict], model: list[dict], board: str, seed: int,
                       n_ref_splits: int = 24) -> dict:
    h = [_tokens(x) for x in human if x["outcome"] != "incomplete"]
    m = [_tokens(x) for x in model]
    if not h or not m:
        raise ValueError(f"No eligible paired rollouts for {board}")
    ne = min(len(h), len(m))
    nr = min(len(h) // 2, len(m))
    hc = _matrix(h, h, symmetric=True)
    hm = _matrix(h, m)
    rng = _rng(seed, board, "matching")

    if len(h) == len(m):
        distance = _assignment_mean(hm)
        n_draws = 1
    else:
        # Do not let a larger sample cherry-pick the closest counterpart.
        distance = mean(_assignment_mean(hm[np.ix_(rng.permutation(len(h))[:ne],
                                                  rng.permutation(len(m))[:ne])])
                        for _ in range(n_ref_splits))
        n_draws = n_ref_splits

    ref_hh, ref_hm = [], []
    if nr:
        for _ in range(n_ref_splits):
            order = rng.permutation(len(h))
            h1, h2 = order[:nr], order[nr:2*nr]
            m1 = rng.permutation(len(m))[:nr]
            ref_hh.append(_assignment_mean(hc[np.ix_(h1, h2)]))
            # The same sizes and human reference half make the reference comparable.
            ref_hm.append(_assignment_mean(hm[np.ix_(h1, m1)]))
    return {"trajectory_distance": distance,
            "n_trajectory_each": ne, "n_trajectory_draws": n_draws,
            "n_human_trajectories_eligible": len(h), "n_model_trajectories_eligible": len(m),
            "human_human_distance": _mean_or_none(ref_hh),
            "model_human_at_reference_n": _mean_or_none(ref_hm),
            "n_reference_each": nr, "n_reference_draws": n_ref_splits if nr else 0}


def _rt_scores(replay: list[dict]) -> dict:
    paired = []
    for row in replay:
        t = row.get("rt_ms")
        expansions = row.get("expansions")
        if t is None or not np.isfinite(t) or t <= 0:
            continue
        if expansions is None or len(expansions) != 3:
            raise ValueError("Each eligible RT observation requires exactly 3 search-count seeds")
        if any(not np.isfinite(x) or x < 0 or int(x) != x for x in expansions):
            raise ValueError("Search counts must be finite nonnegative integer counts")
        paired.append((float(t), float(median(expansions))))
    d = {"n_rt": len(paired), "n_rt_rows": len(replay), "n_rt_invalid": len(replay)-len(paired)}
    if len(paired) < 3:
        return {**d, "rt_spearman": None, "rt_status": "insufficient_pairs"}
    rt, search = np.asarray(paired, dtype=np.float64).T
    if np.ptp(rt) == 0:
        return {**d, "rt_spearman": None, "rt_status": "constant_human_rt"}
    if np.ptp(search) == 0:
        return {**d, "rt_spearman": 0.0, "rt_status": "constant_search_counts"}
    rho = float(np.corrcoef(rankdata(rt), rankdata(search))[0, 1])
    return {**d, "rt_spearman": min(1.0, max(-1.0, rho)), "rt_status": "ok"}


def _diagnostics(trials: list[dict]) -> dict:
    n = len(trials)
    outcomes = {k: 0 for k in (*TERMINAL, "incomplete")}
    for t in trials:
        outcomes[t["outcome"]] += 1
    give_up = [int(t["moves"]) for t in trials if t["outcome"] == "gave_up"]
    n_reset = sum(int(t["resets"]) > 0 for t in trials)
    return {"n": n, "n_solved": outcomes["solved"], "n_gave_up": outcomes["gave_up"],
            "n_capped": outcomes["cap"], "n_incomplete": outcomes["incomplete"], "n_reset": n_reset,
            "solve_rate": outcomes["solved"]/n,
            "give_up_rate": outcomes["gave_up"]/n,
            "cap_rate": outcomes["cap"]/n,
            "incomplete_rate": outcomes["incomplete"]/n,
            "reset_rate": n_reset/n,
            "moves_before_give_up": _mean_or_none(give_up)}


def _aggregate(rows: list[dict]) -> dict:
    return {**{k: _mean_or_none(row[k] for row in rows) for k in AGGREGATE_FIELDS},
            "n_boards": sum(row.get("n_boards", 1) for row in rows),
            "n_rt": sum(row["n_rt"] for row in rows),
            "n_rt_scored_boards": sum(row.get("n_rt_scored_boards", row["rt_spearman"] is not None)
                                      for row in rows)}


def _pool_diagnostics(rows: list[dict]) -> dict:
    """Episode-count diagnostics, separate from the family-macro primary scores."""
    counts = {k: sum(row[k] for row in rows)
              for k in ("n", "n_solved", "n_gave_up", "n_capped", "n_incomplete", "n_reset")}
    n = counts["n"]
    n_give = counts["n_gave_up"]
    return {**counts, "solve_rate": counts["n_solved"]/n, "give_up_rate": n_give/n,
            "cap_rate": counts["n_capped"]/n, "incomplete_rate": counts["n_incomplete"]/n,
            "reset_rate": counts["n_reset"]/n,
            "moves_before_give_up": sum(row["n_gave_up"]*row["moves_before_give_up"] for row in rows
                                        if row["n_gave_up"])/n_give if n_give else None}


def score(human_by_board: dict, rollout_by_board: dict, replay_by_board: dict,
          families: dict[str, list[str]], seed: int = 20261125) -> dict:
    """Board scores, board-equal families, then family-equal macro.

    Canonical trajectories end in SOLVED, GIVE_UP or CAP. Incomplete human
    episodes remain diagnostic but are excluded from the distance matrix.
    Pass RT rows only for geometrically verified legal pre-move transitions.
    Each eligible row has a paired RT and three seeded search expansion counts.
    """
    board_ids = [b for family_boards in families.values() for b in family_boards]
    if len(board_ids) != len(set(board_ids)) or not board_ids or any(not f for f in families.values()):
        raise ValueError("Families must be nonempty and partition the evaluated boards")
    for name, source in (("human", human_by_board), ("rollout", rollout_by_board),
                         ("replay", replay_by_board)):
        if set(source) != set(board_ids):
            missing, extra = set(board_ids)-set(source), set(source)-set(board_ids)
            raise ValueError(f"{name}: board partition mismatch, missing={sorted(missing)}, extra={sorted(extra)}")
    by_board = {}
    for board in board_ids:
        h, m, r = human_by_board[board], rollout_by_board[board], replay_by_board[board]
        by_board[board] = {**_trajectory_scores(h, m, board, seed), **_rt_scores(r),
                           "human_diagnostics": _diagnostics(h), "model_diagnostics": _diagnostics(m)}
    by_family = {}
    for name, boards in families.items():
        rows = [by_board[b] for b in boards]
        by_family[name] = {**_aggregate(rows), "boards": list(boards),
                           **{k: _pool_diagnostics([row[k] for row in rows])
                              for k in ("human_diagnostics", "model_diagnostics")}}
    family_rows = list(by_family.values())
    macro = {**_aggregate(family_rows), "n_families": len(family_rows),
             **{k: _pool_diagnostics([row[k] for row in family_rows])
                for k in ("human_diagnostics", "model_diagnostics")}}
    return {"per_board": by_board, "per_family": by_family, "macro": macro}
