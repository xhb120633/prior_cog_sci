"""Bounded, inspectable candidate space for the validation-only 4x9 pilot.

Candidate data never executes as code. All controls act on search or on the
same transformation of every author parameter vector. There are no
participant, puzzle-ID, labelling, or per-position action overrides.
"""
from dataclasses import fields, replace
from functools import lru_cache
from math import isfinite, log1p
from operator import index
from pathlib import Path
import sys

_BASE_DIR = Path(__file__).resolve().parent.parent / "fourinarow-baseline-eval-2026-11-25"
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))
import model as _base
if Path(_base.__file__).resolve() != _BASE_DIR / "model.py":
    raise ImportError("Expected the pinned sibling baseline model.py")

PARAMETER_NAMES = frozenset(f.name for f in fields(_base.Params))
PROBABILITY_NAMES = frozenset(("stopping_prob", "feature_drop", "lapse_rate"))
MAX_STRUCTURAL_ITERATIONS = 8192
SEARCH_DEFAULTS = {
    "frontier": "principal",
    "gap": 1.0,
    "alternative_interval": 3,
    "pruning": "fixed",
    "root_prune_scale": 1.0,
    "deep_prune_scale": 1.0,
    "deep_ply": 2,
    "stability_limit": None,
    "stability_min_iterations": 1,
}


def _record(x, allowed, name):
    if not isinstance(x, dict) or set(x).difference(allowed):
        raise ValueError(f"{name} must be an object with only supported keys")
    return x


def _number(x, low, high, name):
    if type(x) not in (float, int) or not (low <= x <= high):
        raise ValueError(f"{name} must be a finite number in [{low},{high}]")
    return float(x)


def _integer(x, low, high, name):
    if type(x) is not int or not low <= x <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return x


def parse_spec(raw: dict) -> dict:
    """Validate and canonicalize the allowed JSON fields and bounded values."""
    root = _record(raw, ("search", "scale", "offset"), "candidate")
    search = SEARCH_DEFAULTS | _record(root.get("search", {}), SEARCH_DEFAULTS, "search")
    if search["frontier"] not in ("principal", "close_alternatives"):
        raise ValueError("Unsupported frontier")
    if search["pruning"] not in ("fixed", "adaptive"):
        raise ValueError("Unsupported pruning")
    search["gap"] = _number(search["gap"], 0, 6, "gap")
    search["alternative_interval"] = _integer(search["alternative_interval"], 2, 8, "alternative_interval")
    for key in ("root_prune_scale", "deep_prune_scale"):
        search[key] = _number(search[key], 0.55, 1.8, key)
    search["deep_ply"] = _integer(search["deep_ply"], 1, 4, "deep_ply")
    if search["stability_limit"] is not None:
        search["stability_limit"] = _integer(search["stability_limit"], 8, 80, "stability_limit")
    search["stability_min_iterations"] = _integer(
        search["stability_min_iterations"], 1, 50, "stability_min_iterations"
    )

    if search["frontier"] == "principal":
        for key in ("gap", "alternative_interval"):
            search[key] = SEARCH_DEFAULTS[key]
    if search["pruning"] == "fixed" or (
        search["root_prune_scale"] == 1.0 and search["deep_prune_scale"] == 1.0
    ):
        for key in ("pruning", "root_prune_scale", "deep_prune_scale", "deep_ply"):
            search[key] = SEARCH_DEFAULTS[key]
    if search["stability_limit"] is None:
        search["stability_min_iterations"] = SEARCH_DEFAULTS["stability_min_iterations"]

    scales = {
        key: _number(x, 0.6, 1.65, f"scale.{key}")
        for key, x in sorted(_record(root.get("scale", {}), PARAMETER_NAMES, "scale").items())
    }
    offsets = {
        key: _number(x, -0.08, 0.08, f"offset.{key}")
        for key, x in sorted(_record(root.get("offset", {}), PROBABILITY_NAMES, "offset").items())
    }
    return {
        "search": search,
        "scale": {key: x for key, x in scales.items() if x != 1.0},
        "offset": {key: x for key, x in offsets.items() if x != 0.0},
    }


def build_simulator(spec: dict):
    """Return a callable with the same interpretation as baseline.simulate."""
    config = parse_spec(spec)
    search, scales, offsets = config["search"], config["scale"], config["offset"]
    structural = search != SEARCH_DEFAULTS
    if not structural and not scales and not offsets:
        return _base.simulate

    changed = set(scales) | set(offsets)

    @lru_cache(maxsize=512)
    def shared_params(p):
        if not changed:
            return p
        adjusted = {}
        for key in changed:
            before = getattr(p, key)
            value = before * scales.get(key, 1.0) + offsets.get(key, 0.0)
            if key in PROBABILITY_NAMES:
                # For stopping, bound the mean budget to at most 2x the
                # original even when the submitted absolute offset is negative.
                lower = 0.5 * before if key == "stopping_prob" else 0.0
                value = max(lower, min(1.0, value))
            adjusted[key] = value
        return replace(p, **adjusted)

    if not structural:
        def parameter_only(black, white, player, params, rng, rt_mode=False):
            return _base.simulate(black, white, player, shared_params(params), rng, rt_mode)
        return parameter_only

    root_scale = search["root_prune_scale"]
    deep_scale = search["deep_prune_scale"]
    deep_ply = search["deep_ply"]
    explore = search["frontier"] == "close_alternatives"
    interval = search["alternative_interval"]
    margin = search["gap"]
    limit = search["stability_limit"]
    limit_check_after = search["stability_min_iterations"]

    @lru_cache(maxsize=512)
    def depths(p):
        def widen(k):
            return replace(p, pruning_threshold=p.pruning_threshold * k) if k != 1.0 else p
        return widen(root_scale), widen(deep_scale)

    def simulate(black, white, player, params, rng, rt_mode=False):
        p = shared_params(params)
        black, white, player = index(black), index(white), index(player)
        occupied = black | white
        if min(black, white) < 0 or occupied > _base.FULL or black & white or player not in (0, 1):
            raise ValueError("invalid 4x9 position")
        if occupied == _base.FULL or _base.has_won(black) or _base.has_won(white):
            raise ValueError("cannot simulate a terminal position")
        if p.lapse_rate and rng.random() < p.lapse_rate:
            return rng.choice(tuple(_base._masks(_base.FULL ^ occupied))), 0

        draws = _base._Draws(p, rng)
        root_params, deep_params = depths(p)
        stop = p.stopping_prob
        budget = 1 if stop == 1.0 else 1 + int(log1p(-rng.random()) / log1p(-stop))
        budget = min(MAX_STRUCTURAL_ITERATIONS, budget)
        centre = p.center_weight * (
            sum(_base._CENTRE[b.bit_length() - 1] for b in _base._masks(black))
            - sum(_base._CENTRE[b.bit_length() - 1] for b in _base._masks(white))
        )
        root = _base._Node(black, white, player, centre)
        previous = consecutive = 0
        for i in range(budget):
            path = [root]
            principal = root.frontier
            # Periodically reconsider another plausible, still unresolved
            # root route. The attached subtree uses the normal best-first
            # rule. Never alter the normal choice/backup rule or score ties.
            if explore and principal is not None and i % interval == 0:
                choices = (c for c in root.children if c is not principal and c.lo != c.hi)
                pick = max if player == 0 else min
                alt = pick(choices, key=_base._value, default=None)
                if alt is not None and abs(alt.value - principal.value) <= margin:
                    path.append(alt)
            while path[-1].frontier is not None:
                path.append(path[-1].frontier)
            depth = len(path) - 1
            draws.p = root_params if depth == 0 else deep_params if depth >= deep_ply else p
            _base._expand(path[-1], draws)
            for node in reversed(path):
                node.backup()

            action = root.best.move
            steps = i + 1
            if root.lo == root.hi:
                break
            if rt_mode or limit is not None:
                consecutive = consecutive + 1 if action == previous else 0
                previous = action
                baseline_stop = rt_mode and consecutive >= _base.RT_STABILITY
                custom_stop = limit is not None and steps >= limit_check_after and consecutive >= limit
                if baseline_stop or custom_stop:
                    break
        return action, steps

    return simulate

