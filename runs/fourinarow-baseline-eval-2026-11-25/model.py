"""Paper-spec simulator for the 4x9 four-in-a-row cognitive model.

References: van Opheusden et al. (2023), Methods and Supplement Algorithms
1–4 and §10. This is a Python reconstruction, not the stock C++ entry
point (which uses a deterministic iteration limit in its full-board mode).
It uses the same 495 nonzero-weight line-feature instances as the legacy
board, a geometric search budget, and a separate root-stability RT mode.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import hypot, isfinite, log1p
from operator import index
from random import Random


ROWS, COLS = 4, 9
SIZE = ROWS * COLS
FULL = (1 << SIZE) - 1
RT_STABILITY = 50
_CENTRE = tuple(
    1.0 / hypot(r - (ROWS - 1) * 0.5, c - (COLS - 1) * 0.5)
    for r in range(ROWS)
    for c in range(COLS)
)


def _masks(mask):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


def _make_patterns():
    windows, kinds = [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(ROWS):
            for c in range(COLS):
                if not (0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS):
                    continue
                cells = tuple((r + k * dr) * COLS + c + k * dc for k in range(4))
                line = sum(1 << i for i in cells)
                ids = {}
                # A four-cell window has 6 two-piece patterns, 4 three-piece
                # patterns, and 1 winning pattern. Two adjacent pieces count
                # only when the other two cells in that window are empty.
                for n in (2, 3, 4):
                    for positions in combinations(range(4), n):
                        shape = sum(1 << cells[k] for k in positions)
                        ids[shape] = len(kinds)
                        kinds.append(
                            int(positions[1] - positions[0] != 1)
                            if n == 2
                            else n - 1
                        )
                lookup = {}
                for n in (1, 2, 3):
                    for occupied in combinations(cells, n):
                        shape = sum(1 << i for i in occupied)
                        moves = tuple((i, ids[shape | (1 << i)]) for i in cells if i not in occupied)
                        lookup[shape] = ids.get(shape, -1), moves
                windows.append((line, lookup))
    return tuple(windows), tuple(kinds)


_WINDOWS, _FEATURE_KINDS = _make_patterns()
WIN_MASKS = tuple(line for line, _ in _WINDOWS)
NUM_FEATURES = len(_FEATURE_KINDS)
assert len(WIN_MASKS) == 45 and NUM_FEATURES == 495
_WIN_DIRECTIONS = tuple(
    (d, sum(1 << i for i in starts))
    for d, starts in (
        (1, (r * COLS + c for r in range(ROWS) for c in range(COLS - 3))),
        (COLS, range(COLS)),
        (COLS + 1, range(COLS - 3)),
        (COLS - 1, range(3, COLS)),
    )
)


def has_won(mask: int) -> bool:
    """Check horizontal, vertical, or diagonal agreement without wraparound."""
    for delta, starts in _WIN_DIRECTIONS:
        pairs = mask & (mask >> delta)
        if pairs & (pairs >> (2 * delta)) & starts:
            return True
    return False


@dataclass(frozen=True)
class Params:
    pruning_threshold: float
    stopping_prob: float
    feature_drop: float
    lapse_rate: float
    active_scale: float
    center_weight: float
    connected_two: float
    unconnected_two: float
    three: float
    four: float

    def __post_init__(self):
        vals = (self.pruning_threshold, self.stopping_prob, self.feature_drop, self.lapse_rate,
                self.active_scale, self.center_weight, self.connected_two, self.unconnected_two,
                self.three, self.four)
        if not all(isfinite(x) for x in vals):
            raise ValueError("parameters must be finite")
        if not 0.0 < self.stopping_prob <= 1.0:
            raise ValueError("stopping_prob must be in (0, 1]")
        if not (0.0 <= self.feature_drop <= 1.0 and 0.0 <= self.lapse_rate <= 1.0):
            raise ValueError("feature_drop and lapse_rate must be in [0, 1]")
        if self.pruning_threshold < 0.0 or self.active_scale < 0.0:
            raise ValueError("pruning_threshold and active_scale must be nonnegative")


@lru_cache(maxsize=256)
def _raw_weights(p):
    w = (p.connected_two, p.unconnected_two, p.three, p.four)
    return tuple(w[k] for k in _FEATURE_KINDS)


class _Draws:
    __slots__ = "p", "rng", "raw", "feature_weights"

    def __init__(self, p, rng):
        self.p = p
        self.rng = rng
        self.raw = _raw_weights(p)
        self.feature_weights = {} if 0.0 < p.feature_drop < 1.0 else None

    def feature(self, f):
        if f < 0 or self.p.feature_drop == 1.0:
            return 0.0
        if self.feature_weights is None:
            return self.raw[f]
        try:
            return self.feature_weights[f]
        except KeyError:
            # Lazy independent draws are equivalent to dropping all features
            # upfront because unseen masks cannot have affected earlier search.
            w = 0.0 if self.rng.random() < self.p.feature_drop else self.raw[f]
            self.feature_weights[f] = w
            return w


class _Node:
    __slots__ = (
        "black", "white", "player", "centre", "move", "value", "lo", "hi",
        "children", "best", "frontier",
    )

    def __init__(self, black, white, player, centre, move=0, value=0.0, outcome=None):
        self.black, self.white, self.player = black, white, player
        self.centre, self.move, self.value = centre, move, value
        self.children = None
        self.best = self.frontier = None
        if outcome is None:
            self.lo, self.hi = -1, 1
        else:
            self.lo = self.hi = outcome
            self.value = 10000.0 * outcome

    def backup(self):
        if self.player == 0:
            picker = max
        else:
            picker = min
        children = self.children
        self.lo = picker(c.lo for c in children)
        self.hi = picker(c.hi for c in children)
        self.best = picker(children, key=_value)
        self.value = self.best.value
        if self.lo == self.hi:
            # Exact outcome in the retained tree, preferring the current
            # heuristic ordering among equally proved continuations.
            result = self.lo
            self.best = next(c for c in children if c.lo == result == c.hi)
            self.value = 10000.0 * result
            self.frontier = None
        else:
            # Continue expanding the strongest unresolved variation. A
            # known terminal leaf can be the current choice, but cannot expand.
            self.frontier = picker((c for c in children if c.lo != c.hi), key=_value)


def _value(node):
    return node.value


def _expand(node, draws):
    p = draws.p
    black, white, side = node.black, node.white, node.player
    own, opp = (black, white) if side == 0 else (white, black)
    empty = FULL ^ (own | opp)
    # The opponent is the active player in every evaluated child, so their
    # count of line features carries the active scale. Centre is not scaled.
    base = node.centre if side == 0 else -node.centre
    changes = [0.0] * SIZE
    win_moves = 0
    feature = draws.feature
    scale = p.active_scale
    for line, lookup in _WINDOWS:
        x, y = own & line, opp & line
        if x:
            if y:
                continue
            idx, moves = lookup[x]
            w = feature(idx)
            base += w
            if len(moves) == 1:
                win_moves |= line ^ x
            for cell, added in moves:
                changes[cell] += feature(added) - w
        elif y:
            idx, moves = lookup[y]
            if idx >= 0:
                w = scale * feature(idx)
                base -= w
                for cell, _ in moves:
                    changes[cell] += w

    gauss = draws.rng.gauss
    w_center = p.center_weight
    scores = []
    for action in _masks(empty):
        cell = action.bit_length() - 1
        delta_center = w_center * _CENTRE[cell]
        score = base + delta_center + changes[cell] + gauss(0.0, 1.0)
        scores.append((score, action, delta_center))
    scores.sort(key=lambda x: x[0], reverse=True)
    best = scores[0][0]
    sign = 1 if side == 0 else -1
    children = []
    # Terminal values must not act as an oracle inside pruning. A noisy or
    # ignored 4-line may cause a winning action to be pruned, by design.
    for score, action, dc in scores:
        if best - score > p.pruning_threshold:
            break
        if win_moves & action:
            outcome = sign
        elif action == empty:
            outcome = 0
        else:
            outcome = None
        b, w = (black | action, white) if side == 0 else (black, white | action)
        children.append(_Node(b, w, 1 - side, node.centre + sign * dc,
                              action, sign * score, outcome))
    node.children = children


def simulate(
    black: int,
    white: int,
    player: int,
    params: Params,
    rng: Random,
    rt_mode: bool = False,
) -> tuple[int, int]:
    """Sample a legal one-hot action and count the actual expanded nodes.

    Inputs are raw bitboards in the archive convention, not tile indices.
    player=0 means black, player=1 white. A lapse chooses uniformly before
    searching (zero expansions). A non-lapse expands the root before the
    geometric stopping rule can end the search.
    """
    black, white, player = index(black), index(white), index(player)
    occupied = black | white
    if min(black, white) < 0 or occupied > FULL or black & white or player not in (0, 1):
        raise ValueError("invalid 4x9 position")
    if occupied == FULL or has_won(black) or has_won(white):
        raise ValueError("cannot simulate a terminal position")
    if params.lapse_rate and rng.random() < params.lapse_rate:
        return rng.choice(tuple(_masks(FULL ^ occupied))), 0
    draws = _Draws(params, rng)
    stopping = params.stopping_prob
    budget = 1 if stopping == 1.0 else 1 + int(log1p(-rng.random()) / log1p(-stopping))
    wc = params.center_weight
    centre = wc * (sum(_CENTRE[b.bit_length() - 1] for b in _masks(black))
                   - sum(_CENTRE[b.bit_length() - 1] for b in _masks(white)))
    root = _Node(black, white, player, centre)
    previous, consecutive, iterations = 0, 0, 0
    for _ in range(budget):
        path = [root]
        while path[-1].frontier is not None:
            path.append(path[-1].frontier)
        _expand(path[-1], draws)
        for node in reversed(path):
            node.backup()
        iterations += 1
        action = root.best.move
        if root.lo == root.hi:
            break
        if rt_mode:
            consecutive = consecutive + 1 if previous == action else 0
            if consecutive >= RT_STABILITY:
                break
            previous = action
    return action, iterations
