"""Per-decision sA-LH reconstruction for one-box Sokoban.

Based on Chenyuan Zhang, Planning and Goal Recognition in Humans and
Machines, Ch. 4 (Eqs. 4.1–4.2) and Ch. 5 (Algorithm 3). The paper does not
publish a runnable planner or every parameter of the original actor.
Config records the choices exposed in this reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import exp, log, sqrt
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sokoban_part_a.env import Board, Position

State = tuple[tuple[int,int], tuple[int,int]]
DELTAS = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}


@dataclass(frozen=True)
class Config:
    # Goal counting is printed in the Chapter 5 pseudo-code.
    # Manhattan distance is an alternative from the Chapter 4 Sokoban example.
    heuristic: str = "goal_count"
    exploration: float = 1.0
    beta: float = .2
    gamma: float = 10000.0
    temperature: float = 5.0
    temperature_convention: str = "temperature"
    stagnation_limit: int = 250
    max_expansions: int = 600
    action_cost: float = 1.0
    stop_if_forced: bool = True

    def __post_init__(self):
        if self.heuristic not in {"goal_count","manhattan"}:
            raise ValueError("heuristic must be goal_count or manhattan")
        if self.temperature_convention not in {"temperature","inverse_temperature"}:
            raise ValueError("unknown temperature_convention")
        if self.beta<=0 or self.gamma<=0 or self.temperature<=0 or self.exploration<0:
            raise ValueError("invalid search parameters")
        if self.max_expansions<1 or self.stagnation_limit<1 or self.action_cost<0:
            raise ValueError("invalid search cap or unit cost")


def plus(x, delta):
    return x[0]+delta[0], x[1]+delta[1]


@lru_cache(maxsize=32768)
def successors(board: Board, state: State) -> dict[str,State]:
    player,box=state
    if box==board.goal:
        return {}
    result={}
    for action,delta in DELTAS.items():
        p=plus(player,delta)
        if not board.walkable(p):
            continue
        b=plus(box,delta) if p==box else box
        if not board.walkable(b):
            continue
        result[action]=(p,b)
    return result


def heuristic(board: Board, state: State, kind: str) -> float:
    _,box=state
    if kind=="goal_count":
        return float(box!=board.goal)
    return float(abs(box[0]-board.goal[0])+abs(box[1]-board.goal[1]))


@dataclass(slots=True)
class Node:
    state: State
    estimate: float
    visits: int = 0
    total: float = 0.0
    evidence: int = 0
    children: dict[str,Node] = field(default_factory=dict)
    expanded: bool = False
    exhausted: bool = False

    @property
    def cost(self):
        # Keep the initial estimate as one pseudo-observation (Ch. 4 example).
        return (self.estimate+self.total)/(self.evidence+1)


def pick_leaf(root: Node, c: float, rng: Random):
    path=[root]
    while path[-1].expanded:
        node=path[-1]
        available=[child for child in node.children.values() if not child.exhausted]
        if not available:
            node.exhausted=True
            if len(path)==1:
                return None
            path.pop()
            continue
        unvisited=[c for c in available if c.visits==0]
        if unvisited:
            choice=rng.choice(unvisited)
        else:
            l=log(max(2,node.visits))
            scores=[-ch.cost+c*sqrt(l/ch.visits) for ch in available]
            best=max(scores)
            choice=rng.choice([ch for ch,v in zip(available,scores) if best-v<=1e-12])
        path.append(choice)
    return path


def stopping_probability(costs: list[float], n: int, cfg: Config) -> float:
    # Equation 4.2 in cost-to-go form (smaller is better).
    vi,vj=sorted(costs)[:2]
    if vi<=0:
        return 1.0
    importance=vi/((1+cfg.beta)*vj-vi)
    if importance<=0:
        return 1.0
    return n/(n+importance*cfg.gamma*exp(-n/importance))


def plan(board: Board, state: State, cfg: Config, rng: Random) -> dict:
    """Return a sampled search's policy, computation, and generated-state set."""
    if not successors(board,state):
        raise ValueError("planning requires an active board with at least one legal action")
    root=Node(state,heuristic(board,state,cfg.heuristic))
    seen={state}
    n=0
    stop="expansion_cap"
    while n<cfg.max_expansions:
        spine=pick_leaf(root,cfg.exploration,rng)
        if spine is None:
            stop="tree_exhausted"
            break
        node=spine[-1]
        if node.state[1]==board.goal:
            stop="goal_found_in_search"
            break
        # The predecessor A-LH excludes duplicated states within each tree.
        # Every legal direction remains available at the root of the new tree.
        for action,s in successors(board,node.state).items():
            if len(spine)>1 and s in seen:
                continue
            seen.add(s)
            node.children[action]=Node(s,heuristic(board,s,cfg.heuristic))
        node.expanded=True
        node.exhausted=not node.children
        n+=1
        for j,ancestor in enumerate(spine):
            ancestor.visits+=1
            for child in node.children.values():
                # Mean backup of new cost estimates at the corresponding depth.
                cost=child.estimate+(len(spine)-j)*cfg.action_cost
                ancestor.total+=cost
                ancestor.evidence+=1
        costs=[cfg.action_cost+ch.cost for ch in root.children.values()]
        if len(costs)==1 and cfg.stop_if_forced:
            stop="forced_first_action"
            break
        if len(costs)>=2 and rng.random()<stopping_probability(costs,n,cfg):
            stop="adaptive_stop"
            break
    values={a:cfg.action_cost+ch.cost for a,ch in root.children.items()}
    minimum=min(values.values())
    scale=(cfg.temperature if cfg.temperature_convention=="inverse_temperature" else 1/cfg.temperature)
    weights={a:exp(-scale*(v-minimum)) for a,v in values.items()}
    denominator=sum(weights.values())
    return {
        "policy":{a:v/denominator for a,v in weights.items()},
        "expansions":n,
        "seen":seen,
        "stop_reason":stop,
        "root_costs":values,
    }
