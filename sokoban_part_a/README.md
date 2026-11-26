# Sokoban actor environment — Part A

This is an offline replica of the one-box puzzle-solving task in Zhang, Kemp, and Lipovetzky's [Human Goal Recognition as Bayesian Inference](https://arxiv.org/abs/2402.10510). It does not implement the separate observer experiment or the authors' A-LH cognitive model. It runs with Python 3.10+ and no third-party dependencies.

Run these commands from the `prior_cog_sci` repo root.

```powershell
py -3.12 -B -m sokoban_part_a --list
py -3.12 -B -m sokoban_part_a 30031 --play
py -3.12 -B -m sokoban_part_a 30031 --actions right,right,reset,give_up
py -3.12 -B -m unittest sokoban_part_a.test_env -v
py -3.12 -B -m sokoban_part_a.verify
```

`#` wall, `.` floor, `P` player, `B` box, `G` goal, `+` player on goal, `*` box on goal. In the interactive CLI, use `u/d/l/r`, `reset`, and `n` to give up. Movement is one square by walking or pushing. There is no pulling. A blocked input leaves the state and move counters unchanged. Reset restores the initial positions and zeroes the per-attempt move count while preserving the episode-wide counts. Moving the box onto the goal solves the puzzle. Giving up is a different terminal outcome and does not assert the board is unsolvable.

```python
from sokoban_part_a import SokobanEnv, load_boards

env = SokobanEnv(load_boards()["30031"])
print(env.observation.render())
result = env.step("right")
print(result.event, result.observation.render(), sep="\n")
```

The board starts visible, and reset is immediate. The browser's Space-to-reveal controls, wall-clock timing, and ten-second reset wait are not part of this action API. For human-log replay, retain raw timestamps and `space` as UI events separately from the environment actions. Do not treat every logged arrow as a successful move, or an incomplete record as a give-up.

## Data and answer isolation

`data/actor_boards.json` contains the unmodified author board objects for the 23 IDs present for all 95 retained participants in `download/ps_data_2.json`, with the official repository, source commit, and per-board paths recorded. Human participant records are not bundled. The env uses canonical board orientation, which matches the released direction-normalized actor logs.

`env.observation` contains visible state and counters only. Keep the source records, difficulty or solvability metadata, and the verifier outside any actor prompt. Do not use the published `optimal_cost` as ground truth. The separate offline breadth-first search finds 14 solvable and 9 unsolvable boards; board `30031` has an author cost of `-1` but is solvable in 26 legal moves. The other 22 costs match exhaustive search.

This version includes a deterministic grid and action API. It does not use natural-language scenarios, API keys, external models, or network calls.
