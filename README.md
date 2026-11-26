# Evaluating and Improving Human-Like Planning in LLM Agents

Code and aggregate results for Hanbo Xie's Prior Computers take-home. The report is provided separately.

| Study | Code | Results |
|---|---|---|
| Part A: Sol as a Sokoban proxy | [Sokoban environment and proxy runner](sokoban_part_a/) | [Board-level planning estimates](runs/sokoban-action-only-batch5-2026-11-25/question-tests/paper-results/board-level-stats.json), [human comparison](runs/sokoban-part-a-human-compare-2026-11-25/) |
| Part B: executable cognitive model discovery | [Sokoban winner](runs/sokoban-sol-continuation-2026-11-25/export/incumbent-program.py.txt), [four-in-a-row winner](runs/fourinarow-discovery-2026-11-25/) | [Sokoban test](runs/sokoban-heldout-final-2026-11-25/run/scores.json), [four-in-a-row test](runs/fourinarow-heldout-final-2026-11-25/test-32-paired.json) |
| Free exploration: fixed-model harness search | [Runner and task adapters](meta_harness_pilot/), [frozen variants](meta_harness_pilot/trainval_restart_2026_11_26/candidates/) | [Search protocol and history](meta_harness_pilot/trainval_restart_2026_11_26/), [matched test and transfer](meta_harness_pilot/frozen_confirmation_2026_11_26/final-report.json) |

## Run the offline checks

From the repository root with Python 3.10 or newer:

~~~sh
python -B -m sokoban_part_a.verify
python -B tools/check_part_b.py
python -B tools/check_harnesses.py
~~~

These commands do not need a key, network access, or any participant records. The Part B check runs the two frozen winners on example positions. The harness check verifies the exact source bytes and exercises the same harness on both tasks with mock replies. Neither command reruns the reported benchmark.

## Rebuild the figures

The figure scripts use the included aggregate results and public board layouts.

~~~sh
python -m pip install -r requirements-figures.txt
python -B figures/plot_tasks.py . --output task_overview.png
python -B figures/plot_part_a.py . --output part_a_board_level.png
python -B figures/plot_sokoban.py . --output sokoban_heldout.png
python -B figures/plot_four_tactics.py runs/fourinarow-heldout-final-2026-11-25/test-32-paired.json --output four_tactics.png
~~~

The Part B Sokoban scorer and its unit tests are included. To run the scorer tests, install from [requirements-scorer.txt](requirements-scorer.txt) and run `python -B -m unittest meta_harness_pilot.test_sokoban_scoring`.

## Data scope

The bundled Sokoban grids are from [Zhang, Kemp, and Lipovetzky](https://github.com/chen-yuan-zhang/bayesian_goal_recognition_unsolvable/tree/5c07e4a5ba57f91285a348aac70241c1af7e9eb4). The four-in-a-row experiments use [van Opheusden et al.'s task and released data](https://github.com/basvanopheusden/fourinarow/tree/d276900ccee7890e9195db592b7fef7c1d39ae54). This snapshot omits participant-level files and individual model executions. The optional section's test uses matched evaluation panels so that H0, the Part B winners, and the two frozen variants see comparable inputs. Its four-in-a-row panel is a fixed sample of the Part B test split, not the full Part B workload.
