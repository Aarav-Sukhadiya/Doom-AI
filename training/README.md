# Training

Everything used to build, train and test the agent. The submitted agent in `submission/` never imports from
here; the network code it shares with training lives in `submission/` and training imports it from there.

## What is in here

| File | What it does |
| --- | --- |
| `doomai/maps/grid.py` | Turns a small text drawing of a level into a real DOOM map (UDMF), with doors, coloured locked doors, keys, enemies, items and exits. |
| `doomai/maps/generate.py` | Makes random levels in four tiers: combat rooms, multi-room levels with doors, one key and lock, two keys and locks. Every level is solved by a search before it is used, and seeds are split 80/10/10 into practice, validation and locked test levels. |
| `doomai/maps/walker.py` | A scripted walker used only by the tests, to prove generated levels can be finished in the real game. |
| `doomai/envs/shaped.py` | The official environment plus our practice rewards: exit, keys, doors, new areas and ground, kills, pickups, dodging fireballs, and costs for damage, dying, wasted ammo and pressing USE on a door it has no key for. These rewards exist only in training. |
| `doomai/envs/subgoals.py` | A training-only hint: reward for getting closer to the next key, its door, then the exit, worked out from the level's solution. |
| `doomai/worker.py` | A game process that plays several practice levels at once. It has no PyTorch, so dozens of them fit in memory. |
| `doomai/ppo.py` | The learning loop (recurrent PPO): plays, learns, adjusts difficulty per tier, tests itself every 200 updates and saves snapshots. |
| `doomai/train.py` | A small launcher for `ppo.py` that keeps the game processes light. |
| `doomai/evaluate.py` | Plays a snapshot through the real submitted agent class on validation or test levels and scores it. |
| `doomai/spec_test.py` | Our copy of the organisers' evaluation: locked test levels in four tiers, the official time and compute limits, points in the four scoring categories. |
| `doomai/autoship.py` | Watches a training run and ships a snapshot to `submission/weights` only when it beats the current one on validation and does not fall behind on the spec test, and `validate.py` passes. |
| `doomai/focus.py` | Shifts practice between tiers while training runs, in stages chosen from spec-test results, and moves back if a tier slips. |
| `doomai/final_pick.py` | Plays the leading snapshots on 100 locked levels per tier and ranks them, to choose the agent to submit. |
| `doomai/package.py` | Copies a snapshot and its settings into `submission/weights`. |
| `doomai/best_attempts.py` | Keeps the best run on each of the hardest levels, with a video, a replay and a written review. |
| `doomai/watch.py` | Plays one level with the real agent and shows it at normal speed, or saves it as a video with a step-by-step review. |
| `doomai/live.py` | Shows the current agent playing, level after level, in a window. |
| `doomai/dashboard_data.py` | Collects test results into one file for the progress dashboard. |
| `tests/` | Our tests: level maker, generator, rewards, the rules checks, and that training and the submitted agent choose the same moves. |

## Running it

```bash
uv sync --group train
PYTHONPATH=training uv run python -m doomai.train --run myrun --slots --crumbs --map --weapon --space \
    --fire-mask --entity-range 4 --tiers 1 2 3 4 --tier-weights 1.25 1 1 1.25 --freedoom-share 0.15 \
    --hard-share 0.35 --subgoal 0.05 --hours 6
PYTHONPATH=training uv run python -m doomai.autoship --run myrun --shipped-score 11.4
PYTHONPATH=training uv run python -m doomai.spec_test --weights runs/myrun/model_u400.pt
PYTHONPATH=training uv run python -m doomai.final_pick runs/myrun/model_u400.pt runs/myrun/model_u600.pt
uv run pytest training/tests
```

Snapshots, logs, test results and videos go to `runs/`, which is not committed. Progress can be followed in
TensorBoard (`tensorboard --logdir runs`).

## Rules we follow

The organisers allow anything during training, so practice uses custom levels, extra rewards, easier difficulty
and the game's hidden state (positions, which doors are locked) to work out rewards. None of that reaches the
submitted agent: it sees only the official observations and decides everything with its trained network.
