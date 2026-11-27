# AGENTS.md

Guide for AI coding agents working in this repository. Read `ARCHITECTURE.md` for how the system works.

## What this is

A DOOM-playing agent for DevForge Track 1. The organisers import `submission/agent.py` (`DoomAgent`) and run it in
their frozen copy of `env/` on unseen maps. Everything under `training/` builds and tests it.

## Never change

- `env/`, `docs/`, `tests/`, `maps/`, `docker/`, `Dockerfile`, `validate.py`: the organisers' template. The scorer
  ignores our copies; they must stay identical to the template.
- The submitted agent must decide only through trained parameters. No path-finding, aimbots (no `atan2` aiming),
  rule trees or use of `privileged_state()` in `submission/`. Masks from observation fields are allowed.
- `submission/` must not import from `training/`. It may import only `doomnet`, `doomobs`, `landmarks`, numpy and
  torch.

## Invariants

- Trainer and agent parity: both build inputs with `submission/doomobs.observe` and the same `LandmarkTracker`
  settings, and must choose the same actions (`training/tests/test_parity.py`). Change both sides together.
- New network inputs are residual branches with a zero-initialised output layer, named with a prefix that
  `ppo.py --init` accepts as missing (`slot_`, `crumb_`, `map_`, `weapon.`, `space_`), and switched on by a config key.
- `submission/weights/config.json` decides which branches the agent builds; weights load with `strict=True`.
- Weights are stored with Git LFS (`git lfs ls-files` must list `submission/weights/model.pt`).
- `runs/` is not committed. Snapshots, logs, test results and videos live there.

## Commands

```bash
uv sync --group train
uv run pytest                                             # all tests (template + ours)
uv run python validate.py                                 # organisers' checks; must pass before any push
uv run python validate.py --docker --no-gpu --image devforge-doom   # same, inside the official image
PYTHONPATH=training uv run python -m doomai.train --run NAME ...     # train (flags in training/README.md)
PYTHONPATH=training uv run python -m doomai.spec_test --weights runs/NAME/model_uN.pt
PYTHONPATH=training uv run python -m doomai.final_pick --per-tier 100 runs/NAME/model_uA.pt runs/NAME/model_uB.pt
PYTHONPATH=training uv run python -m doomai.package --run NAME --checkpoint model_uN.pt   # copy into submission/
```

Stop a training run with `touch runs/NAME/STOP`; resume with `--resume runs/NAME/latest.pt` (the `--lr` given then
applies). `runs/NAME/tier_weights.json` changes the tier mix while it runs. `NOSHIP`, `NOSPEC`, `NOBEST` and
`NOFOCUS` in a run folder stop the matching watcher.

## Where things are set

| What | Where |
| --- | --- |
| Rewards | `training/doomai/envs/shaped.py`, `REWARD` |
| PPO and curriculum settings | `training/doomai/ppo.py` arguments |
| Level tiers, seeds, tick caps | `training/doomai/maps/generate.py` |
| Observation encoding, action heads, USE cooldown | `submission/doomobs.py` |
| Network | `submission/doomnet.py` |
| Odometry, landmarks, breadcrumbs, level map | `submission/landmarks.py` |
| Spec-test point values | `training/doomai/spec_test.py`, `points()` |
| Ship gates | `training/doomai/autoship.py` |

## Known traps

- USE and SELECT_NEXT_WEAPON fire only on a fresh press; pressing USE on a moving door reverses it.
- Killing every enemy does not end a map; only the exit, death or the tick cap do.
- `pkill -f PATTERN` or `pgrep -f PATTERN` where PATTERN appears in your own command line matches your own shell and
  kills it. Select processes with `ps -eo pid,args | awk '/python -m doomai\.X/ && !/awk/'` instead.
- `torch.load` of a checkpoint dict needs `weights_only=False`; `submission/weights/model.pt` is a plain state dict.
- Validation (20 levels per tier) and the spec test (25) are noisy: one tier moves about 9 points between runs.
  Compare candidates on 100 levels per tier before replacing the submitted agent.
- Freedoom 2 MAP31 starts the player in a cage it cannot leave.
