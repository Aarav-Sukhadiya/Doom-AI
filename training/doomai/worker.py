"""Game worker process for the PPO trainer: steps ShapedDoomEnv games on procedural levels.

Kept free of torch so each worker process stays small (it only needs numpy, ViZDoom and the
level generator); the learner in ppo.py does all the network work.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submission"))

from doomobs import observe  # noqa: E402
from landmarks import LandmarkTracker  # noqa: E402

TIER_COUNT = 4


def worker_main(remote, worker_id: int, envs_per_worker: int, difficulty, weights, seed: int, map_dir: str, reward_cfg: dict | None = None, hard_share: float = 0.0):
    from doomai.envs.shaped import ShapedDoomEnv
    from doomai.maps.generate import generate, seeds_for
    from doomai.maps.grid import write_grid_wad

    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(seed)
    build_rng = np.random.default_rng(seed + 7919)
    train_seeds = seeds_for("train", 200000)
    envs = [None] * envs_per_worker
    trackers = [LandmarkTracker() for _ in range(envs_per_worker)]
    tiers = [0] * envs_per_worker
    plays = [0] * envs_per_worker
    counter = [0]
    builder = ThreadPoolExecutor(max_workers=1)

    def build(i):
        w = np.array(weights[:], np.float64)
        tier = int(build_rng.choice(TIER_COUNT, p=w / w.sum())) + 1
        if build_rng.random() < hard_share:
            d = 1.0
        else:
            d = float(np.clip(difficulty[tier - 1] + build_rng.normal(0, 0.1), 0.0, 1.0))
        level = generate(tier, int(build_rng.choice(train_seeds)), difficulty=round(d, 2))
        counter[0] += 1
        path = Path(map_dir) / f"w{worker_id}_e{i}_{counter[0]}.wad"
        write_grid_wad(level.grid, path, player_angle=level.player_angle)
        env = ShapedDoomEnv(str(path), "MAP01", episode_ticks=level.episode_ticks, reward=reward_cfg, subgoal_grid=level.grid)
        return env, tier

    pending = [builder.submit(build, i) for i in range(envs_per_worker)]

    def new_level(i):
        env, tier = pending[i].result()
        pending[i] = builder.submit(build, i)
        if envs[i] is not None:
            old = Path(envs[i].wad)
            envs[i].close()
            old.unlink(missing_ok=True)
        envs[i] = env
        tiers[i] = tier
        plays[i] = 0

    def reset(i):
        if envs[i] is None or plays[i] >= 2:
            new_level(i)
        plays[i] += 1
        obs, _ = envs[i].reset(seed=int(rng.integers(1 << 30)))
        trackers[i].reset()
        return observe(obs, trackers[i], None)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send([reset(i) for i in range(envs_per_worker)])
            elif cmd == "step":
                out = []
                for i, action in enumerate(data):
                    obs, reward, term, trunc, info = envs[i].step(action)
                    done = term or trunc
                    stats = None
                    if done:
                        stats = dict(info["episode_stats"], tier=tiers[i], terms=info["episode_terms"], cap=envs[i].episode_ticks / 4)
                        pre = reset(i)
                    else:
                        pre = observe(obs, trackers[i], action)
                    out.append((pre, reward, done, stats))
                remote.send(out)
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        traceback.print_exc()
        remote.send(("error", traceback.format_exc()))
    finally:
        for env in envs:
            if env is not None:
                env.close()
        for f in pending:
            try:
                f.result()[0].close()
            except Exception:
                pass
        builder.shutdown(wait=False)
