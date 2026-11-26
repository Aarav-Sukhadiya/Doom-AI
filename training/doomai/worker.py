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
FREEDOOM_TIER = 5
FREEDOOM_TRAIN_MAPS = 20
FREEDOOM_TICKS = 4200
FREEDOOM1_MAPS = [f"E{e}M{m}" for e in range(1, 5) for m in range(1, 10)]
REAL_MAP_SKILLS = (1, 2, 3)
REAL_MAP_SKILL_P = (0.25, 0.35, 0.4)


def worker_main(remote, worker_id: int, envs_per_worker: int, difficulty, weights, seed: int, map_dir: str, reward_cfg: dict | None = None, hard_share: float = 0.0, freedoom_share: float = 0.0, entity_range: float = 1.0, mapping: bool = False):
    from doomai.envs.shaped import ShapedDoomEnv
    from doomai.maps.generate import generate, seeds_for
    from doomai.maps.grid import write_grid_wad

    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(seed)
    build_rng = np.random.default_rng(seed + 7919)
    train_seeds = seeds_for("train", 200000)
    envs = [None] * envs_per_worker
    trackers = [LandmarkTracker(entity_range, mapping) for _ in range(envs_per_worker)]
    tiers = [0] * envs_per_worker
    plays = [0] * envs_per_worker
    counter = [0]
    builder = ThreadPoolExecutor(max_workers=1)

    real_maps = []
    if freedoom_share > 0:
        import vizdoom

        vz = Path(vizdoom.__file__).parent
        real_maps = [(str(vz / "freedoom2.wad"), f"MAP{k:02d}") for k in range(1, FREEDOOM_TRAIN_MAPS + 1)]
        if (vz / "freedoom1.wad").exists():
            real_maps += [(str(vz / "freedoom1.wad"), m) for m in FREEDOOM1_MAPS]

    def build(i):
        if real_maps and build_rng.random() < freedoom_share:
            import doom_env.wrapper as wrapper

            wad, map_name = real_maps[int(build_rng.integers(len(real_maps)))]
            # Practice only: real maps sometimes on easier skills (fewer, weaker monsters) so the agent
            # lives long enough to learn their layouts. Evaluation always uses the official skill 3.
            wrapper.SKILL = int(build_rng.choice(REAL_MAP_SKILLS, p=REAL_MAP_SKILL_P))
            try:
                env = ShapedDoomEnv(wad, map_name, episode_ticks=FREEDOOM_TICKS, reward=reward_cfg)
            finally:
                wrapper.SKILL = 3
            return env, FREEDOOM_TIER
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
            old = Path(envs[i].wad).resolve()
            envs[i].close()
            if old.parent == Path(map_dir).resolve():
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
