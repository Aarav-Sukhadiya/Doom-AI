"""Score the hand-weighted network on the same held-out levels doomai.evaluate uses.

    python -m handcrafted.compare --tier 4 --per-tier 20
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def run_one(job):
    tier, seed, difficulty, trace = job
    from doomai.envs.shaped import ShapedDoomEnv
    from doomai.maps.generate import generate
    from doomai.maps.grid import write_grid_wad
    from handcrafted.handnet import HandAgent

    agent = HandAgent()
    level = generate(tier, seed, difficulty=difficulty)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(write_grid_wad(level.grid, Path(tmp) / "m.wad", player_angle=level.player_angle))
        env = ShapedDoomEnv(path, "MAP01", episode_ticks=level.episode_ticks)
        obs, _ = env.reset(seed=seed)
        agent.reset()
        lines = []
        while True:
            action = agent.act(obs)
            if trace:
                st = env.privileged_state()
                lines.append(f"{len(lines):4d} pos=({st['x']:.0f},{st['y']:.0f}) hp={st['health']:.0f} a={''.join(map(str, action))} "
                             + " ".join(f"{k}={v:.2f}" for k, v in agent.probe("gE", "gEsc", "gS", "gK", "gX", "gW", "gI", "gF", "theta").items()))
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        env.close()
    out = dict(info["episode_stats"], tier=tier, seed=seed)
    if trace:
        out["trace"] = lines
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, default=4)
    p.add_argument("--per-tier", type=int, default=20)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--trace", type=int, default=-1, help="print a step trace for this seed index")
    args = p.parse_args()
    from doomai.maps.generate import seeds_for

    seeds = seeds_for("val", args.per_tier, start=10_000 * args.tier)
    if args.trace >= 0:
        r = run_one((args.tier, seeds[args.trace], args.difficulty, True))
        print("\n".join(r.pop("trace")))
        print(r)
        return
    with mp.get_context("forkserver").Pool(args.procs) as pool:
        results = pool.map(run_one, [(args.tier, s, args.difficulty, False) for s in seeds], chunksize=1)
    for r in results:
        print(r["seed"], " ".join(f"{k}={r[k]:.0f}" for k in ("exit", "dead", "keys", "doors_opened", "kills", "sectors", "steps")))
    print(json.dumps({k: round(float(np.mean([r[k] for r in results])), 3)
                      for k in ("exit", "dead", "kills", "sectors", "keys", "doors_opened", "steps")}))


if __name__ == "__main__":
    main()
