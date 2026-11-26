"""Local scorer: runs the real submission DoomAgent on held-out generated levels, per tier.

    python -m doomai.evaluate --weights runs/v1/model.pt --split val --per-tier 20

The agent is the exact class the organisers load (submission/agent.py), pointed at the given weights.
Levels come from the validation or locked test seed split, never from training seeds.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TIERS = (1, 2, 3, 4)


def _load_agent(weights: str, greedy: bool):
    sys.path.insert(0, str(REPO / "submission"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("submission_agent_eval", REPO / "submission" / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch

    tmp = Path(tempfile.mkdtemp(prefix="doomai_eval_w_"))
    config = {"greedy": greedy}
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=False)
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        torch.save(state, tmp / "model.pt")
        run_config = Path(weights).resolve().parent / "config.json"
        if run_config.exists():
            config = dict(json.loads(run_config.read_text()), greedy=greedy)
    (tmp / "config.json").write_text(json.dumps(config))
    module.WEIGHTS_DIR = tmp
    from doom_env import spaces as S

    return module.DoomAgent(S.observation_space(), S.action_space(), "cpu")


def _run(job):
    tier, seed, weights, greedy, wad = job
    import torch

    torch.set_num_threads(1)
    from doomai.envs.shaped import ShapedDoomEnv
    from doomai.maps.generate import generate
    from doomai.maps.grid import write_grid_wad

    agent = _load_agent(weights, greedy)
    with tempfile.TemporaryDirectory() as tmp:
        if wad:
            path, ticks = wad, 2100
        else:
            level = generate(tier, seed, difficulty=1.0)
            path = str(write_grid_wad(level.grid, Path(tmp) / "m.wad", player_angle=level.player_angle))
            ticks = level.episode_ticks
        env = ShapedDoomEnv(path, "MAP01", episode_ticks=ticks)
        obs, _ = env.reset(seed=seed)
        agent.reset()
        while True:
            obs, _, term, trunc, info = env.step(agent.act(obs))
            if term or trunc:
                break
        env.close()
    return dict(info["episode_stats"], tier=tier, seed=seed, terms=info["episode_terms"])


def evaluate(weights: str, split: str = "val", per_tier: int = 20, tiers=TIERS, greedy: bool = False, procs: int = 12):
    from doomai.maps.generate import seeds_for

    jobs = [(t, s, weights, greedy, None) for t in tiers for s in seeds_for(split, per_tier, start=10_000 * t)]
    jobs.append((0, 0, weights, greedy, str(REPO / "maps" / "two_rooms.wad")))
    with mp.get_context("forkserver").Pool(procs) as pool:
        results = pool.map(_run, jobs, chunksize=1)
    summary = {}
    for t in sorted({r["tier"] for r in results}):
        rs = [r for r in results if r["tier"] == t]
        summary[f"tier{t}" if t else "two_rooms"] = {
            k: round(float(np.mean([r[k] for r in rs])), 3) for k in ("exit", "dead", "kills", "sectors", "keys", "doors_opened", "steps")
        } | {"n": len(rs)}
    return summary, results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--per-tier", type=int, default=20)
    p.add_argument("--tiers", type=int, nargs="+", default=list(TIERS))
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--out", default="")
    args = p.parse_args()
    summary, results = evaluate(args.weights, args.split, args.per_tier, args.tiers, args.greedy, args.procs)
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))


if __name__ == "__main__":
    main()
