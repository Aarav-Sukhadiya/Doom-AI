"""Test environment that follows the organisers' evaluation spec (docs/EVALUATION.md) as closely as we can.

    python -m doomai.spec_test --weights runs/main14/model_u200.pt          # one snapshot
    python -m doomai.spec_test --follow main18                             # every new snapshot of a run

What it copies from the spec:
- Unseen maps in four tiers (25 each, so a few lucky or unlucky maps move the total little), played in order of difficulty with fixed seeds: tier 1 combat & survival,
  tier 2 multi-room exploration with doors, tier 3 keycards and locked doors, tier 4 everything with
  more enemies and two locks. They come from the locked test split of the level generator, which is
  never used for training or for choosing which snapshot ships. The held-out Freedoom 2 maps
  MAP21-32 are reported beside them (not in the total).
- The submitted DoomAgent class, the official environment at skill 3 and each map's tick cap.
- The compute rules: 40 ms per decision of the map's tick cap in total, 10 s for the first act(),
  5 s for any other. An agent that breaks one stands still for the rest of the map (the game keeps
  running to the tick cap, death or exit) and keeps what it earned.
- Partial-credit scoring in the spec's four categories, harder tiers weighted more.

The spec gives categories, not point values, so the points below are our estimate:
    Exit reached       50, plus up to 5 for health left and up to 5 for time left
    Progression        10 per keycard, 3 per door opened, 10 per locked door opened
    Exploration        1 per unique area (sector) visited
    Combat & survival  2 per kill, minus up to 10 for dying early (scaled by time left)
    Tier weights       1, 2, 3, 4
Results go to runs/spec/<run>_u<N>.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "spec"
TIER_WEIGHT = {1: 1, 2: 2, 3: 3, 4: 4}
TIER_NAME = {1: "Combat & survival", 2: "Multi-room exploration", 3: "Long-horizon progression", 4: "Full gauntlet", 9: "Freedoom MAP21-32"}
PER_TIER = 25
FREEDOOM_MAPS = range(21, 33)
FREEDOOM_TICKS = 4200
DECISION_BUDGET_S = 0.040
FIRST_ACT_S = 10.0
ACT_S = 5.0
NOOP = np.zeros(9, np.int8)


def points(r: dict) -> dict:
    left = max(0.0, 1.0 - r["tics"] / max(1, r["ticks"]))
    exit_pts = (50.0 + 5.0 * max(0.0, r["health"]) / 100.0 + 5.0 * left) if r["exit"] else 0.0
    progression = 10.0 * r["keys"] + 3.0 * r["doors"] + 10.0 * r["locked"]
    exploration = 1.0 * r["areas"]
    combat = 2.0 * r["kills"] - (10.0 * left if r["dead"] else 0.0)
    return {"exit": exit_pts, "progression": progression, "exploration": exploration, "combat": combat,
            "total": exit_pts + progression + exploration + combat}


def play(job) -> dict:
    tier, seed, weights, map_name = job
    import torch

    torch.set_num_threads(1)
    from doomai.evaluate import _load_agent
    from doomai.envs.shaped import ShapedDoomEnv

    agent = _load_agent(weights, False)
    with tempfile.TemporaryDirectory() as tmp:
        if tier == 9:
            import vizdoom

            wad, mname, ticks = str(Path(vizdoom.__file__).parent / "freedoom2.wad"), map_name, FREEDOOM_TICKS
        else:
            from doomai.maps.generate import generate
            from doomai.maps.grid import write_grid_wad

            level = generate(tier, seed, difficulty=1.0)
            wad = str(write_grid_wad(level.grid, Path(tmp) / "m.wad", player_angle=level.player_angle))
            mname, ticks = "MAP01", level.episode_ticks
        env = ShapedDoomEnv(wad, mname, episode_ticks=ticks)
        obs, _ = env.reset(seed=seed)
        agent.reset()
        budget = DECISION_BUDGET_S * (ticks // 4)
        used, worst, stopped, why, first = 0.0, 0.0, False, "", True
        while True:
            if stopped:
                action = NOOP
            else:
                t0 = time.perf_counter()
                try:
                    action = np.asarray(agent.act(obs), np.int8)
                    ok = action.shape == (9,) and set(np.unique(action)) <= {0, 1}
                except Exception as e:
                    ok, why = False, f"exception: {type(e).__name__}"
                dt = time.perf_counter() - t0
                used += dt
                worst = max(worst, dt)
                limit = FIRST_ACT_S if first else ACT_S
                first = False
                if not ok or dt > limit or used > budget:
                    stopped, action = True, NOOP
                    why = why or ("invalid action" if not ok else f"act() took {dt:.1f} s" if dt > limit else "compute budget used up")
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        st = env.privileged_state()
        env.close()
    s = info["episode_stats"]
    opened = env._opened & (env._door >= 0)
    r = {"tier": tier, "seed": seed, "map": map_name if tier == 9 else f"tier{tier}-{seed}", "ticks": ticks,
         "tics": int(s["steps"]) * 4, "exit": bool(s["exit"]), "dead": bool(s["dead"]), "health": float(st["health"]),
         "kills": int(s["kills"]), "keys": int(s["keys"]), "doors": int(np.sum(opened & (env._door == 0))),
         "locked": int(np.sum(opened & (env._door > 0))), "areas": int(s["sectors"]), "pickups": int(s.get("pickups", 0)),
         "decisions": int(s["steps"]), "compute_s": round(used, 2), "budget_s": round(budget, 1), "worst_act_ms": round(1000 * worst, 1),
         "stopped": stopped, "stop_reason": why}
    r["points"] = points(r)
    return r


def summarise(results: list[dict]) -> dict:
    out = {"tiers": {}, "categories": {}}
    scored = [r for r in results if r["tier"] in TIER_WEIGHT]
    wsum = sum(TIER_WEIGHT[r["tier"]] for r in scored)
    out["total"] = round(sum(TIER_WEIGHT[r["tier"]] * r["points"]["total"] for r in scored) / max(1, wsum), 2)
    for cat in ("exit", "progression", "exploration", "combat"):
        out["categories"][cat] = round(sum(TIER_WEIGHT[r["tier"]] * r["points"][cat] for r in scored) / max(1, wsum), 2)
    for t in sorted({r["tier"] for r in results}):
        rs = [r for r in results if r["tier"] == t]
        out["tiers"][str(t)] = {
            "name": TIER_NAME[t], "maps": len(rs), "points": round(float(np.mean([r["points"]["total"] for r in rs])), 2),
            "exit_rate": round(float(np.mean([r["exit"] for r in rs])), 3), "died": round(float(np.mean([r["dead"] for r in rs])), 3),
            "out_of_time": round(float(np.mean([not r["exit"] and not r["dead"] for r in rs])), 3),
            "kills": round(float(np.mean([r["kills"] for r in rs])), 2), "keys": round(float(np.mean([r["keys"] for r in rs])), 2),
            "areas": round(float(np.mean([r["areas"] for r in rs])), 2),
            **{f"pts_{c}": round(float(np.mean([r["points"][c] for r in rs])), 2) for c in ("exit", "progression", "exploration", "combat")},
        }
    out["agent_stopped"] = sum(r["stopped"] for r in results)
    out["worst_act_ms"] = max(r["worst_act_ms"] for r in results)
    out["compute_share"] = round(max(r["compute_s"] / r["budget_s"] for r in results), 3)
    return out


def run_one(weights: Path, procs: int) -> Path:
    from doomai.maps.generate import seeds_for

    jobs = [(t, s, str(weights), None) for t in (1, 2, 3, 4) for s in seeds_for("test", PER_TIER, start=10_000 * t)]
    jobs += [(9, 0, str(weights), f"MAP{k:02d}") for k in FREEDOOM_MAPS]
    with mp.get_context("forkserver").Pool(procs) as pool:
        results = pool.map(play, jobs, chunksize=1)
    run, update = weights.parent.name, int(weights.stem.split("_u")[1]) if "_u" in weights.stem else 0
    record = {"run": run, "update": update, "snapshot": f"{run}/{weights.stem}", "at": datetime.now().isoformat(timespec="seconds"), "per_tier": PER_TIER,
              "summary": summarise(results), "maps": results}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{run}_u{update}.json"
    path.write_text(json.dumps(record, indent=1))
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="*", default=[])
    p.add_argument("--follow", default="", help="test every new snapshot of runs/<run>")
    p.add_argument("--procs", type=int, default=3)
    p.add_argument("--interval", type=int, default=120)
    args = p.parse_args()
    for w in args.weights:
        path = run_one(Path(w), args.procs)
        s = json.loads(path.read_text())["summary"]
        print(f"[{datetime.now():%H:%M:%S}] {w}: spec score {s['total']} {s['categories']} stopped {s['agent_stopped']}", flush=True)
    while args.follow:
        run_dir = REPO / "runs" / args.follow
        if (run_dir / "NOSPEC").exists():
            return
        for w in sorted(run_dir.glob("model_u*.pt"), key=lambda f: int(f.stem.split("_u")[1])):
            if not (OUT / f"{args.follow}_u{int(w.stem.split('_u')[1])}.json").exists() and time.time() - w.stat().st_mtime > 20:
                path = run_one(w, args.procs)
                s = json.loads(path.read_text())["summary"]
                print(f"[{datetime.now():%H:%M:%S}] {w.name}: spec score {s['total']} {s['categories']}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
