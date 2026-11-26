"""Watch the trained agent play, at normal Doom speed (35 frames per second) with the HUD.

The agent plays a level exactly as at evaluation (official DoomEnv, 160x120 observations, one decision
every 4 tics) while ViZDoom records a demo. The demo is then replayed in a visible window at a higher
resolution and real-time speed, and can also be saved as an MP4.

    python -m doomai.watch --tier 3 --seed 2              # one generated validation level, in a window
    python -m doomai.watch --wad maps/two_rooms.wad       # the organisers' test map
    python -m doomai.watch --all --video runs/demo.mp4    # one level per tier + two_rooms, saved to video
    python -m doomai.watch --all --no-window --video runs/demo.mp4   # video only

--weights runs/<run>/model_uN.pt watches a training snapshot instead of the shipped submission.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import vizdoom as vd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "watch"
RESOLUTIONS = {"640x480": vd.ScreenResolution.RES_640X480, "1024x768": vd.ScreenResolution.RES_1024X768, "1280x960": vd.ScreenResolution.RES_1280X960}


def load_agent(weights: str, greedy: bool):
    if weights:
        from doomai.evaluate import _load_agent

        return _load_agent(weights, greedy)
    sys.path.insert(0, str(REPO / "submission"))
    import importlib.util

    from doom_env import spaces as S

    spec = importlib.util.spec_from_file_location("submission_agent_watch", REPO / "submission" / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.DoomAgent(S.observation_space(), S.action_space(), "cpu")
    agent.greedy = agent.greedy or greedy
    return agent


def play(agent, wad: str, map_name: str, ticks: int, lmp: Path, seed: int) -> tuple[dict, list]:
    from doomai.envs.shaped import ShapedDoomEnv

    env = ShapedDoomEnv(wad, map_name, episode_ticks=ticks)
    obs, _ = env.reset(seed=seed, options={"record_path": str(lmp)})
    agent.reset()
    trace = []
    while True:
        action = agent.act(obs)
        seen = env._sees_enemy(obs)
        obs, _, term, trunc, info = env.step(action)
        st = env.privileged_state()
        terms = info["reward_terms"]
        trace.append({
            "t": max(st["tic"] / 35.0, trace[-1]["t"] if trace else 0.0),
            "health": float(st["health"]),
            "ammo": float(obs["vitals"][3]),
            "fired": bool(np.asarray(action)[6]),
            "enemy": seen,
            "cells": len(env._cells_seen),
            "kill": terms["kill"] > 0,
            "hurt": -terms["damage_taken"] / 0.01 if terms["damage_taken"] else 0.0,
            "event": ",".join(k for k in ("key", "door", "locked_door", "new_sector", "exit", "death", "futile_use") if terms.get(k, 0) > 0 or (k in ("death", "futile_use") and terms.get(k, 0) < 0)),
        })
        if term or trunc:
            break
    env.close()
    return info["episode_stats"], trace


def summarise(name: str, stats: dict, trace: list) -> str:
    """Plain-text review of one episode: where it got stuck, what hurt it, how it used ammo."""
    out = [f"== {name}"]
    fired = [s for s in trace if s["fired"]]
    blind = [s for s in fired if not s["enemy"]]
    out.append(f"decisions {len(trace)}, fired on {len(fired)} ({100 * len(fired) / max(1, len(trace)):.0f}%), "
               f"{len(blind)} of those with no enemy in view; ammo {trace[0]['ammo']:.0f} -> {trace[-1]['ammo']:.0f}")
    stuck, start = [], 0
    for i in range(1, len(trace)):
        if trace[i]["cells"] > trace[start]["cells"] or i == len(trace) - 1:
            if trace[i]["t"] - trace[start]["t"] >= 10:
                stuck.append((trace[start]["t"], trace[i]["t"]))
            start = i
    out.append("stuck (no new ground for 10 s+): " + (", ".join(f"{a // 60:.0f}:{a % 60:04.1f}-{b // 60:.0f}:{b % 60:04.1f}" for a, b in stuck) or "never"))
    stuck_time = sum(b - a for a, b in stuck)
    out.append(f"time stuck {stuck_time:.0f} s of {trace[-1]['t']:.0f} s")
    for s in trace:
        mark = []
        if s["hurt"]:
            mark.append(f"hurt {s['hurt']:.0f} (health {s['health']:.0f})")
        if s["kill"]:
            mark.append("kill")
        if s["event"]:
            mark.append(s["event"])
        if mark:
            out.append(f"  {s['t'] // 60:.0f}:{s['t'] % 60:04.1f}  " + "; ".join(mark))
    result = "EXIT" if stats["exit"] else "DIED" if stats["dead"] else "OUT OF TIME"
    out.append(f"result {result}: kills {stats['kills']:.0f}, keys {stats['keys']:.0f}, doors {stats['doors_opened']:.0f}, areas {stats['sectors']:.0f}")
    return "\n".join(out)


def replay(wad: str, map_name: str, ticks: int, lmp: Path, res: str, window: bool, video) -> None:
    from doom_env import spaces as S
    from doom_env.wrapper import IWAD, SKILL

    g = vd.DoomGame()
    g.set_doom_game_path(IWAD)
    g.set_doom_scenario_path(wad)
    g.set_doom_map(map_name)
    g.set_doom_skill(SKILL)
    g.set_mode(vd.Mode.PLAYER)
    g.set_screen_resolution(RESOLUTIONS[res])
    g.set_screen_format(vd.ScreenFormat.RGB24)
    g.set_render_hud(True)
    g.set_render_crosshair(True)
    g.set_render_weapon(True)
    g.set_render_messages(True)
    g.set_window_visible(window)
    g.set_sound_enabled(False)
    g.set_available_buttons([getattr(vd.Button, b) for b in S.BUTTONS])
    g.set_episode_timeout(ticks)
    g.init()
    g.replay_episode(str(lmp))
    frame_time = 1.0 / 35.0
    next_frame = time.perf_counter()
    while not g.is_episode_finished():
        state = g.get_state()
        if video is not None and state is not None:
            video.stdin.write(np.ascontiguousarray(state.screen_buffer).tobytes())
        g.advance_action()
        if window:
            next_frame += frame_time
            delay = next_frame - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    if window:
        time.sleep(1.5)
    g.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="index into the validation seeds of that tier")
    p.add_argument("--wad", default="", help="play this WAD instead of a generated level")
    p.add_argument("--map", default="MAP01", help="map inside --wad, e.g. MAP21 of freedoom2.wad")
    p.add_argument("--freedoom", default="", help="shortcut: Freedoom 2 map, e.g. MAP21 (held-out from training)")
    p.add_argument("--ticks", type=int, default=0, help="episode length in tics for --wad (default 2100, Freedoom 4200)")
    p.add_argument("--all", action="store_true", help="one level per tier plus maps/two_rooms.wad")
    p.add_argument("--weights", default="", help="a training snapshot; default is the shipped submission")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--res", default="1024x768", choices=sorted(RESOLUTIONS))
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--video", default="", help="also save the replay as this .mp4")
    args = p.parse_args()

    from doomai.maps.generate import generate, seeds_for
    from doomai.maps.grid import write_grid_wad

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.all:
        for tier in (1, 2, 3, 4):
            jobs.append(("tier", tier, seeds_for("val", args.seed + 1, start=10_000 * tier)[args.seed]))
        jobs.append(("wad", str(REPO / "maps" / "two_rooms.wad"), 0))
    elif args.freedoom:
        jobs.append(("wad", str(Path(vd.__file__).parent / "freedoom2.wad"), 0))
        args.map, args.ticks = args.freedoom.upper(), args.ticks or 4200
    elif args.wad:
        jobs.append(("wad", str(Path(args.wad).resolve()), 0))
    else:
        jobs.append(("tier", args.tier, seeds_for("val", args.seed + 1, start=10_000 * args.tier)[args.seed]))

    w, h = (int(v) for v in args.res.split("x"))
    video = None
    if args.video:
        video = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "35",
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(Path(args.video).resolve())],
            stdin=subprocess.PIPE)

    agent = load_agent(args.weights, args.greedy)
    for kind, what, seed in jobs:
        if kind == "tier":
            level = generate(what, seed, difficulty=1.0)
            wad = str(write_grid_wad(level.grid, OUT / f"tier{what}_seed{seed}.wad", player_angle=level.player_angle))
            ticks, name, map_name = level.episode_ticks, f"tier {what} (validation seed {seed})", "MAP01"
        else:
            map_name = args.map if not args.all else "MAP01"
            wad, ticks, name = what, args.ticks or 2100, f"{Path(what).name} {map_name}"
        lmp = OUT / f"{Path(wad).stem}_{map_name}.lmp"
        stats, trace = play(agent, wad, map_name, ticks, lmp, seed)
        report = summarise(name, stats, trace)
        print(report, flush=True)
        lmp.with_suffix(".trace.txt").write_text(report + "\n")
        replay(wad, map_name, ticks, lmp, args.res, not args.no_window, video)
    if video is not None:
        video.stdin.close()
        video.wait()
        print(f"saved {args.video}")


if __name__ == "__main__":
    main()
