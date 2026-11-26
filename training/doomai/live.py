"""Watch the current best player, level after level, in a window at normal Doom speed.

    python -m doomai.live                       # the agent on GitHub (submission/weights), forever
    python -m doomai.live --weights runs/main18/model_u400.pt

Each level is first played at full speed with the exact agent the organisers load, seeing the game at
the official 160x120 (a few seconds), and the same game is then shown in a 1024x768 window with the
HUD at 35 frames per second. Nothing is kept on disk. The viewer cycles through combat, multi-room,
keycard and gauntlet levels from the locked test split and the held-out Freedoom 2 maps, prints a
line per level, and switches to a new agent as soon as auto-ship replaces submission/weights.
Close the window or press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import itertools
import os
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "submission" / "weights" / "model.pt"


def levels(seed: int):
    import vizdoom

    from doomai.maps.generate import seeds_for

    freedoom = str(Path(vizdoom.__file__).parent / "freedoom2.wad")
    pools = {t: seeds_for("test", 50, start=10_000 * t) for t in (1, 2, 3, 4)}
    for k in itertools.count(seed):
        for t in (1, 2, 3, 4):
            yield ("tier", t, pools[t][k % 50])
        yield ("freedoom", freedoom, f"MAP{21 + k % 12:02d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="", help="a snapshot; default is the agent on GitHub, reloaded when it changes")
    p.add_argument("--res", default="1024x768", choices=["640x480", "1024x768", "1280x960"])
    p.add_argument("--start", type=int, default=0, help="where to start in the level cycle")
    args = p.parse_args()

    from doomai.maps.generate import generate
    from doomai.maps.grid import write_grid_wad
    from doomai.watch import load_agent, play, replay

    def stamp():
        return SHIPPED.stat().st_mtime if not args.weights and SHIPPED.exists() else 0.0

    agent, seen = load_agent(args.weights, False), stamp()
    label = args.weights or "agent on GitHub"
    print(f"watching {label}", flush=True)
    with tempfile.TemporaryDirectory(prefix="doomai_live_") as tmp:
        for n, (kind, a, b) in enumerate(levels(args.start)):
            if stamp() != seen:
                agent, seen = load_agent("", False), stamp()
                print("auto-ship replaced the agent on GitHub: now watching the new one", flush=True)
            if kind == "tier":
                level = generate(a, b, difficulty=1.0)
                wad = str(write_grid_wad(level.grid, Path(tmp) / f"l{n % 2}.wad", player_angle=level.player_angle))
                map_name, ticks, name = "MAP01", level.episode_ticks, f"tier {a} test level {b}"
            else:
                wad, map_name, ticks, name = a, b, 4200, f"Freedoom 2 {b}"
            lmp = Path(tmp) / f"l{n % 2}.lmp"
            stats, _ = play(agent, wad, map_name, ticks, lmp, seed=n)
            result = "EXIT" if stats["exit"] else "died" if stats["dead"] else "out of time"
            print(f"{name}: {result} after {stats['steps'] * 4 / 35:.0f} s, kills {stats['kills']:.0f}, keys {stats['keys']:.0f}, "
                  f"doors {stats['doors_opened']:.0f}, areas {stats['sectors']:.0f}", flush=True)
            try:
                replay(wad, map_name, ticks, lmp, args.res, True, None)
            except Exception as e:
                print(f"window closed ({type(e).__name__}); stopping", flush=True)
                return


if __name__ == "__main__":
    main()
