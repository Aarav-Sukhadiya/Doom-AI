"""Keep the best attempt on each of the hardest maps, with a replay, a video, a written review and the weights.

    python -m doomai.best_attempts --run main10                 # follow a training run's snapshots
    python -m doomai.best_attempts --weights runs/main5/model_u1800.pt --once

Every new snapshot (runs/<run>/model_uN.pt) plays each hard map a few times. If its best attempt beats
the stored best for that map by --margin or more (reaching the exit is worth 10 on its own), that attempt
becomes the new best, saved in runs/best_attempts/<map>/:

    best.mp4          the attempt at normal Doom speed with the HUD
    best.lmp          the Doom demo (replays exactly, any resolution)
    best.trace.txt    where it got stuck, every hit taken, kills, keys, doors, firing
    best.json         score, stats, which snapshot and attempt
    model.pt          the snapshot's weights, with config.json

The previous best moves to <map>/history/ (demo, review and record only, no video). Every attempt
on every map is also appended to runs/best_attempts/log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "best_attempts"
FREEDOOM_TICKS = 4200
HARD_MAPS = (
    ("tier4_40223", 4, 40223, None),
    ("tier4_40051", 4, 40051, None),
    ("tier4_40204", 4, 40204, None),
    ("freedoom_MAP31", 9, 0, "MAP31"),
)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def map_files(name: str, tier: int, seed: int, freedoom_map) -> tuple[str, str, int]:
    """(wad, map name, episode tics); generated levels are written once into the map's folder."""
    if freedoom_map:
        import vizdoom

        return str(Path(vizdoom.__file__).parent / "freedoom2.wad"), freedoom_map, FREEDOOM_TICKS
    from doomai.maps.generate import generate
    from doomai.maps.grid import write_grid_wad

    level = generate(tier, seed, difficulty=1.0)
    wad = OUT / name / "level.wad"
    wad.parent.mkdir(parents=True, exist_ok=True)
    if not wad.exists():
        write_grid_wad(level.grid, wad, player_angle=level.player_angle)
    return str(wad), "MAP01", level.episode_ticks


def attempt(job) -> dict:
    name, weights, k, wad, map_name, ticks, tmp = job
    import torch

    torch.set_num_threads(1)
    from doomai.evaluate import _load_agent, episode_score
    from doomai.watch import play, summarise

    agent = _load_agent(weights, False)
    torch.manual_seed(1000 + k)
    lmp = Path(tmp) / f"{name}_a{k}.lmp"
    stats, trace = play(agent, wad, map_name, ticks, lmp, seed=k)
    return {"map": name, "attempt": k, "score": round(episode_score(stats), 3), "stats": stats,
            "lmp": str(lmp), "review": summarise(f"{name}, attempt {k}", stats, trace)}


def render(wad: str, map_name: str, ticks: int, lmp: Path, mp4: Path) -> None:
    from doomai.watch import replay

    video = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "35",
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(mp4)], stdin=subprocess.PIPE)
    replay(wad, map_name, ticks, lmp, "640x480", False, video)
    video.stdin.close()
    video.wait()


def store(best: dict, weights: Path, wad: str, map_name: str, ticks: int) -> None:
    folder = OUT / best["map"]
    folder.mkdir(parents=True, exist_ok=True)
    old = folder / "best.json"
    if old.exists():
        prev = json.loads(old.read_text())
        hist = folder / "history" / f"{prev['snapshot'].replace('/', '_')}_a{prev['attempt']}"
        hist.mkdir(parents=True, exist_ok=True)
        for f in ("best.lmp", "best.trace.txt", "best.json"):
            if (folder / f).exists():
                shutil.move(str(folder / f), hist / f)
        (folder / "best.mp4").unlink(missing_ok=True)
    shutil.copy(best["lmp"], folder / "best.lmp")
    (folder / "best.trace.txt").write_text(best["review"] + "\n")
    shutil.copy(weights, folder / "model.pt")
    config = weights.parent / "config.json"
    if config.exists():
        shutil.copy(config, folder / "config.json")
    render(wad, map_name, ticks, folder / "best.lmp", folder / "best.mp4")
    record = {k: best[k] for k in ("map", "attempt", "score", "stats", "snapshot", "stored_at", "previous_best")}
    (folder / "best.json").write_text(json.dumps(record, indent=2))


def process(weights: Path, attempts: int, margin: float, procs: int) -> None:
    snapshot = f"{weights.parent.name}/{weights.stem}"
    maps = {name: map_files(name, tier, seed, fmap) for name, tier, seed, fmap in HARD_MAPS}
    with tempfile.TemporaryDirectory(prefix="doomai_best_") as tmp:
        jobs = [(name, str(weights), k, *maps[name], tmp) for name in maps for k in range(attempts)]
        with mp.get_context("forkserver").Pool(procs) as pool:
            results = pool.map(attempt, jobs, chunksize=1)
        with open(OUT / "log.jsonl", "a") as f:
            for r in results:
                f.write(json.dumps({"snapshot": snapshot, "map": r["map"], "attempt": r["attempt"], "score": r["score"], "stats": r["stats"]}) + "\n")
        for name in maps:
            mine = [r for r in results if r["map"] == name]
            best = max(mine, key=lambda r: r["score"])
            prev_file = OUT / name / "best.json"
            prev = json.loads(prev_file.read_text())["score"] if prev_file.exists() else None
            scores = ", ".join(f"{r['score']:.1f}" for r in mine)
            if prev is not None and best["score"] < prev + margin:
                log(f"{snapshot} {name}: attempts {scores}; best {best['score']:.1f} does not beat stored {prev:.1f} by {margin}")
                continue
            best.update(snapshot=snapshot, stored_at=datetime.now().isoformat(timespec="seconds"), previous_best=prev)
            store(best, weights, *maps[name])
            log(f"{snapshot} {name}: attempts {scores}; NEW BEST {best['score']:.1f} (was {prev}) stored")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="", help="follow runs/<run>/model_u*.pt as they appear")
    p.add_argument("--weights", nargs="*", default=[], help="snapshots to process first (e.g. the shipped one)")
    p.add_argument("--attempts", type=int, default=4, help="attempts per map per snapshot")
    p.add_argument("--margin", type=float, default=1.0, help="how much better a new best must be")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done_file = OUT / "processed.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()

    def handle(w: Path) -> None:
        key = str(w.resolve().relative_to(REPO))
        if key in done:
            return
        log(f"playing {key}")
        process(w, args.attempts, args.margin, args.procs)
        done.add(key)
        with open(done_file, "a") as f:
            f.write(key + "\n")

    for w in args.weights:
        handle(Path(w))
    while not args.once and args.run:
        run_dir = REPO / "runs" / args.run
        if (run_dir / "NOBEST").exists():
            log("NOBEST found, exiting")
            return
        snaps = sorted(run_dir.glob("model_u*.pt"), key=lambda f: int(f.stem.split("_u")[1]))
        for w in snaps:
            if time.time() - w.stat().st_mtime > 20:
                handle(w)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
