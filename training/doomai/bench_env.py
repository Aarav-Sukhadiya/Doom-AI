"""Environment throughput: decisions per second for N parallel workers taking random actions."""

import argparse
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from doom_env import make_env


def worker(wad: str, seconds: float, seed: int, queue) -> None:
    env = make_env(wad, "MAP01", episode_ticks=2100)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    steps = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        action = (rng.random(9) < 0.3).astype(np.int8)
        _, _, term, trunc, _ = env.step(action)
        steps += 1
        if term or trunc:
            env.reset()
    env.close()
    queue.put(steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wad", default="maps/two_rooms.wad")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 12, 14])
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    for n in args.workers:
        queue = mp.Queue()
        procs = [mp.Process(target=worker, args=(args.wad, args.seconds, i, queue)) for i in range(n)]
        for p in procs:
            p.start()
        total = sum(queue.get() for _ in procs)
        for p in procs:
            p.join()
        print(f"{n:2d} workers: {total / args.seconds:7.0f} decisions/s ({total / args.seconds / n:.0f} per worker)", flush=True)


if __name__ == "__main__":
    main()
