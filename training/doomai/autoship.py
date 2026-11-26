"""Unattended shipping: package and push a training snapshot when it clearly beats the shipped agent.

    python -m doomai.autoship --run main5 --shipped-score 4.236 --margin 0.3

Every few minutes it reads runs/<run>/eval_u*.json (written by the trainer's background validation).
When a snapshot's score beats the shipped score by --margin (about the evaluation noise), it copies
the snapshot into submission/weights, runs validate.py, and only if that passes commits and pushes.
Stop it with: touch runs/<run>/NOSHIP
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--shipped-score", type=float, required=True)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--interval", type=int, default=300)
    args = p.parse_args()
    run_dir = REPO / "runs" / args.run
    best = args.shipped_score
    seen: set[str] = set()
    log(f"watching {run_dir}, shipped score {best}, margin {args.margin}")
    while True:
        if (run_dir / "NOSHIP").exists():
            log("NOSHIP found, exiting")
            return
        for ev in sorted(run_dir.glob("eval_u*.json"), key=lambda f: int(f.stem.split("_u")[1])):
            if ev.name in seen:
                continue
            seen.add(ev.name)
            try:
                score = json.loads(ev.read_text())["summary"]["score"]
            except (json.JSONDecodeError, KeyError, OSError):
                seen.discard(ev.name)
                continue
            update = ev.stem.split("_u")[1]
            log(f"{ev.name}: score {score:.3f} (best shipped {best:.3f})")
            if score < best + args.margin:
                continue
            snap = f"model_u{update}.pt"
            if not (run_dir / snap).exists():
                continue
            pkg = run([sys.executable, "-m", "doomai.package", "--run", args.run, "--checkpoint", snap])
            if pkg.returncode != 0:
                log(f"package failed: {pkg.stderr[-500:]}")
                continue
            val = run([sys.executable, "validate.py"])
            if val.returncode != 0:
                log(f"validate.py failed, reverting weights: {val.stdout[-800:]}")
                run(["git", "checkout", "--", "submission/weights"])
                continue
            run(["git", "add", "submission/weights"])
            msg = (f"Auto-ship {args.run} update {update}: validation score {score:.3f}\n\n"
                   f"Beat the previously shipped score {best:.3f} by at least {args.margin} on the same\n"
                   f"validation levels (20 per tier) and passed validate.py.\n\n"
                   "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
            commit = run(["git", "commit", "-m", msg])
            push = run(["git", "push", "origin", "main"], timeout=600)
            log(f"shipped update {update} (commit rc {commit.returncode}, push rc {push.returncode})")
            if push.returncode == 0:
                best = score
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
