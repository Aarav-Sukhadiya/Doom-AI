"""Practice-mix controller: focus training on the tiers that need it, in stages, from spec-test results.

    python -m doomai.focus --run main20

It reads every new runs/spec/<run>_u<N>.json (25 locked test levels per tier) and writes
runs/<run>/tier_weights.json, which the trainer picks up without restarting. Decisions use the mean of
the last two spec tests, so one lucky result cannot switch stages.

    Stage 1, rooms and keycards   weights 0.5 / 1.5 / 1.5 / 0.5   until tiers 2 and 3 both reach the exit 80%
    Stage 2, combat               weights 1.5 / 0.75 / 0.75 / 0.5 until tier 1 reaches 70% exits and at most 25% deaths
    Stage 3, the gauntlet         weights 1 / 1 / 1 / 1.5          everything learned, put to work on tier 4
If tier 2 or 3 falls below 70% in a later stage, it goes back to stage 1.
    Stage 4, combat and gauntlet  weights 1.25 / 1 / 1 / 1.25     chosen by hand (--stage 4); goes back to
                                                                   stage 1 only if tier 2 or 3 falls below 60% Tier 4 always keeps at least
half weight so it is not forgotten. Stop it with: touch runs/<run>/NOFOCUS
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGES = {
    1: ([0.5, 1.5, 1.5, 0.5], "stage 1: rooms and keycards until tiers 2 and 3 reach 80% exits"),
    2: ([1.5, 0.75, 0.75, 0.5], "stage 2: combat until tier 1 reaches 70% exits and at most 25% deaths"),
    3: ([1.0, 1.0, 1.0, 1.5], "stage 3: the gauntlet, tier 4 with everything learned"),
    4: ([1.25, 1.0, 1.0, 1.25], "stage 4: more combat and gauntlet, rooms and keycards kept at their usual share"),
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def decide(stage: int, recent: list[dict]) -> int:
    def mean(t: int, key: str) -> float:
        return sum(r["summary"]["tiers"][str(t)][key] for r in recent) / len(recent)

    t1_exit, t1_died = mean(1, "exit_rate"), mean(1, "died")
    t2, t3 = mean(2, "exit_rate"), mean(3, "exit_rate")
    if stage == 4:
        return 1 if (t2 < 0.6 or t3 < 0.6) else 4
    if stage > 1 and (t2 < 0.7 or t3 < 0.7):
        return 1
    if stage == 1 and t2 >= 0.8 and t3 >= 0.8:
        return 2
    if stage == 2 and t1_exit >= 0.7 and t1_died <= 0.25:
        return 3
    return stage


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()
    run_dir = REPO / "runs" / args.run
    spec_dir = REPO / "runs" / "spec"
    stage, seen = args.stage, set()

    def write(stage: int, why: str) -> None:
        weights, label = STAGES[stage]
        (run_dir / "tier_weights.json").write_text(json.dumps({"weights": weights, "stage": stage, "reason": f"{label}; {why}"}))
        log(f"-> {label} ({why})")

    write(stage, "start")
    while not (run_dir / "NOFOCUS").exists():
        results = sorted(spec_dir.glob(f"{args.run}_u*.json"), key=lambda f: int(f.stem.split("_u")[1]))
        results = [json.loads(f.read_text()) for f in results]
        results = [r for r in results if r.get("per_tier", 10) >= 25]
        new = [r for r in results if r["update"] not in seen]
        if new:
            seen.update(r["update"] for r in new)
            recent = results[-2:]
            t = {k: recent[-1]["summary"]["tiers"][str(k)] for k in (1, 2, 3, 4)}
            log(f"update {recent[-1]['update']}: exits t1 {t[1]['exit_rate']:.0%} t2 {t[2]['exit_rate']:.0%} t3 {t[3]['exit_rate']:.0%} "
                f"t4 {t[4]['exit_rate']:.0%}, t1 deaths {t[1]['died']:.0%} (stage {stage})")
            nxt = decide(stage, recent)
            if nxt != stage:
                stage = nxt
                write(stage, f"mean of updates {', '.join(str(r['update']) for r in recent)}")
        time.sleep(args.interval)
    log("NOFOCUS found, exiting")


if __name__ == "__main__":
    main()
