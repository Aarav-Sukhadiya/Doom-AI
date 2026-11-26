"""Collect training results into CSV files for the progress dashboard (runs/dashboard/*.csv).

    python -m doomai.dashboard_data --run main18
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
OUT = RUNS / "dashboard"
LINE = re.compile(r"\[\s*([\d.]+) min\] update (\d+) steps (\d+)\s+(\d+)/s kl [\d.]+ clip [\d.]+ ent ([\d.]+) ev ([-\d.]+) \| (.*)")


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        if not rows:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def shipped() -> list[dict]:
    log = subprocess.run(["git", "log", "--reverse", "--format=%cI|%s"], cwd=REPO, capture_output=True, text=True).stdout
    rows = []
    for line in log.splitlines():
        at, msg = line.split("|", 1)
        m = re.match(r"Auto-ship (\w+) update (\d+): validation score ([\d.]+)", msg)
        if m:
            rows.append({"shipped_at": at[:16].replace("T", " "), "snapshot": f"{m[1]}/model_u{m[2]}", "validation_score": float(m[3])})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="the run now training")
    args = p.parse_args()
    ships = shipped()
    write("shipped.csv", ships)
    current = ships[-1]["snapshot"] if ships else ""

    def trained(f):
        d = json.loads(f.read_text())
        model = RUNS / d["run"] / f"model_u{d['update']}.pt"
        return model.stat().st_mtime if model.exists() else 0.0

    spec = []
    for f in sorted((RUNS / "spec").glob("*.json"), key=trained):
        d = json.loads(f.read_text())
        s, t = d["summary"], d["summary"]["tiers"]
        spec.append({
            "snapshot": d["snapshot"], "tested_at": d["at"].replace("T", " ")[:16], "total": s["total"],
            "exit_pts": s["categories"]["exit"], "progression_pts": s["categories"]["progression"],
            "exploration_pts": s["categories"]["exploration"], "combat_pts": s["categories"]["combat"],
            **{f"tier{k}_pts": t[str(k)]["points"] for k in (1, 2, 3, 4)},
            **{f"tier{k}_exit": t[str(k)]["exit_rate"] for k in (1, 2, 3, 4)},
            **{f"tier{k}_died": t[str(k)]["died"] for k in (1, 2, 3, 4)},
            "freedoom_areas": t["9"]["areas"], "freedoom_died": t["9"]["died"],
            "agent_stopped": s["agent_stopped"], "worst_act_ms": s["worst_act_ms"], "compute_share": s["compute_share"],
            "is_shipped": d["snapshot"] == current,
        })
    write("spec_scores.csv", spec)

    if spec:
        latest = max((RUNS / "spec").glob("*.json"), key=trained)
        d = json.loads(latest.read_text())
        write("spec_latest_maps.csv", [{
            "snapshot": d["snapshot"], "map": m["map"], "tier": "Freedoom" if m["tier"] == 9 else f"Tier {m['tier']}",
            "result": "Exit" if m["exit"] else "Died" if m["dead"] else "Out of time", "points": round(m["points"]["total"], 1),
            "kills": m["kills"], "keys": m["keys"], "doors": m["doors"] + m["locked"], "areas": m["areas"],
            "seconds": round(m["tics"] / 35), "worst_act_ms": m["worst_act_ms"],
        } for m in d["maps"]])

    val = []
    for f in sorted(RUNS.glob("main*/eval_u*.json"), key=lambda f: f.stat().st_mtime):
        if not re.fullmatch(r"eval_u\d+", f.stem):
            continue
        s = json.loads(f.read_text())["summary"]
        run, upd = f.parent.name, int(f.stem.split("_u")[1])
        if not run[4:].isdigit() or int(run[4:]) < 9:
            continue
        val.append({"snapshot": f"{run}/model_u{upd}", "run": run, "update": upd, "validation_score": s["score"],
                    **{f"tier{k}_exit": s[f"tier{k}"]["exit"] for k in (1, 2, 3, 4) if f"tier{k}" in s}})
    write("validation.csv", val)

    train = []
    log = RUNS / f"{args.run}.log"
    if log.exists():
        for line in log.read_text().splitlines():
            m = LINE.search(line)
            if not m:
                continue
            row = {"run": args.run, "update": int(m[2]), "minutes": float(m[1]), "decisions_per_s": int(m[4]), "randomness": float(m[5])}
            for t, ex in re.findall(r"t(\d):exit=([\d.]+)", m[7]):
                row[f"tier{t}_exit"] = float(ex)
            fd = re.search(r"freedoom:exit=[\d.]+/areas=(\d+)", m[7])
            row["freedoom_areas"] = int(fd[1]) if fd else None
            if all(f"tier{k}_exit" in row for k in (1, 2, 3, 4)):
                train.append(row)
    write("training.csv", train)

    best = []
    for f in sorted((RUNS / "best_attempts").glob("*/best.json")):
        b = json.loads(f.read_text())
        s = b["stats"]
        best.append({"map": b["map"], "best_score": b["score"], "result": "Exit" if s["exit"] else "Died" if s["dead"] else "Out of time",
                     "snapshot": b["snapshot"], "stored_at": b["stored_at"].replace("T", " ")[:16], "kills": s["kills"], "keys": s["keys"]})
    write("best_attempts.csv", best)
    progress = {name: list(csv.DictReader(open(OUT / f"{name}.csv"))) if (OUT / f"{name}.csv").stat().st_size else []
                for name in ("spec_scores", "spec_latest_maps", "validation", "training", "best_attempts", "shipped")}
    for rows in progress.values():
        for row in rows:
            for k, v in row.items():
                if v in ("True", "False"):
                    row[k] = v == "True"
                elif v == "":
                    row[k] = None
                else:
                    try:
                        row[k] = float(v) if any(c in v for c in ".e") else int(v)
                    except ValueError:
                        pass
    (OUT / "progress.json").write_text(json.dumps(progress))
    print(f"spec {len(spec)}, validation {len(val)}, training {len(train)}, best {len(best)}, shipped {len(ships)}")


if __name__ == "__main__":
    main()
