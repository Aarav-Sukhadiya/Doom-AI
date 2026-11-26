"""Pick the agent to submit: play the leading snapshots on many locked test levels and rank them.

    python -m doomai.final_pick --per-tier 100 runs/main20/model_u200.pt runs/main20/model_u1400.pt ...

Each candidate plays --per-tier locked levels in each of the four tiers plus the 12 held-out Freedoom maps
through doomai.spec_test (official time and compute limits), into runs/final/. With 100 levels per tier a
tier's exit rate is known to about +-4.5 points, against +-9 at 25. The table ranks candidates by the
spec-test total and shows each tier, so the pick can be checked by eye. It changes nothing in submission/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doomai.spec_test import run_one

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "final"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("weights", nargs="+")
    p.add_argument("--per-tier", type=int, default=100)
    p.add_argument("--procs", type=int, default=14)
    args = p.parse_args()
    rows = []
    for w in args.weights:
        path = OUT / f"{Path(w).parent.name}_u{int(Path(w).stem.split('_u')[1])}.json"
        if not path.exists() or json.loads(path.read_text()).get("per_tier") != args.per_tier:
            path = run_one(Path(w), args.procs, args.per_tier, OUT)
        d = json.loads(path.read_text())
        s, t = d["summary"], d["summary"]["tiers"]
        rows.append((s["total"], d["snapshot"], s, t))
        print(f"{d['snapshot']}: total {s['total']}", flush=True)
    rows.sort(key=lambda r: -r[0])
    print(f"\n{'snapshot':22s} {'total':>6s}  " + "  ".join(f"t{k} exit/died" for k in (1, 2, 3, 4)) + "   stopped  worst ms")
    for total, snap, s, t in rows:
        tiers = "  ".join(f"{t[str(k)]['exit_rate']:5.0%}/{t[str(k)]['died']:<4.0%}" for k in (1, 2, 3, 4))
        print(f"{snap:22s} {total:6.2f}  {tiers}   {s['agent_stopped']:7d}  {s['worst_act_ms']:8.0f}")
    print(f"\nbest: {rows[0][1]}")


if __name__ == "__main__":
    main()
