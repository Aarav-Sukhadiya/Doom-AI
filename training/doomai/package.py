"""Copy a trained run into submission/weights/ (model.pt as a plain state_dict, plus config.json).

    python -m doomai.package --run v1 [--checkpoint model_u500.pt] [--greedy]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="model.pt")
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    run = REPO / "runs" / args.run
    state = torch.load(run / args.checkpoint, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    config = json.loads((run / "config.json").read_text())
    config["greedy"] = args.greedy
    config["source"] = f"runs/{args.run}/{args.checkpoint}"
    out = REPO / "submission" / "weights"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.contiguous() for k, v in state.items()}, out / "model.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    size = (out / "model.pt").stat().st_size / 1e6
    print(f"packaged {config['source']} -> submission/weights/model.pt ({size:.1f} MB), config {config}")


if __name__ == "__main__":
    main()
