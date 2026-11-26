"""Evaluation entry point: the scorer instantiates DoomAgent from this file (docs/CONTRACT.md section 6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from doomnet import USE_COOLDOWN, DoomNet, decode_action, preprocess, sample, to_tensors

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
NOOP = np.zeros(9, np.int8)


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        config_path = WEIGHTS_DIR / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.greedy = bool(config.get("greedy", False))
        self.net = DoomNet(hidden=int(config.get("hidden", 256)))
        weights = WEIGHTS_DIR / "model.pt"
        if weights.exists():
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.net.load_state_dict(state, strict=True)
        self.net.to(self.device).eval()
        self.generator = torch.Generator(device="cpu").manual_seed(0)
        torch.manual_seed(0)
        self.reset()

    def reset(self) -> None:
        self.h = self.net.initial_state(1, self.device)
        self.prev_action = np.zeros(9, np.float32)
        self.use_cooldown = 0

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        try:
            x = to_tensors({k: v[None] for k, v in preprocess(observation).items()}, self.device)
            prev = torch.from_numpy(self.prev_action[None]).to(self.device)
            allowed = torch.tensor([1.0 if self.use_cooldown == 0 else 0.0], device=self.device)
            logits, _, self.h = self.net(x, prev, self.h, allowed)
            choice = sample(logits, greedy=self.greedy).cpu().numpy()
            action = decode_action(choice)[0]
        except Exception:
            action = NOOP.copy()
        self.use_cooldown = USE_COOLDOWN if action[7] else max(0, self.use_cooldown - 1)
        self.prev_action = action.astype(np.float32)
        return action
