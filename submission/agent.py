"""Evaluation entry point: the scorer instantiates DoomAgent from this file (docs/CONTRACT.md section 6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from doomnet import USE_COOLDOWN, DoomNet, decode_action, observe, sample, to_tensors
from landmarks import LandmarkTracker

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
        self.fire_mask = bool(config.get("fire_mask", False))
        self.net = DoomNet(hidden=int(config.get("hidden", 256)), slots=bool(config.get("slots", False)),
                           crumbs=bool(config.get("crumbs", False)), mapping=bool(config.get("map", False)),
                           weapon=bool(config.get("weapon", False)))
        self.tracker = LandmarkTracker(float(config.get("entity_range", 1.0)), mapping=bool(config.get("map", False)))
        weights = WEIGHTS_DIR / "model.pt"
        if weights.exists():
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.net.load_state_dict(state, strict=True)
        self.net.to(self.device).eval()
        torch.manual_seed(0)
        self.reset()

    def reset(self) -> None:
        self.h = self.net.initial_state(1, self.device)
        self.prev_action = np.zeros(9, np.float32)
        self.use_cooldown = 0
        self.tracker.reset()
        self.first = True

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        try:
            arrays = observe(observation, self.tracker, None if self.first else self.prev_action)
            if not self.fire_mask:
                arrays.pop("attack_ok", None)
            x = to_tensors({k: v[None] for k, v in arrays.items()}, self.device)
            prev = torch.from_numpy(self.prev_action[None]).to(self.device)
            allowed = torch.tensor([1.0 if self.use_cooldown == 0 else 0.0], device=self.device)
            logits, _, self.h = self.net(x, prev, self.h, allowed)
            choice = sample(logits, greedy=self.greedy).cpu().numpy()
            action = decode_action(choice)[0]
        except Exception:
            action = NOOP.copy()
        self.first = False
        self.use_cooldown = USE_COOLDOWN if action[7] else max(0, self.use_cooldown - 1)
        self.prev_action = action.astype(np.float32)
        return action
