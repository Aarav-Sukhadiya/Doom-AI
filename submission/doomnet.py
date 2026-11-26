"""The Doom-AI policy network, shared by the submitted agent and the training code.

Everything here depends only on numpy and torch and on the observation dict act() receives, so the
trainer and the evaluated agent run exactly the same preprocessing and network.

Inputs (all ego-centric, as the official wrapper provides them):
  image     rgb + depth, downsampled 2x to 4 x 60 x 80
  rays      64 rays: distance + one-hot hit type, read by a circular 1-D conv (no seam behind the player)
  entities  up to 32 visible things: features + type embedding, pooled over the mask (any number of enemies)
  vitals    health, armour, ammo, keys, velocity, plus the agent's previous action
  masks     USE is blocked for 8 steps after a press (door rules); ATTACK is blocked when no enemy is
            in view, so ammo is not wasted on empty rooms (both are action masks from observation fields)
  slots     landmark memory (landmarks.py): where keys, locked doors, doors and pickups were seen,
            relative to the agent now; read by learned attention and added to the GRU input through a
            zero-initialised layer, so turning it on leaves an existing policy unchanged at first
A GRU carries short-term memory between decisions. The action head is factorised into five small
choices (move, strafe, turn, attack, use); weapon switching is left to the engine's auto-switch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from doomobs import (  # noqa: F401
    ATTACK_HEAD,
    HEAD_SIZES,
    HEADS,
    IMG_SHAPE,
    MAX_ENTITIES,
    NUM_BUTTONS,
    NUM_ENTITY_FEATURES,
    NUM_ENTITY_TYPES,
    NUM_HIT_TYPES,
    NUM_RAYS,
    NUM_VITALS,
    USE_COOLDOWN,
    USE_HEAD,
    VITAL_SCALE,
    decode_action,
    observe,
    preprocess,
    stack,
)
from landmarks import CRUMB_FEATURES, MAP_CHANNELS, MAP_VIEW, NUM_SLOTS, SLOT_FEATURES, LandmarkTracker  # noqa: F401

def to_tensors(arrays: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in arrays.items()}


class DoomNet(nn.Module):
    def __init__(self, hidden: int = 256, slots: bool = False, crumbs: bool = False, mapping: bool = False):
        super().__init__()
        self.hidden = hidden
        self.use_slots = slots
        self.use_crumbs = crumbs
        self.use_map = mapping
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, *IMG_SHAPE)).shape[1]
        self.img_fc = nn.Linear(cnn_out, 256)
        self.ray1 = nn.Conv1d(1 + NUM_HIT_TYPES, 32, 5, padding=2, padding_mode="circular")
        self.ray2 = nn.Conv1d(32, 32, 5, stride=2, padding=2, padding_mode="circular")
        self.ray_fc = nn.Linear(32 * NUM_RAYS // 2, 128)
        self.ent_embed = nn.Embedding(NUM_ENTITY_TYPES, 16)
        self.ent_mlp = nn.Sequential(nn.Linear(NUM_ENTITY_FEATURES + 16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.vit_fc = nn.Sequential(nn.Linear(NUM_VITALS + NUM_BUTTONS, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(256 + 128 + 128 + 64, hidden), nn.ReLU())
        if slots:
            self.slot_type = nn.Embedding(NUM_SLOTS, 8)
            self.slot_mlp = nn.Sequential(nn.Linear(SLOT_FEATURES + 8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.slot_q = nn.Linear(hidden, 64)
            self.slot_k = nn.Linear(64, 64)
            self.slot_v = nn.Linear(64, 64)
            self.slot_out = nn.Linear(64, hidden)
            nn.init.zeros_(self.slot_out.weight)
            nn.init.zeros_(self.slot_out.bias)
        if crumbs:
            self.crumb_mlp = nn.Sequential(nn.Linear(CRUMB_FEATURES, 64), nn.ReLU())
            self.crumb_out = nn.Linear(64, hidden)
            nn.init.zeros_(self.crumb_out.weight)
            nn.init.zeros_(self.crumb_out.bias)
        if mapping:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(MAP_CHANNELS, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * (MAP_VIEW // 8) ** 2, 128), nn.ReLU(),
            )
            self.map_out = nn.Linear(128, hidden)
            nn.init.zeros_(self.map_out.weight)
            nn.init.zeros_(self.map_out.bias)
        self.gru = nn.GRUCell(hidden, hidden)
        self.policy = nn.Linear(hidden, sum(HEAD_SIZES))
        self.value = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.policy.weight, 0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, 1.0)
        nn.init.zeros_(self.value.bias)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    def encode(self, x: dict[str, torch.Tensor], prev_action: torch.Tensor) -> torch.Tensor:
        img = self.img_fc(self.cnn(x["image"].float() / 255.0)).relu()
        r = F.relu(self.ray1(x["rays"]))
        r = F.relu(self.ray2(r))
        r = self.ray_fc(r.flatten(1)).relu()
        e = self.ent_mlp(torch.cat([x["ent_feat"], self.ent_embed(x["ent_type"])], dim=-1))
        m = x["ent_mask"].unsqueeze(-1)
        any_entity = m.sum(1) > 0
        e_mean = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        e_max = e.masked_fill(m < 0.5, -1e4).max(1).values
        e_max = torch.where(any_entity, e_max, torch.zeros_like(e_max))
        v = self.vit_fc(torch.cat([x["vitals"], prev_action], dim=-1))
        z = self.trunk(torch.cat([img, r, torch.cat([e_mean, e_max], -1), v], dim=-1))
        if self.use_crumbs:
            z = z + self.crumb_out(self.crumb_mlp(x["crumbs"]))
        if self.use_map:
            z = z + self.map_out(self.map_cnn(x["map"].float() / 255.0))
        if self.use_slots:
            z = z + self.read_slots(x["slots"], z)
        return z

    def read_slots(self, slots: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        types = self.slot_type.weight.unsqueeze(0).expand(slots.shape[0], -1, -1)
        s = self.slot_mlp(torch.cat([slots, types], dim=-1))
        score = (self.slot_q(z).unsqueeze(1) * self.slot_k(s)).sum(-1) / 8.0
        score = score.masked_fill(slots[..., 0] < 0.5, -1e9)
        att = torch.softmax(score, dim=-1).unsqueeze(-1)
        return self.slot_out((att * self.slot_v(s)).sum(1))

    def heads(self, h: torch.Tensor, use_allowed: torch.Tensor, attack_ok: torch.Tensor | None = None):
        logits = mask_use(self.policy(h), use_allowed)
        if attack_ok is not None:
            logits = mask_attack(logits, attack_ok)
        return logits, self.value(h).squeeze(-1)

    def forward(self, x, prev_action, h, use_allowed):
        """One step: returns per-head logits (B, 13), value (B,), next hidden (B, H)."""
        h = self.gru(self.encode(x, prev_action), h)
        logits, value = self.heads(h, use_allowed, x.get("attack_ok"))
        return logits, value, h

    def forward_sequence(self, x, prev_action, h, use_allowed, dones):
        """L steps for N sequences at once. x tensors are (L, N, ...); dones (L, N) reset h after step t."""
        L, N = dones.shape
        flat = {k: v.reshape(L * N, *v.shape[2:]) for k, v in x.items()}
        feats = self.encode(flat, prev_action.reshape(L * N, -1)).reshape(L, N, -1)
        hs = []
        for t in range(L):
            h = self.gru(feats[t], h)
            hs.append(h)
            h = h * (1.0 - dones[t]).unsqueeze(-1)
        hs = torch.stack(hs).reshape(L * N, -1)
        attack_ok = flat.get("attack_ok")
        logits, value = self.heads(hs, use_allowed.reshape(L * N), attack_ok)
        return logits, value


def mask_attack(logits: torch.Tensor, attack_ok: torch.Tensor) -> torch.Tensor:
    """Fire only when the observation shows an enemy in view (action masking from observation fields)."""
    start = sum(HEAD_SIZES[:ATTACK_HEAD])
    blocked = (attack_ok < 0.5).unsqueeze(-1)
    att = logits[:, start : start + 2]
    att = torch.where(blocked, torch.tensor([0.0, -1e9], device=logits.device).expand_as(att), att)
    return torch.cat([logits[:, :start], att, logits[:, start + 2 :]], dim=1)


def mask_use(logits: torch.Tensor, use_allowed: torch.Tensor) -> torch.Tensor:
    start = sum(HEAD_SIZES[:USE_HEAD])
    blocked = (use_allowed < 0.5).unsqueeze(-1)
    use = logits[:, start : start + 2]
    use = torch.where(blocked, torch.tensor([0.0, -1e9], device=logits.device).expand_as(use), use)
    return torch.cat([logits[:, :start], use], dim=1)


def split_logits(logits: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.split(logits, HEAD_SIZES, dim=-1))


def sample(logits: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    choices = []
    for part in split_logits(logits):
        if not greedy:
            u = torch.rand_like(part).clamp_(1e-9, 1.0 - 1e-9)
            part = part - torch.log(-torch.log(u))
        choices.append(part.argmax(-1))
    return torch.stack(choices, dim=-1)


def log_prob_entropy(logits: torch.Tensor, choice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lp = torch.zeros(logits.shape[0], device=logits.device)
    ent = torch.zeros(logits.shape[0], device=logits.device)
    for i, part in enumerate(split_logits(logits)):
        logp = torch.log_softmax(part, dim=-1)
        lp = lp + logp.gather(-1, choice[:, i : i + 1]).squeeze(-1)
        ent = ent - (logp.exp() * logp).sum(-1)
    return lp, ent
