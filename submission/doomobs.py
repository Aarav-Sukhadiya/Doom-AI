"""Observation preprocessing and action decoding (numpy only, no torch).

Shared by the submitted agent, the network and the training workers. Game workers import only this
module, so they do not load PyTorch.
"""

from __future__ import annotations

import numpy as np

from landmarks import CRUMB_FEATURES, MAP_CHANNELS, MAP_VIEW, NUM_SLOTS, SLOT_FEATURES, LandmarkTracker

NUM_RAYS = 64
NUM_HIT_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_FEATURES = 5
NUM_ENTITY_TYPES = 12
NUM_VITALS = 17
NUM_BUTTONS = 9
IMG_SHAPE = (4, 60, 80)

VITAL_SCALE = np.array(
    [100, 100, 7, 50, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 16, 16, 16], np.float32
)

HEADS = (("move", 3), ("strafe", 3), ("turn", 3), ("attack", 2), ("use", 2), ("weapon", 2))
HEAD_SIZES = [n for _, n in HEADS]
USE_HEAD = 4
WEAPON_HEAD = 5
USE_COOLDOWN = 8
ATTACK_HEAD = 3
ENEMY_TYPE_IDS = (1, 2, 3)


def preprocess(obs: dict) -> dict[str, np.ndarray]:
    """One observation -> compact numpy arrays (uint8 image, float32 rest)."""
    rgb = obs["rgb"][::2, ::2]
    depth = obs["depth"][::2, ::2, None]
    image = np.ascontiguousarray(np.concatenate([rgb, depth], axis=2).transpose(2, 0, 1), dtype=np.uint8)
    rays = obs["rays"]
    hit = np.asarray(rays["hit_type"], np.int64).clip(0, NUM_HIT_TYPES - 1)
    ray = np.zeros((1 + NUM_HIT_TYPES, NUM_RAYS), np.float32)
    ray[0] = np.nan_to_num(np.asarray(rays["distance"], np.float32), nan=1.0).clip(0.0, 1.0)
    ray[1 + hit, np.arange(NUM_RAYS)] = 1.0
    ents = obs["visible_entities"]
    feats = np.nan_to_num(np.asarray(ents["features"], np.float32), nan=0.0, posinf=0.0, neginf=0.0).clip(-4.0, 4.0)
    etype = np.asarray(ents["entity_type"], np.int64).clip(0, NUM_ENTITY_TYPES - 1)
    emask = np.asarray(ents["mask"], np.float32).clip(0.0, 1.0)
    vitals = np.nan_to_num(np.asarray(obs["vitals"], np.float32), nan=0.0, posinf=0.0, neginf=0.0) / VITAL_SCALE
    return {
        "image": image,
        "rays": ray,
        "ent_feat": feats,
        "ent_type": etype,
        "ent_mask": emask,
        "vitals": vitals.clip(-10.0, 10.0),
        "attack_ok": np.float32(bool(np.isin(etype[emask > 0.5], ENEMY_TYPE_IDS).any())),
    }


def observe(obs: dict, tracker: LandmarkTracker, action) -> dict[str, np.ndarray]:
    """preprocess() plus landmark slot features; action is the one that led to obs (None after reset)."""
    out = preprocess(obs)
    try:
        out["slots"] = tracker.observe(obs, action)
    except Exception:
        out["slots"] = np.zeros((NUM_SLOTS, SLOT_FEATURES), np.float32)
    try:
        out["crumbs"] = tracker.crumbs()
    except Exception:
        out["crumbs"] = np.zeros(CRUMB_FEATURES, np.float32)
    if tracker.level_map is not None:
        try:
            out["map"] = tracker.map_view()
        except Exception:
            out["map"] = np.zeros((MAP_CHANNELS, MAP_VIEW, MAP_VIEW), np.uint8)
    return out


def stack(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([b[k] for b in batch]) for k in batch[0]}


def decode_action(choice: np.ndarray) -> np.ndarray:
    """(B, 5) head choices -> (B, 9) MultiBinary buttons."""
    choice = np.asarray(choice)
    out = np.zeros((choice.shape[0], NUM_BUTTONS), np.int8)
    rows = np.arange(choice.shape[0])
    for head, (lo, hi) in enumerate(((0, 1), (2, 3), (4, 5))):
        c = choice[:, head]
        out[rows[c == 1], lo] = 1
        out[rows[c == 2], hi] = 1
    out[:, 6] = choice[:, 3] == 1
    out[:, 7] = choice[:, 4] == 1
    if choice.shape[1] > WEAPON_HEAD:
        out[:, 8] = choice[:, WEAPON_HEAD] == 1
    return out
