"""Landmark memory inputs: dead-reckoned pose plus typed slots of where important things were seen.

The game never tells the agent its position, so Odometry rebuilds it from the agent's own turn presses
(Doom's exact turn rule) and the observed player-frame velocity. LandmarkTracker then writes a slot
whenever the observation shows a key, a locked door, a door or a pickup, storing the sighting in the
start frame. Each decision it reports every slot relative to where the agent is now, so the network
can learn to head back to a locked door once it holds the key. Writes are rule-triggered
preprocessing of observation fields; how the slots are used is learned (attention in DoomNet).
"""

from __future__ import annotations

import math

import numpy as np

TURN_LEFT, TURN_RIGHT = 4, 5
VEL_FWD, VEL_RIGHT = 14, 15
SLOW_TURN_TICS = 6
SLOW_DEG = 320 / 65536 * 360
FAST_DEG = 640 / 65536 * 360
FIRST_TURN_TIC = 8
BLEND = (1.6020745162454768, 2.821983769288885)

RAY_RANGE = 1024.0
NUM_RAYS = 64
RAY_DOOR, RAY_LOCKED = 2, 3
ENT_HEALTH, ENT_ARMOR, ENT_AMMO, ENT_WEAPON, ENT_KEY = 5, 6, 7, 8, 9

SLOT_TYPES = ("start", "key", "locked_door", "door", "health", "ammo", "weapon", "armor")
NUM_SLOTS = len(SLOT_TYPES)
SLOT_FEATURES = 7
_ENTITY_SLOT = {ENT_KEY: 1, ENT_HEALTH: 4, ENT_AMMO: 5, ENT_WEAPON: 6, ENT_ARMOR: 7}

CRUMB_CELL = 128.0
CRUMB_DIRECTIONS = 8
CRUMB_RADII = (1, 2, 4)
CRUMB_FEATURES = 2 + CRUMB_DIRECTIONS * len(CRUMB_RADII)

MAP_CELL = 64.0
MAP_WORLD = 1024
MAP_VIEW = 32
MAP_CHANNELS = 4  # open floor seen, wall, door (128) or locked door (255), walked on
RAY_WALL, RAY_HIT_DOOR, RAY_HIT_LOCKED = 1, 2, 3
_STEPS = np.arange(16.0, RAY_RANGE, 32.0)
_view = (np.arange(MAP_VIEW) - MAP_VIEW / 2 + 0.5) * MAP_CELL
_VIEW_FWD = np.repeat(_view[::-1], MAP_VIEW).reshape(MAP_VIEW, MAP_VIEW)
_VIEW_LEFT = np.tile(_view[::-1], MAP_VIEW).reshape(MAP_VIEW, MAP_VIEW)


class LevelMap:
    """A map of the level built as the agent goes, from its own rays and dead-reckoned position.

    Every decision each of the 64 rays marks the cells it passed through as open floor and the cell
    it stopped at as wall, door or locked door; the cell under the agent is marked as walked on. A
    door that later opens is cleared because rays now pass through it. view() returns a 32x32 crop
    (64 units per cell, about 1000 units each way) centred on the agent and turned to its heading,
    forward up, so the network can see which way is unexplored. Pure preprocessing of observation
    fields; what to do with it is learned."""

    def __init__(self):
        self.grid = np.zeros((MAP_CHANNELS, MAP_WORLD, MAP_WORLD), np.uint8)

    def reset(self) -> None:
        self.grid[:] = 0

    @staticmethod
    def _index(x, y):
        i = np.floor(np.asarray(x) / MAP_CELL).astype(np.int64) + MAP_WORLD // 2
        j = np.floor(np.asarray(y) / MAP_CELL).astype(np.int64) + MAP_WORLD // 2
        return np.clip(i, 0, MAP_WORLD - 1), np.clip(j, 0, MAP_WORLD - 1)

    def update(self, x: float, y: float, heading_deg: float, dist: np.ndarray, hit: np.ndarray) -> None:
        ang = math.radians(heading_deg) + np.arange(NUM_RAYS) * 2.0 * math.pi / NUM_RAYS
        c, s = np.cos(ang), np.sin(ang)
        d = np.clip(dist, 0.0, 1.0) * RAY_RANGE
        reach = _STEPS[None, :] < (d[:, None] - 24.0)
        px = x + c[:, None] * _STEPS[None, :]
        py = y + s[:, None] * _STEPS[None, :]
        fi, fj = self._index(px[reach], py[reach])
        self.grid[0, fi, fj] = 255
        self.grid[1, fi, fj] = 0
        self.grid[2, fi, fj] = 0
        solid = (hit >= RAY_WALL) & (hit <= RAY_HIT_LOCKED) & (dist < 0.999)
        if solid.any():
            hx, hy = x + c[solid] * (d[solid] + 8.0), y + s[solid] * (d[solid] + 8.0)
            wi, wj = self._index(hx, hy)
            kind = hit[solid]
            self.grid[1, wi[kind == RAY_WALL], wj[kind == RAY_WALL]] = 255
            self.grid[2, wi[kind == RAY_HIT_DOOR], wj[kind == RAY_HIT_DOOR]] = 128
            self.grid[2, wi[kind == RAY_HIT_LOCKED], wj[kind == RAY_HIT_LOCKED]] = 255
        ai, aj = self._index(x, y)
        self.grid[3, ai, aj] = 255

    def view(self, x: float, y: float, heading_deg: float) -> np.ndarray:
        h = math.radians(heading_deg)
        ch, sh = math.cos(h), math.sin(h)
        wx = x + _VIEW_FWD * ch - _VIEW_LEFT * sh
        wy = y + _VIEW_FWD * sh + _VIEW_LEFT * ch
        i, j = self._index(wx, wy)
        return self.grid[:, i, j]


def ego_to_world(fwd: float, right: float, heading_deg: float) -> tuple[float, float]:
    h = math.radians(heading_deg)
    return fwd * math.cos(h) + right * math.sin(h), fwd * math.sin(h) - right * math.cos(h)


class Odometry:
    def __init__(self, blend: tuple[float, float] = BLEND):
        self.blend = blend
        self.reset()

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.turnheld = 0
        self.tic = 0
        self.prev_vel = None

    def turn(self, action) -> float:
        """Doom counts held turn tics from the first tic, but the player only rotates from tic 8 after reset."""
        direction = int(action[TURN_LEFT]) - int(action[TURN_RIGHT])
        delta = 0.0
        for _ in range(4):
            self.tic += 1
            if action[TURN_LEFT] or action[TURN_RIGHT]:
                self.turnheld += 1
                if self.tic >= FIRST_TURN_TIC:
                    delta += direction * (SLOW_DEG if self.turnheld < SLOW_TURN_TICS else FAST_DEG)
            else:
                self.turnheld = 0
        return delta

    def update(self, action, vitals) -> None:
        h0 = self.heading
        self.heading = (self.heading + self.turn(action)) % 360.0
        vel = (float(vitals[VEL_FWD]), float(vitals[VEL_RIGHT]))
        if self.prev_vel is not None:
            dx0, dy0 = ego_to_world(*self.prev_vel, h0)
            dx1, dy1 = ego_to_world(*vel, self.heading)
            a, b = self.blend
            self.x += a * dx0 + b * dx1
            self.y += a * dy0 + b * dy1
        self.prev_vel = vel


class LandmarkTracker:
    """Call observe(obs, None) after reset and observe(obs, action) after every step."""

    def __init__(self, entity_range: float = 1.0, mapping: bool = False):
        """entity_range caps entity distances (in RAY_RANGE units); the entity list reaches 4 (4096 units).
        mapping=True also builds a LevelMap (map_view())."""
        self.entity_range = float(entity_range)
        self.level_map = LevelMap() if mapping else None
        self.odo = Odometry()
        self.reset()

    def reset(self) -> None:
        self.odo.reset()
        self.pos = np.zeros((NUM_SLOTS, 2), np.float64)
        self.valid = np.zeros(NUM_SLOTS, bool)
        self.age = np.zeros(NUM_SLOTS, np.float64)
        self.seen_now = np.zeros(NUM_SLOTS, bool)
        self.valid[0] = True
        self.visits: dict[tuple[int, int], int] = {}
        if getattr(self, "level_map", None) is not None:
            self.level_map.reset()

    def _write(self, slot: int, dist_norm: float, rel_sin: float, rel_cos: float, cap: float = 1.0) -> None:
        """Store a sighting at distance dist_norm and relative direction (sin, cos) in the start frame."""
        d = float(np.clip(dist_norm, 0.0, cap)) * RAY_RANGE
        norm = math.hypot(rel_sin, rel_cos) or 1.0
        rs, rc = rel_sin / norm, rel_cos / norm
        h = math.radians(self.odo.heading)
        hs, hc = math.sin(h), math.cos(h)
        dir_cos, dir_sin = hc * rc - hs * rs, hs * rc + hc * rs
        self.pos[slot] = (self.odo.x + d * dir_cos, self.odo.y + d * dir_sin)
        self.valid[slot] = True
        self.age[slot] = 0.0
        self.seen_now[slot] = True

    def observe(self, obs: dict, action) -> np.ndarray:
        vitals = np.nan_to_num(np.asarray(obs["vitals"], np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if action is None:
            self.odo.prev_vel = (float(vitals[VEL_FWD]), float(vitals[VEL_RIGHT]))
        else:
            self.odo.update(np.asarray(action), vitals)
        self.age += 1.0
        self.seen_now[:] = False
        here = self._cell(self.odo.x, self.odo.y)
        self.visits[here] = self.visits.get(here, 0) + 1

        rays = obs["rays"]
        dist = np.nan_to_num(np.asarray(rays["distance"], np.float64), nan=1.0, posinf=1.0, neginf=1.0)
        hit = np.asarray(rays["hit_type"], np.int64)
        if self.level_map is not None:
            self.level_map.update(self.odo.x, self.odo.y, self.odo.heading, dist, hit)
        for slot, kind in ((2, RAY_LOCKED), (3, RAY_DOOR)):
            idx = np.flatnonzero(hit == kind)
            if len(idx):
                k = int(idx[np.argmin(dist[idx])])
                ray = k * 2.0 * math.pi / NUM_RAYS
                self._write(slot, dist[k], math.sin(ray), math.cos(ray))

        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]) > 0
        etype = np.asarray(ents["entity_type"], np.int64)
        feats = np.nan_to_num(np.asarray(ents["features"], np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        best: dict[int, tuple[float, float]] = {}
        for i in np.flatnonzero(mask):
            slot = _ENTITY_SLOT.get(int(etype[i]))
            if slot is None:
                continue
            d = feats[i, 0]
            if slot not in best or d < best[slot][0]:
                best[slot] = (d, feats[i, 1], feats[i, 2])
        for slot, (d, rel_sin, rel_cos) in best.items():
            self._write(slot, d, rel_sin, rel_cos, self.entity_range)
        return self.features()

    @staticmethod
    def _cell(x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x / CRUMB_CELL)), int(math.floor(y / CRUMB_CELL))

    def map_view(self) -> np.ndarray:
        """(4, 32, 32) uint8 ego-centric level map; zeros when mapping is off."""
        if self.level_map is None:
            return np.zeros((MAP_CHANNELS, MAP_VIEW, MAP_VIEW), np.uint8)
        return self.level_map.view(self.odo.x, self.odo.y, self.odo.heading)

    def crumbs(self) -> np.ndarray:
        """Breadcrumbs: how often the agent has already been here and at 1, 2 and 4 cells (128 units each)
        in 8 directions around its current heading, from its own dead-reckoned path. Lets the network
        tell ground it has covered from new ground on large maps."""
        out = np.zeros(CRUMB_FEATURES, np.float32)
        x, y = self.odo.x, self.odo.y
        out[0] = min(1.0, math.log1p(self.visits.get(self._cell(x, y), 0)) / 4.0)
        out[1] = min(4.0, len(self.visits) / 100.0)
        h = math.radians(self.odo.heading)
        k = 2
        for d in range(CRUMB_DIRECTIONS):
            a = h + d * 2.0 * math.pi / CRUMB_DIRECTIONS
            ca, sa = math.cos(a), math.sin(a)
            for r in CRUMB_RADII:
                n = self.visits.get(self._cell(x + r * CRUMB_CELL * ca, y + r * CRUMB_CELL * sa), 0)
                out[k] = min(1.0, math.log1p(n) / 4.0)
                k += 1
        return out

    def features(self) -> np.ndarray:
        """(NUM_SLOTS, 7): valid, forward and left offset, distance, sin and cos of bearing (ego), age."""
        out = np.zeros((NUM_SLOTS, SLOT_FEATURES), np.float32)
        h = math.radians(self.odo.heading)
        c, s = math.cos(h), math.sin(h)
        dx = self.pos[:, 0] - self.odo.x
        dy = self.pos[:, 1] - self.odo.y
        fwd = dx * c + dy * s
        left = -dx * s + dy * c
        dist = np.hypot(fwd, left)
        safe = np.where(dist > 1e-6, dist, 1.0)
        out[:, 0] = self.valid
        out[:, 1] = np.clip(fwd / RAY_RANGE, -4, 4)
        out[:, 2] = np.clip(left / RAY_RANGE, -4, 4)
        out[:, 3] = np.clip(dist / RAY_RANGE, 0, 4)
        out[:, 4] = np.where(dist > 1e-6, left / safe, 0.0)
        out[:, 5] = np.where(dist > 1e-6, fwd / safe, 1.0)
        out[:, 6] = np.clip(self.age / 256.0, 0, 4)
        out[~self.valid] = 0.0
        return out
