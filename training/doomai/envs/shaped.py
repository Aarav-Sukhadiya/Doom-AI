"""Training environment: the official DoomEnv plus our shaped reward (PRD section C5).

Observations are untouched: the policy sees exactly what act() sees at evaluation. Every reward term
is computed from privileged_state(), sector_polygons() and engine game variables, which the rules
allow for training-time shaping. Per-term rewards are returned in info["reward_terms"] for logging.
"""

from __future__ import annotations


import numpy as np
import vizdoom as vd

from doom_env import spaces as S
from doom_env.wrapper import DoomEnv

REWARD = {
    "exit": 25.0,
    "exit_health_bonus": 2.0,
    "exit_time_bonus": 5.0,
    "key": 3.0,
    "door": 1.0,
    "locked_door": 3.0,
    "new_sector": 0.2,
    "new_cell": 0.02,
    "new_cell_decay": 0.998,
    "kill": 1.0,
    "damage_dealt": 0.005,
    "damage_taken": -0.01,
    "death": -2.0,
    "death_early": -3.0,
    "ammo_used": -0.02,
    "pickup_health": 0.02,
    "pickup_armor": 0.005,
    "pickup_ammo": 0.01,
    "pickup_weapon": 1.5,
    "futile_use": -0.1,
    "dodge": 0.02,
    "cornered": -0.003,
    "idle_fire": -0.02,
    "living": -0.001,
    "subgoal": 0.0,
    "subgoal_exit_boost": 2.0,
    "new_cell_exit_boost": 3.0,
}

DOOR_OPEN_GAP = 56.0
CELL = 64.0
LOOKUP_CELL = 16.0
AMMO_SLICE = slice(S.VITALS_INDEX["ammo_bullets"], S.VITALS_INDEX["ammo_cells"] + 1)
ENEMY_TYPES = np.array([i for i, t in enumerate(S.ENTITY_TYPES) if t.startswith("enemy")])
SELECTED_AMMO = S.VITALS_INDEX["selected_weapon_ammo"]
HEALTH = S.VITALS_INDEX["health"]
PROJECTILE = S.ENTITY_TYPES.index("projectile")
THREAT_RANGE = 0.5  # entity distance in 1024-unit steps
VEL_RIGHT = S.VITALS_INDEX["vel_right"]
STRAFE_SPEED = 8.0  # map units per tic at full sideways speed
CLOSE_RAY = 128.0 / 1024.0
BOXED_SHARE = 0.75
KEY_COLOUR = ("blue", "yellow", "red", "blue", "yellow", "red")  # vitals key_* order: cards then skulls
LOCK_COLOUR = {1: "red", 2: "blue", 3: "yellow", 4: "red", 5: "blue", 6: "yellow", 129: "red", 130: "blue", 131: "yellow"}
ARMOR = S.VITALS_INDEX["armor"]
KEY_SLICE = slice(S.VITALS_INDEX["key_blue_card"], S.VITALS_INDEX["key_red_skull"] + 1)


class SectorLocator:
    """Which sector contains a point, by even-odd crossing over sector_polygons(); memoised per 16 units."""

    def __init__(self, polygons):
        ids, segs = [], []
        for sid, lines in polygons:
            if len(lines):
                ids.append(np.full(len(lines), sid))
                segs.append(np.asarray(lines, np.float64))
        self.sector = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        seg = np.concatenate(segs) if segs else np.zeros((0, 4))
        self.x1, self.y1, self.x2, self.y2 = seg.T
        self.num = int(self.sector.max()) + 1 if len(self.sector) else 0
        self.cache: dict[tuple[int, int], int] = {}

    def locate(self, x: float, y: float) -> int:
        key = (int(x // LOOKUP_CELL), int(y // LOOKUP_CELL))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        px, py = (key[0] + 0.5) * LOOKUP_CELL, (key[1] + 0.5) * LOOKUP_CELL
        spans = (self.y1 > py) != (self.y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = self.x1 + (py - self.y1) * (self.x2 - self.x1) / (self.y2 - self.y1)
        crossing = spans & (xint > px)
        counts = np.bincount(self.sector[crossing], minlength=self.num)
        inside = np.flatnonzero(counts % 2 == 1)
        sid = int(inside[0]) if len(inside) else -1
        self.cache[key] = sid
        return sid


class ShapedDoomEnv(DoomEnv):
    def __init__(self, *args, reward: dict | None = None, subgoal_grid: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_cfg = dict(REWARD, **(reward or {}))
        self._locator: SectorLocator | None = None
        self._subgoals = None
        if subgoal_grid and self.reward_cfg["subgoal"] > 0:
            from doomai.envs.subgoals import Subgoals

            self._subgoals = Subgoals(subgoal_grid)

    def _opened_locks(self) -> set[int]:
        return {int(lock) for lock in self._door[self._opened] if lock > 0}

    def _potential(self, st, vitals) -> float:
        if self._subgoals is None:
            return 0.0
        return self._subgoals.potential(st["x"], st["y"], vitals, self._opened_locks())

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._locator is None:
            self._locator = SectorLocator(self.sector_polygons())
        st = self.privileged_state()
        self._door = st["door_lock"].copy()
        self._opened = (self._door >= 0) & (st["sector_ceiling"] - st["sector_floor"] >= DOOR_OPEN_GAP)
        self._sectors_seen: set[int] = set()
        self._cells_seen: set[tuple[int, int]] = set()
        self._prev = {
            "kills": st["kills"],
            "damage_taken": st["damage_taken"],
            "damage_dealt": self._game.get_game_variable(vd.GameVariable.DAMAGECOUNT),
            "keys": obs["vitals"][KEY_SLICE].copy(),
            "ammo": obs["vitals"][AMMO_SLICE].copy(),
            "health": float(obs["vitals"][HEALTH]),
            "armor": float(obs["vitals"][ARMOR]),
            "weapons": self._weapons_owned(),
        }
        self.pickups = 0
        self.futile_uses = 0
        self.dodges = 0
        self.boxed_fighting = 0
        self._threat = self._incoming(obs)
        self._visit(st)
        self._enemy_in_view = self._sees_enemy(obs)
        self._has_ammo = obs["vitals"][SELECTED_AMMO] > 0
        self.episode_terms = {}
        self.episode_steps = 0
        self._phi = self._potential(st, obs["vitals"])
        self._stage = (self._subgoals.stage(obs["vitals"], self._opened_locks()) if self._subgoals is not None and self._subgoals.stages else -1)
        return obs, info

    def _locked_ahead_without_key(self, st, keys) -> bool:
        """Training-only check: is the closed door right in front of the player locked to a key colour
        it does not hold? Used to discourage pressing USE on it over and over."""
        a = np.radians(st["angle"])
        held = {KEY_COLOUR[i] for i in range(len(KEY_COLOUR)) if keys[i] > 0.5}
        for d in (24.0, 48.0, 72.0, 96.0):
            sec = self._locator.locate(st["x"] + d * np.cos(a), st["y"] + d * np.sin(a))
            if sec < 0 or sec >= len(self._door):
                continue
            lock = int(self._door[sec])
            if lock > 0:
                closed = st["sector_ceiling"][sec] - st["sector_floor"][sec] < DOOR_OPEN_GAP
                return closed and LOCK_COLOUR.get(lock) is not None and LOCK_COLOUR[lock] not in held
        return False

    def _weapons_owned(self) -> int:
        return sum(self._game.get_game_variable(getattr(vd.GameVariable, f"WEAPON{i}")) > 0 for i in range(2, 8))

    @staticmethod
    def _incoming(obs) -> bool:
        """A projectile in view, within 512 units and heading toward the player."""
        ents = obs["visible_entities"]
        m = (np.asarray(ents["mask"]) > 0) & (np.asarray(ents["entity_type"]) == PROJECTILE)
        if not m.any():
            return False
        f = np.asarray(ents["features"], np.float64)[m]
        d, s, c, vf, vr = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
        closing = (c * vf + s * -vr) < 0
        return bool(np.any((d < THREAT_RANGE) & closing))

    @staticmethod
    def _boxed(obs) -> bool:
        """Most of the 64 rays hit something within 128 units: a corridor or corner with little room."""
        dist = np.nan_to_num(np.asarray(obs["rays"]["distance"], np.float64), nan=1.0)
        return bool(np.mean(dist < CLOSE_RAY) >= BOXED_SHARE)

    @staticmethod
    def _sees_enemy(obs) -> bool:
        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]) > 0
        return bool(np.isin(np.asarray(ents["entity_type"])[mask], ENEMY_TYPES).any())

    def _visit(self, st) -> tuple[int, float]:
        sector = self._locator.locate(st["x"], st["y"])
        new_sector = int(sector >= 0 and sector not in self._sectors_seen)
        self._sectors_seen.add(sector)
        cell = (int(st["x"] // CELL), int(st["y"] // CELL))
        cell_reward = 0.0
        if cell not in self._cells_seen:
            cell_reward = self.reward_cfg["new_cell"] * self.reward_cfg["new_cell_decay"] ** len(self._cells_seen)
            self._cells_seen.add(cell)
        return new_sector, cell_reward

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        self.episode_steps += 1
        cfg = self.reward_cfg
        st = self.privileged_state()
        terms: dict[str, float] = {"living": cfg["living"]}

        kills = st["kills"] - self._prev["kills"]
        terms["kill"] = cfg["kill"] * kills
        taken = st["damage_taken"] - self._prev["damage_taken"]
        terms["damage_taken"] = cfg["damage_taken"] * taken
        dealt_now = self._game.get_game_variable(vd.GameVariable.DAMAGECOUNT)
        terms["damage_dealt"] = cfg["damage_dealt"] * max(0.0, dealt_now - self._prev["damage_dealt"])

        keys = obs["vitals"][KEY_SLICE]
        terms["key"] = cfg["key"] * float(np.sum((keys > 0.5) & (self._prev["keys"] < 0.5)))
        ammo = obs["vitals"][AMMO_SLICE]
        terms["ammo_used"] = cfg["ammo_used"] * float(np.sum(np.maximum(0.0, self._prev["ammo"] - ammo)))
        alive = not st["dead"]
        health, armor = float(obs["vitals"][HEALTH]), float(obs["vitals"][ARMOR])
        weapons = self._weapons_owned()
        gains = {
            "pickup_health": max(0.0, health - self._prev["health"]) if alive else 0.0,
            "pickup_armor": max(0.0, armor - self._prev["armor"]) if alive else 0.0,
            "pickup_ammo": float(np.sum(np.maximum(0.0, ammo - self._prev["ammo"]))) if alive else 0.0,
            "pickup_weapon": float(max(0, weapons - self._prev["weapons"])),
        }
        for k, g in gains.items():
            terms[k] = cfg[k] * g
        self.pickups += sum(g > 0 for g in gains.values())
        futile = bool(np.asarray(action)[7]) and not st["finished"] and self._locked_ahead_without_key(st, self._prev["keys"])
        terms["futile_use"] = cfg["futile_use"] * float(futile)
        self.futile_uses += int(futile)
        fired_blind = bool(np.asarray(action)[6]) and not self._enemy_in_view and self._has_ammo
        terms["idle_fire"] = cfg["idle_fire"] * float(fired_blind)
        sideways = min(1.0, abs(float(obs["vitals"][VEL_RIGHT])) / STRAFE_SPEED)
        dodge = sideways if (self._threat and not st["finished"]) else 0.0
        terms["dodge"] = cfg["dodge"] * dodge
        self.dodges += int(dodge > 0.5)
        boxed = self._enemy_in_view and not st["finished"] and self._boxed(obs)
        terms["cornered"] = cfg["cornered"] * float(boxed)
        self.boxed_fighting += int(boxed)
        self._threat = self._incoming(obs)
        self._enemy_in_view = self._sees_enemy(obs)
        self._has_ammo = obs["vitals"][SELECTED_AMMO] > 0

        gap = st["sector_ceiling"] - st["sector_floor"]
        newly_open = (self._door >= 0) & ~self._opened & (gap >= DOOR_OPEN_GAP)
        self._opened |= newly_open
        terms["door"] = cfg["door"] * float(np.sum(newly_open & (self._door == 0)))
        terms["locked_door"] = cfg["locked_door"] * float(np.sum(newly_open & (self._door > 0)))

        if not st["finished"]:
            new_sector, cell_reward = self._visit(st)
            terms["new_sector"] = cfg["new_sector"] * new_sector
            if self._subgoals is not None and self._subgoals.stages and self._stage == len(self._subgoals.stages) - 1:
                # Final stretch: only the exit is left and no sensor shows it, so new ground is worth
                # more than walking rooms it has already seen.
                cell_reward *= cfg["new_cell_exit_boost"]
            terms["new_cell"] = cell_reward
            if self._subgoals is not None:
                phi = self._potential(st, obs["vitals"])
                scale = cfg["subgoal"]
                stage = self._subgoals.stage(obs["vitals"], self._opened_locks()) if self._subgoals.stages else -1
                last = len(self._subgoals.stages) - 1
                if stage == last and self._stage == last:
                    # Final stretch (every key and lock done): pull harder toward the exit, which no
                    # sensor shows, so it learns to link the exit's look with walking to it.
                    scale *= cfg["subgoal_exit_boost"]
                terms["subgoal"] = scale * (phi - self._phi)
                self._phi = phi
                self._stage = stage

        left = max(0.0, 1.0 - st["tic"] / max(1, st["episode_ticks"]))
        if info["exit_reached"]:
            terms["exit"] = cfg["exit"] + cfg["exit_health_bonus"] * max(0.0, st["health"]) / 100.0 + cfg["exit_time_bonus"] * left
        if st["dead"]:
            terms["death"] = cfg["death"] + cfg["death_early"] * left

        self._prev.update(kills=st["kills"], damage_taken=st["damage_taken"], damage_dealt=dealt_now, keys=keys.copy(), ammo=ammo.copy(),
                          health=health, armor=armor, weapons=weapons)
        for k, v in terms.items():
            self.episode_terms[k] = self.episode_terms.get(k, 0.0) + v
        info = dict(info, reward_terms=terms)
        if terminated or truncated:
            info["episode_terms"] = dict(self.episode_terms)
            info["episode_stats"] = {
                "exit": float(info["exit_reached"]),
                "dead": float(st["dead"]),
                "kills": float(st["kills"]),
                "sectors": float(len(self._sectors_seen - {-1})),
                "cells": float(len(self._cells_seen)),
                "keys": float(np.sum(keys > 0.5)),
                "doors_opened": float(np.sum(self._opened)),
                "steps": float(self.episode_steps),
                "pickups": float(self.pickups),
                "futile_uses": float(self.futile_uses),
                "dodges": float(self.dodges),
                "boxed_fighting": float(self.boxed_fighting),
            }
        return obs, float(sum(terms.values())), terminated, truncated, info
