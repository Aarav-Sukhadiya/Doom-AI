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
    "exit": 10.0,
    "exit_health_bonus": 2.0,
    "exit_time_bonus": 2.0,
    "key": 3.0,
    "door": 1.0,
    "locked_door": 3.0,
    "new_sector": 0.2,
    "new_cell": 0.02,
    "new_cell_decay": 0.995,
    "kill": 1.0,
    "damage_dealt": 0.01,
    "damage_taken": -0.01,
    "death": -2.0,
    "death_early": -3.0,
    "ammo_used": -0.005,
    "living": -0.001,
    "subgoal": 0.0,
}

DOOR_OPEN_GAP = 56.0
CELL = 64.0
LOOKUP_CELL = 16.0
AMMO_SLICE = slice(S.VITALS_INDEX["ammo_bullets"], S.VITALS_INDEX["ammo_cells"] + 1)
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
        self._opened = np.zeros(len(self._door), bool)
        self._sectors_seen: set[int] = set()
        self._cells_seen: set[tuple[int, int]] = set()
        self._prev = {
            "kills": st["kills"],
            "damage_taken": st["damage_taken"],
            "damage_dealt": self._game.get_game_variable(vd.GameVariable.DAMAGECOUNT),
            "keys": obs["vitals"][KEY_SLICE].copy(),
            "ammo": obs["vitals"][AMMO_SLICE].copy(),
        }
        self._visit(st)
        self.episode_terms = {}
        self.episode_steps = 0
        self._phi = self._potential(st, obs["vitals"])
        return obs, info

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

        gap = st["sector_ceiling"] - st["sector_floor"]
        newly_open = (self._door >= 0) & ~self._opened & (gap >= DOOR_OPEN_GAP)
        self._opened |= newly_open
        terms["door"] = cfg["door"] * float(np.sum(newly_open & (self._door == 0)))
        terms["locked_door"] = cfg["locked_door"] * float(np.sum(newly_open & (self._door > 0)))

        if not st["finished"]:
            new_sector, cell_reward = self._visit(st)
            terms["new_sector"] = cfg["new_sector"] * new_sector
            terms["new_cell"] = cell_reward
            if self._subgoals is not None:
                phi = self._potential(st, obs["vitals"])
                terms["subgoal"] = cfg["subgoal"] * (phi - self._phi)
                self._phi = phi

        left = max(0.0, 1.0 - st["tic"] / max(1, st["episode_ticks"]))
        if info["exit_reached"]:
            terms["exit"] = cfg["exit"] + cfg["exit_health_bonus"] * max(0.0, st["health"]) / 100.0 + cfg["exit_time_bonus"] * left
        if st["dead"]:
            terms["death"] = cfg["death"] + cfg["death_early"] * left

        self._prev.update(kills=st["kills"], damage_taken=st["damage_taken"], damage_dealt=dealt_now, keys=keys.copy(), ammo=ammo.copy())
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
            }
        return obs, float(sum(terms.values())), terminated, truncated, info
