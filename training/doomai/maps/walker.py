"""Training-only scripted walker: follows the solver's cell path using privileged_state().

It proves that generated levels are completable in the real engine (doors open, keys unlock, exits
fire). It reads ground-truth position, so it is scaffolding for map validation and never part of the
submitted agent.
"""

from __future__ import annotations

import math

import numpy as np

from doomai.maps.grid import CELL

FWD, TURN_L, TURN_R, USE = 0, 4, 5, 7


def cell_center(r: int, c: int, rows: int) -> tuple[float, float]:
    return c * CELL + CELL / 2, (rows - 1 - r) * CELL + CELL / 2


def walk(env, lines: list[str], path: list[tuple[int, int]], max_steps: int = 3000) -> dict:
    rows = len(lines)
    targets = path[1:]
    idx = 0
    last_use = -100
    last_pos = None
    stuck = 0
    info = {}
    decisions = 0
    for step in range(max_steps * 4):
        if decisions >= max_steps:
            break
        st = env.privileged_state()
        x, y, angle = st["x"], st["y"], st["angle"]
        r, c = targets[idx]
        tx, ty = cell_center(r, c, rows)
        ch = lines[r][c]
        if ch not in "XRBYD" and math.hypot(tx - x, ty - y) < 24 and idx < len(targets) - 1:
            idx += 1
            continue
        bearing = math.degrees(math.atan2(ty - y, tx - x))
        err = ((bearing - angle + 180) % 360) - 180
        action = np.zeros(9, np.int8)
        if err > 8:
            action[TURN_L] = 1
        elif err < -8:
            action[TURN_R] = 1
        if abs(err) < 30:
            action[FWD] = 1
        moved = last_pos is not None and math.hypot(x - last_pos[0], y - last_pos[1]) > 1.0
        stuck = 0 if moved else stuck + 1
        last_pos = (x, y)
        near_special = ch in "XRBYD" and math.hypot(tx - x, ty - y) < 100
        if (near_special or stuck > 3) and decisions - last_use > 20:
            action[USE] = 1
            last_use = decisions
        if ch in "RBYD" and math.hypot(tx - x, ty - y) < 20 and idx < len(targets) - 1:
            idx += 1
        _, _, term, trunc, info = env.step(action)
        decisions += 1
        if term or trunc:
            return {"exit_reached": bool(info.get("exit_reached")), "steps": decisions, "dead": env.privileged_state()["dead"]}
    return {"exit_reached": False, "steps": max_steps, "dead": False}
