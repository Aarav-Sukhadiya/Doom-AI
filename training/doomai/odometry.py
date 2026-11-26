"""Dead reckoning from what act() sees: a turn table for heading and a velocity regression for position.

Run `python -m doomai.odometry` to refit the constants against privileged_state() and print drift (CP4).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TURN_LEFT, TURN_RIGHT = 4, 5
VEL_FWD, VEL_RIGHT = 14, 15


SLOW_TURN_TICS = 6
SLOW_DEG = 320 / 65536 * 360
FAST_DEG = 640 / 65536 * 360
START_TICS = (0, 2)


class Odometry:
    def __init__(self, blend: tuple[float, float]):
        self.blend = blend
        self.reset()

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.turnheld = 0
        self.step = 0
        self.prev_vel = None

    def turn(self, action: np.ndarray) -> float:
        direction = int(action[TURN_LEFT]) - int(action[TURN_RIGHT])
        tics = START_TICS[self.step] if self.step < len(START_TICS) else 4
        self.step += 1
        delta = 0.0
        for _ in range(tics):
            if action[TURN_LEFT] or action[TURN_RIGHT]:
                self.turnheld += 1
                delta += direction * (SLOW_DEG if self.turnheld < SLOW_TURN_TICS else FAST_DEG)
            else:
                self.turnheld = 0
        return delta

    def update(self, action: np.ndarray, vitals: np.ndarray) -> None:
        h0 = self.heading
        self.heading = (self.heading + self.turn(action)) % 360.0
        vel = (float(vitals[VEL_FWD]), float(vitals[VEL_RIGHT]))
        if self.prev_vel is not None:
            dx0, dy0 = _ego_to_world(*self.prev_vel, h0)
            dx1, dy1 = _ego_to_world(*vel, self.heading)
            a, b = self.blend
            self.x += a * dx0 + b * dx1
            self.y += a * dy0 + b * dy1
        self.prev_vel = vel


def _ego_to_world(fwd: float, right: float, heading_deg: float) -> tuple[float, float]:
    h = math.radians(heading_deg)
    return fwd * math.cos(h) + right * math.sin(h), fwd * math.sin(h) - right * math.cos(h)


def _record(env, steps: int, seed: int):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    action = np.zeros(9, np.int8)
    for t in range(steps):
        if t % rng.integers(1, 6) == 0:
            action = np.zeros(9, np.int8)
            action[rng.choice([0, 1, 2, 3], p=[0.5, 0.15, 0.175, 0.175])] = rng.random() < 0.8
            turn = rng.choice([0, TURN_LEFT, TURN_RIGHT], p=[0.4, 0.3, 0.3])
            if turn:
                action[turn] = 1
        st = env.privileged_state()
        rows.append((action.copy(), obs["vitals"].copy(), st["x"], st["y"], st["angle"]))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    st = env.privileged_state()
    rows.append((np.zeros(9, np.int8), obs["vitals"].copy(), st["x"], st["y"], st["angle"]))
    return rows


def turn_mismatches(rows) -> int:
    odo = Odometry((0.5, 0.5))
    bad = 0
    for (action, _, _, _, a0), (_, _, _, _, a1) in zip(rows, rows[1:]):
        real = ((a1 - a0 + 180) % 360) - 180
        bad += abs(odo.turn(action) - real) > 1e-3
    return bad


def fit_blend(rows_list) -> tuple[float, float]:
    xs, ys = [], []
    for rows in rows_list:
        for (_, v0, x0, y0, a0), (_, v1, x1, y1, a1) in zip(rows[1:], rows[2:]):
            w0 = _ego_to_world(v0[VEL_FWD], v0[VEL_RIGHT], a0)
            w1 = _ego_to_world(v1[VEL_FWD], v1[VEL_RIGHT], a1)
            xs += [[w0[0], w1[0]], [w0[1], w1[1]]]
            ys += [x1 - x0, y1 - y0]
    coef, *_ = np.linalg.lstsq(np.asarray(xs), np.asarray(ys), rcond=None)
    return float(coef[0]), float(coef[1])


def drift(rows, odo: Odometry) -> dict:
    odo.reset()
    _, _, x0, y0, a0 = rows[0]
    odo.heading = a0
    path = 0.0
    out = {}
    for t, ((action, vitals, px, py, _), (_, v_next, x, y, a)) in enumerate(zip(rows, rows[1:]), 1):
        if t == 1:
            odo.prev_vel = (float(vitals[VEL_FWD]), float(vitals[VEL_RIGHT]))
        odo.update(action, v_next)
        path += math.hypot(x - px, y - py)
        if t in (100, 500):
            out[t] = {
                "heading_err_deg": abs(((odo.heading - a + 180) % 360) - 180),
                "pos_err": math.hypot(x0 + odo.x - x, y0 + odo.y - y),
                "path_len": path,
            }
    return out


def main() -> None:
    import tempfile

    from doom_env import make_env

    from doomai.maps.grid import write_grid_wad

    room = "\n".join(["#" * 26] + ["#" + "." * 24 + "#" for _ in range(24)] + ["#" * 26])
    room = room[:28] + "P" + room[29:]
    with tempfile.TemporaryDirectory() as tmp:
        env = make_env(str(write_grid_wad(room, Path(tmp) / "room.wad")), "MAP01", episode_ticks=2500)
        train = [_record(env, 600, s) for s in range(6)]
        print("turn model mismatches per episode (consecutive resets):", [turn_mismatches(r) for r in train])
        blend = fit_blend(train)
        print("velocity blend (start, end):", [round(v, 4) for v in blend])
        odo = Odometry(blend)
        for s in range(100, 103):
            print(f"held-out seed {s}:", json.dumps(drift(_record(env, 520, s), odo), default=lambda v: round(v, 1)))
        env.close()
    out = Path(__file__).with_name("odometry.json")
    out.write_text(json.dumps({"blend": blend}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
