"""Training-only subgoal shaping: a potential that rises as the agent gets closer to its next goal.

The goals come from the generator's solver path, in order: each key, then the locked door it opens,
and finally the exit. The potential of a state is

    scale * (stage * D_max - distance_to_current_goal)

where distance is the walking distance in grid cells with still-closed locks treated as walls.
Reaching a goal moves to the next stage, which never lowers the potential. ShapedDoomEnv adds the
change in potential to the reward each step, so the agent is paid for moving toward the key, then
back to its door, then on to the exit, which the analysis showed it stopped doing (it learned to
avoid locked doors even after holding the key). None of this is visible to the policy.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from doomai.maps.generate import solve_path
from doomai.maps.grid import CELL

KEY_VITAL = {"b": 8, "y": 9, "r": 10}
LOCK_NUMBER = {"R": 129, "B": 130, "Y": 131}
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Subgoals:
    def __init__(self, grid: str):
        self.lines = grid.strip("\n").splitlines()
        width = max(len(line) for line in self.lines)
        self.lines = [line.ljust(width, "#") for line in self.lines]
        self.rows = len(self.lines)
        path = solve_path(self.lines) or []
        seq, seen = [], set()
        for r, c in path:
            ch = self.lines[r][c]
            if ch in "rbyRBY" and ch not in seen:
                seen.add(ch)
                seq.append((ch, (r, c)))
        if path:
            seq.append(("exit", path[-1]))
        self.stages = seq
        self.dist = []
        for k, (ch, target) in enumerate(seq):
            closed = {cell for (c2, cell) in seq[k:] if c2 in LOCK_NUMBER} - {target}
            self.dist.append(self._bfs(target, closed))
        self.d_max = 1 + max((max(d.values()) for d in self.dist if d), default=0)

    def _bfs(self, target, closed) -> dict:
        dist = {target: 0}
        queue = deque([target])
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRS:
                n = (r + dr, c + dc)
                if not (0 <= n[0] < self.rows and 0 <= n[1] < len(self.lines[0])) or n in dist:
                    continue
                ch = self.lines[n[0]][n[1]]
                if ch in "#X" or n in closed:
                    continue
                dist[n] = dist[(r, c)] + 1
                queue.append(n)
        return dist

    def stage(self, vitals: np.ndarray, opened_locks: set[int]) -> int:
        for k, (ch, _) in enumerate(self.stages):
            if ch in KEY_VITAL and vitals[KEY_VITAL[ch]] < 0.5:
                return k
            if ch in LOCK_NUMBER and LOCK_NUMBER[ch] not in opened_locks:
                return k
            if ch == "exit":
                return k
        return len(self.stages) - 1

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return self.rows - 1 - int(y // CELL), int(x // CELL)

    def potential(self, x: float, y: float, vitals: np.ndarray, opened_locks: set[int]) -> float:
        if not self.stages:
            return 0.0
        k = self.stage(vitals, opened_locks)
        d = self.dist[k].get(self.cell(x, y), self.d_max)
        return float(k * self.d_max - d)
