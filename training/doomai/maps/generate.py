"""Procedural training levels for the four evaluation tiers, built on the grid compiler.

Rooms are rectangles attached to each other as a tree through 1-cell-wide corridors, so every map has
branches and dead ends. Tier 1: combat arenas. Tier 2: 3-8 rooms with doors. Tier 3: the exit sits
behind a locked door whose key is down another branch. Tier 4: two nested locks, ambushes, little ammo.

The breadth-first solver here is training-only scaffolding (map validation and tick budgets); the
agent never sees it.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field

HITSCAN = ["z", "z", "s"]
MELEE_PROJ = ["i", "i", "d"]
HARD = ["c", "k", "l"]
HEALTH = ["h", "h", "m"]
AMMO = ["a", "a", "e"]
ARMOR = ["v", "v", "o"]
KEYS = {"R": "r", "B": "b", "Y": "y"}
TIERS = (1, 2, 3, 4)


@dataclass
class Room:
    x: int
    y: int
    w: int
    h: int
    parent: int | None = None
    door: tuple[int, int] | None = None
    depth: int = 0

    def cells(self):
        for yy in range(self.y, self.y + self.h):
            for xx in range(self.x, self.x + self.w):
                yield yy, xx


@dataclass
class Level:
    grid: str
    tier: int
    seed: int
    player_angle: int
    episode_ticks: int
    exit_kind: str
    locks: list[str] = field(default_factory=list)
    shortest_steps: int = 0


class _Canvas:
    def __init__(self, size: int):
        self.size = size
        self.g = [["#"] * size for _ in range(size)]

    def free(self, x0: int, y0: int, x1: int, y1: int, ignore: set | None = None) -> bool:
        if x0 < 1 or y0 < 1 or x1 >= self.size - 1 or y1 >= self.size - 1:
            return False
        for yy in range(y0 - 1, y1 + 2):
            for xx in range(x0 - 1, x1 + 2):
                if self.g[yy][xx] != "#" and (ignore is None or (yy, xx) not in ignore):
                    return False
        return True

    def carve(self, room: Room) -> None:
        for yy, xx in room.cells():
            self.g[yy][xx] = "."


DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}


def _attach(rng: random.Random, canvas: _Canvas, rooms: list[Room], parent_idx: int, size_range: tuple[int, int]) -> bool:
    a = rooms[parent_idx]
    for _ in range(40):
        side = rng.choice("NSWE")
        length = rng.randint(1, 4)
        w, h = rng.randint(*size_range), rng.randint(*size_range)
        dy, dx = DIRS[side]
        if side in "NS":
            cx = rng.randint(a.x, a.x + a.w - 1)
            cy = a.y - 1 if side == "N" else a.y + a.h
            corridor = [(cy + dy * k, cx) for k in range(length)]
            end_y = corridor[-1][0] + dy
            nx = rng.randint(cx - w + 1, cx)
            ny = end_y - h + 1 if side == "N" else end_y
        else:
            cy = rng.randint(a.y, a.y + a.h - 1)
            cx = a.x - 1 if side == "W" else a.x + a.w
            corridor = [(cy, cx + dx * k) for k in range(length)]
            end_x = corridor[-1][1] + dx
            ny = rng.randint(cy - h + 1, cy)
            nx = end_x - w + 1 if side == "W" else end_x
        new = Room(nx, ny, w, h, parent=parent_idx, depth=a.depth + 1)
        if not canvas.free(nx, ny, nx + w - 1, ny + h - 1):
            continue
        ok = all(0 < yy < canvas.size - 1 and 0 < xx < canvas.size - 1 and canvas.g[yy][xx] == "#" for yy, xx in corridor)
        side_cells = set()
        for yy, xx in corridor:
            for py, px in ((yy + dx, xx + dy), (yy - dx, xx - dy)):
                side_cells.add((py, px))
        ok = ok and all(canvas.g[yy][xx] == "#" for yy, xx in side_cells if (yy, xx) not in set(a.cells()))
        if not ok:
            continue
        canvas.carve(new)
        for yy, xx in corridor:
            canvas.g[yy][xx] = "."
        new.door = corridor[len(corridor) // 2]
        rooms.append(new)
        return True
    return False


def _path_to_root(rooms: list[Room], idx: int) -> list[int]:
    out = [idx]
    while rooms[out[-1]].parent is not None:
        out.append(rooms[out[-1]].parent)
    return out


def _subtree(rooms: list[Room], idx: int) -> set[int]:
    out = {idx}
    changed = True
    while changed:
        changed = False
        for i, r in enumerate(rooms):
            if r.parent in out and i not in out:
                out.add(i)
                changed = True
    return out


def _empty_cells(canvas: _Canvas, room: Room, rng: random.Random) -> list[tuple[int, int]]:
    cells = [(yy, xx) for yy, xx in room.cells() if canvas.g[yy][xx] == "."]
    rng.shuffle(cells)
    return cells


def _place(canvas: _Canvas, room: Room, ch: str, rng: random.Random, avoid: set | None = None) -> tuple[int, int] | None:
    for yy, xx in _empty_cells(canvas, room, rng):
        if avoid and (yy, xx) in avoid:
            continue
        canvas.g[yy][xx] = ch
        return yy, xx
    return None


def _distances(canvas: _Canvas, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dy, dx in DIRS.values():
            n = (r + dy, c + dx)
            if n not in dist and canvas.g[n[0]][n[1]] != "#":
                dist[n] = dist[(r, c)] + 1
                queue.append(n)
    return dist


def _far_choice(options: list[tuple[int, int]], dist: dict, rng: random.Random) -> tuple[int, int]:
    ranked = sorted(options, key=lambda cell: -dist.get(cell, 0))
    return rng.choice(ranked[: max(1, len(ranked) // 3)])


def _exit_pad(canvas: _Canvas, room: Room, rng: random.Random, dist: dict) -> bool:
    options = [(yy, xx) for yy, xx in room.cells() if canvas.g[yy][xx] == "."]
    if not options:
        return False
    yy, xx = _far_choice(options, dist, rng)
    canvas.g[yy][xx] = "E"
    return True


def _exit_switch(canvas: _Canvas, room: Room, rng: random.Random, dist: dict) -> bool:
    options = []
    for yy, xx in room.cells():
        for dy, dx in DIRS.values():
            wy, wx = yy + dy, xx + dx
            if not (1 <= wy <= canvas.size - 2 and 1 <= wx <= canvas.size - 2) or canvas.g[wy][wx] != "#":
                continue
            floor_neighbours = sum(canvas.g[wy + a][wx + b] not in "#X" for a, b in DIRS.values() if 0 <= wy + a < canvas.size and 0 <= wx + b < canvas.size)
            if floor_neighbours == 1:
                options.append((wy, wx))
    if not options:
        return False
    adjacent = {}
    for wy, wx in options:
        adjacent[(wy, wx)] = max(dist.get((wy + a, wx + b), 0) for a, b in DIRS.values())
    wy, wx = _far_choice(options, adjacent, rng)
    canvas.g[wy][wx] = "X"
    return True


def _add_pillars(canvas: _Canvas, room: Room, rng: random.Random, count: int) -> None:
    if room.w < 5 or room.h < 5:
        return
    for _ in range(count):
        yy = rng.randint(room.y + 1, room.y + room.h - 2)
        xx = rng.randint(room.x + 1, room.x + room.w - 2)
        if canvas.g[yy][xx] == ".":
            canvas.g[yy][xx] = "#"


def _crop(canvas: _Canvas) -> list[str]:
    rows = [r for r in range(canvas.size) if any(c != "#" for c in canvas.g[r])]
    cols = [c for c in range(canvas.size) if any(canvas.g[r][c] != "#" for r in range(canvas.size))]
    r0, r1, c0, c1 = rows[0] - 1, rows[-1] + 1, cols[0] - 1, cols[-1] + 1
    return ["".join(canvas.g[r][c0 : c1 + 1]) for r in range(r0, r1 + 1)]


def solve_path(lines: list[str]) -> list[tuple[int, int]] | None:
    """Shortest cell path from P to an exit cell (E, or the X switch itself), collecting keys on the way."""
    start = next((r, c) for r, line in enumerate(lines) for c, ch in enumerate(line) if ch == "P")
    key_bit = {"r": 1, "b": 2, "y": 4}
    lock_bit = {"R": 1, "B": 2, "Y": 4}
    parent = {(start, 0): None}
    queue = deque([(start, 0)])
    while queue:
        (r, c), keys = queue.popleft()
        for dy, dx in DIRS.values():
            nr, nc = r + dy, c + dx
            ch = lines[nr][nc]
            if ch in "XE":
                path = [(nr, nc)]
                node = ((r, c), keys)
                while node is not None:
                    path.append(node[0])
                    node = parent[node]
                return path[::-1]
            if ch == "#" or (ch in lock_bit and not keys & lock_bit[ch]):
                continue
            state = ((nr, nc), keys | key_bit.get(ch, 0))
            if state not in parent:
                parent[state] = ((r, c), keys)
                queue.append(state)
    return None


def solve(lines: list[str]) -> int | None:
    path = solve_path(lines)
    return None if path is None else len(path) - 1


def _populate(canvas, rooms, rng, n_enemies, pool, n_health, n_ammo, n_armor, skip_room=0):
    candidates = [i for i in range(len(rooms)) if i != skip_room] or [0]
    for _ in range(n_enemies):
        _place(canvas, rooms[rng.choice(candidates)], rng.choice(pool), rng)
    for pool_items, count in ((HEALTH, n_health), (AMMO, n_ammo), (ARMOR, n_armor)):
        for _ in range(count):
            _place(canvas, rooms[rng.randrange(len(rooms))], rng.choice(pool_items), rng)


def _build(tier: int, rng: random.Random, difficulty: float):
    canvas = _Canvas(72)
    if tier == 1:
        size = (6 + int(5 * difficulty), 9 + int(6 * difficulty))
        n_rooms = 1 + (rng.random() < 0.5)
    elif tier == 2:
        size = (3, 8)
        n_rooms = rng.randint(3, 4 + int(4 * difficulty))
    elif tier == 3:
        size = (3, 7)
        n_rooms = rng.randint(4, 5 + int(3 * difficulty))
    else:
        size = (3, 7)
        n_rooms = rng.randint(6, 7 + int(3 * difficulty))
    w, h = rng.randint(*size), rng.randint(*size)
    rooms = [Room(36 - w // 2, 36 - h // 2, w, h)]
    canvas.carve(rooms[0])
    tries = 0
    while len(rooms) < n_rooms and tries < 200:
        tries += 1
        parent = rng.randrange(len(rooms))
        _attach(rng, canvas, rooms, parent, size)

    player = _place(canvas, rooms[0], "P", rng)
    leaves = [i for i in range(len(rooms)) if not any(r.parent == i for r in rooms)]
    far_leaves = sorted(leaves, key=lambda i: -rooms[i].depth)
    exit_room = far_leaves[0] if far_leaves and far_leaves[0] != 0 else len(rooms) - 1
    locks: list[str] = []

    if tier == 2:
        for i, r in enumerate(rooms[1:], 1):
            if rng.random() < 0.6:
                canvas.g[r.door[0]][r.door[1]] = "D"
    if tier in (3, 4):
        colours = rng.sample(list(KEYS), 2 if tier == 4 else 1)
        path = _path_to_root(rooms, exit_room)
        lockable = [i for i in path if i != 0]
        lock_rooms = sorted(rng.sample(lockable, min(len(colours), len(lockable))), key=lambda i: rooms[i].depth)
        blocked: set[int] = set()
        for colour, lock_room in zip(colours, lock_rooms):
            behind = _subtree(rooms, lock_room)
            open_rooms = [i for i in range(len(rooms)) if i not in behind and i not in blocked]
            if tier == 4 and locks:
                open_rooms = [i for i in _subtree(rooms, lock_rooms[0]) if i not in behind] or open_rooms
            if len(open_rooms) > 1:
                open_rooms = [i for i in open_rooms if i != 0]
            elif not open_rooms or open_rooms == [0]:
                return None
            key_room = max(open_rooms, key=lambda i: (rooms[i].depth, rng.random()))
            dy, dx = rooms[lock_room].door
            canvas.g[dy][dx] = colour
            _place(canvas, rooms[key_room], KEYS[colour], rng)
            locks.append(colour)
            blocked |= behind
        for i, r in enumerate(rooms[1:], 1):
            if canvas.g[r.door[0]][r.door[1]] == "." and rng.random() < 0.4:
                canvas.g[r.door[0]][r.door[1]] = "D"

    dist = _distances(canvas, player)
    exit_kind = "switch" if rng.random() < 0.3 else "pad"
    if exit_kind == "switch" and not _exit_switch(canvas, rooms[exit_room], rng, dist):
        exit_kind = "pad"
    if exit_kind == "pad":
        _exit_pad(canvas, rooms[exit_room], rng, dist)

    if tier == 1:
        pool = HITSCAN + MELEE_PROJ + (HARD if difficulty > 0.5 else [])
        for r in rooms:
            _add_pillars(canvas, r, rng, rng.randint(0, 4))
        _populate(canvas, rooms, rng, rng.randint(2, 3 + int(5 * difficulty)), pool, rng.randint(1, 3), rng.randint(1, 3), rng.randint(0, 1), skip_room=-1)
    elif tier == 2:
        _populate(canvas, rooms, rng, rng.randint(0, 1 + int(3 * difficulty)), HITSCAN + MELEE_PROJ, rng.randint(0, 2), rng.randint(0, 2), 0)
    elif tier == 3:
        _populate(canvas, rooms, rng, rng.randint(1, 2 + int(3 * difficulty)), HITSCAN + MELEE_PROJ, rng.randint(1, 2), rng.randint(1, 2), rng.randint(0, 1))
    else:
        pool = HITSCAN + MELEE_PROJ + HARD
        _populate(canvas, rooms, rng, rng.randint(4, 6 + int(6 * difficulty)), pool, rng.randint(0, 2), rng.randint(0, 1), rng.randint(0, 1))
    return _crop(canvas), exit_kind, locks


EXPECTED_LOCKS = {1: 0, 2: 0, 3: 1, 4: 2}
TICK_FLOOR = {1: 2100, 2: 3150, 3: 4200, 4: 4200}


WEAPON_CHANCE = (("g", 0.5), ("n", 0.2))


def _add_weapons(lines: list[str], wrng: random.Random) -> list[str]:
    """Sometimes drop a shotgun and/or chaingun on open floor. Uses its own random stream, so the
    layout, enemies and items of every level stay exactly as they were without weapons."""
    rows = [list(line) for line in lines]
    for ch, chance in WEAPON_CHANCE:
        if wrng.random() >= chance:
            continue
        free = [
            (r, c) for r, row in enumerate(rows) for c, cell in enumerate(row)
            if cell == "." and all(
                0 <= r + dr < len(rows) and 0 <= c + dc < len(rows[r + dr]) and rows[r + dr][c + dc] not in "DRBYEP"
                for dr, dc in DIRS.values()
            )
        ]
        if free:
            r, c = wrng.choice(free)
            rows[r][c] = ch
    return ["".join(row) for row in rows]


def _all_reachable(lines: list[str]) -> bool:
    start = next((r, c) for r, line in enumerate(lines) for c, ch in enumerate(line) if ch == "P")
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dy, dx in DIRS.values():
            n = (r + dy, c + dx)
            if n not in seen and lines[n[0]][n[1]] not in "#X":
                seen.add(n)
                queue.append(n)
    open_cells = sum(ch not in "#X" for line in lines for ch in line)
    return len(seen) == open_cells


def generate(tier: int, seed: int, difficulty: float = 1.0) -> Level:
    """A solvable level for `tier` (1-4). `difficulty` in [0, 1] scales size and enemy count."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    rng = random.Random(f"{tier}-{seed}-{difficulty:.3f}")
    for _ in range(50):
        try:
            built = _build(tier, rng, difficulty)
        except IndexError:
            continue
        if built is None:
            continue
        lines, exit_kind, locks = built
        if sum(ch == "P" for line in lines for ch in line) != 1 or len(locks) != EXPECTED_LOCKS[tier]:
            continue
        if not _all_reachable(lines):
            continue
        moves = solve(lines)
        if moves is None:
            continue
        steps = moves * 2
        ticks = int(min(10500, max(TICK_FLOOR[tier], steps * 4 * 5)))
        lines = _add_weapons(lines, random.Random(f"weapons-{tier}-{seed}-{difficulty:.3f}"))
        return Level("\n".join(lines), tier, seed, rng.randrange(0, 360, 45), ticks, exit_kind, locks, steps)
    raise RuntimeError(f"could not build a solvable tier {tier} level for seed {seed}")


def split_of(seed: int) -> str:
    """Stable 80/10/10 train/validation/test split by seed."""
    bucket = int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) % 10
    return "test" if bucket == 9 else "val" if bucket == 8 else "train"


def seeds_for(split: str, count: int, start: int = 0) -> list[int]:
    out, seed = [], start
    while len(out) < count:
        if split_of(seed) == split:
            out.append(seed)
        seed += 1
    return out
