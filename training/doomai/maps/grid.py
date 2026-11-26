"""Compile ASCII grid layouts into UDMF (zdoom namespace) WADs in the same format as the template fixtures.

Legend (one character per 64x64 cell; row 0 is the top / north edge):
  #  wall                 .  floor              P  player start (faces east)
  E  walk-over exit pad   X  exit switch (wall cell; USE its face from an adjacent floor cell)
  D  door                 R/B/Y  red/blue/yellow locked door
  r/b/y  red/blue/yellow key card
  z zombieman  s shotgun guy  c chaingunner  i imp  d demon  k cacodemon  l lost soul
  h stimpack  m medikit  a clip  e shell box  g shotgun  n chaingun  o green armor  v armor bonus
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from doom_env.wad import Lump, write_wad

CELL = 64

ROOM_STYLES = [
    ("FLOOR0_1", "STARTAN2"),
    ("FLAT1", "STONE2"),
    ("FLAT14", "BROWN1"),
    ("FLOOR4_8", "GRAY1"),
    ("FLOOR4_8", "STARTAN3"),
    ("FLOOR5_1", "BROWN96"),
    ("FLAT20", "TEKGREN2"),
    ("FLOOR1_1", "COMPBLUE"),
    ("FLAT5_5", "WOOD1"),
    ("RROCK19", "STONE3"),
    ("FLOOR7_1", "BROWNGRN"),
    ("FLAT3", "METAL1"),
    ("GRNROCK", "GSTONE1"),
]

DOOR_STYLE = {"D": ("BIGDOOR2", 0), "R": ("DOORRED", 129), "B": ("DOORBLU", 130), "Y": ("DOORYEL", 131)}

THINGS = {
    "P": 1,
    "r": 13, "b": 5, "y": 6,
    "z": 3004, "s": 9, "c": 65, "i": 3001, "d": 3002, "k": 3005, "l": 3006,
    "h": 2011, "m": 2012, "a": 2007, "e": 2049, "g": 2001, "n": 2002, "o": 2018, "v": 2015,
}

SOLID = {"#", "X"}
KNOWN = SOLID | set(DOOR_STYLE) | set(THINGS) | {".", "E"}


@dataclass
class _Builder:
    rows: int
    vertices: dict[tuple[int, int], int] = field(default_factory=dict)
    sectors: list[str] = field(default_factory=list)
    sidedefs: list[str] = field(default_factory=list)
    linedefs: list[str] = field(default_factory=list)

    def vertex(self, x: int, y: int) -> int:
        return self.vertices.setdefault((x, y), len(self.vertices))

    def side(self, sector: int, top: str = "-", middle: str = "-") -> int:
        self.sidedefs.append(
            f'sidedef {{ sector = {sector}; texturetop = "{top}"; texturemiddle = "{middle}"; texturebottom = "-"; }}'
        )
        return len(self.sidedefs) - 1


def _edge(r: int, c: int, side: str, rows: int) -> tuple[tuple[int, int], tuple[int, int]]:
    x0, x1 = c * CELL, (c + 1) * CELL
    y0 = (rows - 1 - r) * CELL
    y1 = y0 + CELL
    return {
        "W": ((x0, y0), (x0, y1)),
        "E": ((x1, y1), (x1, y0)),
        "N": ((x0, y1), (x1, y1)),
        "S": ((x1, y0), (x0, y0)),
    }[side]


NEIGHBOUR = {"W": (0, -1), "E": (0, 1), "N": (-1, 0), "S": (1, 0)}


def _parse(grid: str) -> list[str]:
    lines = grid.strip("\n").splitlines()
    width = max(len(line) for line in lines)
    return [line.ljust(width, "#") for line in lines]


def _kind(ch: str) -> str:
    if ch in SOLID:
        return "solid"
    if ch in DOOR_STYLE:
        return "door"
    if ch == "E":
        return "exit"
    return "room"


def _sectors(cells: list[str]) -> tuple[list[list[int]], list[tuple[str, str]]]:
    rows, cols = len(cells), len(cells[0])
    owner = [[-1] * cols for _ in range(rows)]
    groups: list[tuple[str, str]] = []
    for r in range(rows):
        for c in range(cols):
            kind = _kind(cells[r][c])
            if kind == "solid" or owner[r][c] >= 0:
                continue
            sid = len(groups)
            groups.append((kind, cells[r][c]))
            stack = [(r, c)]
            owner[r][c] = sid
            while stack:
                y, x = stack.pop()
                for dy, dx in NEIGHBOUR.values():
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < rows and 0 <= nx < cols) or owner[ny][nx] >= 0:
                        continue
                    nch = cells[ny][nx]
                    same = _kind(nch) == kind and (kind != "door" or nch == cells[r][c])
                    if same:
                        owner[ny][nx] = sid
                        stack.append((ny, nx))
    return owner, groups


def compile_grid(grid: str, style_seed: int | None = None, player_angle: int = 0) -> str:
    """Return the TEXTMAP text for an ASCII layout."""
    cells = _parse(grid)
    unknown = {ch for line in cells for ch in line} - KNOWN
    if unknown:
        raise ValueError(f"unknown grid characters: {sorted(unknown)}")
    rows, cols = len(cells), len(cells[0])
    owner, groups = _sectors(cells)
    styles = list(ROOM_STYLES)
    if style_seed is not None:
        random.Random(style_seed).shuffle(styles)

    b = _Builder(rows)
    for sid, (kind, ch) in enumerate(groups):
        floor, wall = styles[sid % len(styles)]
        if kind == "room":
            b.sectors.append(
                f'sector {{ texturefloor = "{floor}"; textureceiling = "CEIL1_1"; heightfloor = 0; heightceiling = 128; lightlevel = 192; }}'
            )
        elif kind == "door":
            b.sectors.append(
                'sector { texturefloor = "FLAT5_4"; textureceiling = "CEIL1_1"; heightfloor = 0; heightceiling = 0; lightlevel = 192; }'
            )
        else:
            b.sectors.append(
                'sector { texturefloor = "GATE3"; textureceiling = "CEIL1_1"; heightfloor = 0; heightceiling = 128; lightlevel = 192; }'
            )

    def wall_texture(sid: int) -> str:
        kind = groups[sid][0]
        if kind == "door":
            return "DOORTRAK"
        return styles[sid % len(styles)][1]

    runs: dict[tuple, list] = {}
    order: list[tuple] = []

    def add_run(key: tuple, start: tuple[int, int], end: tuple[int, int]) -> None:
        if key in runs:
            last_start, last_end = runs[key][-1]
            if last_end == start:
                runs[key][-1] = (last_start, end)
                return
            if end == last_start:
                runs[key][-1] = (start, last_end)
                return
        if key not in runs:
            runs[key] = []
            order.append(key)
        runs[key].append((start, end))

    for r in range(rows):
        for c in range(cols):
            sid = owner[r][c]
            if sid < 0:
                continue
            for side, (dy, dx) in NEIGHBOUR.items():
                ny, nx = r + dy, c + dx
                inside = 0 <= ny < rows and 0 <= nx < cols
                nsid = owner[ny][nx] if inside else -1
                start, end = _edge(r, c, side, rows)
                line_pos = start[0] if side in ("W", "E") else start[1]
                if nsid < 0:
                    switch = inside and cells[ny][nx] == "X"
                    add_run(("wall", sid, side, line_pos, switch), start, end)
                    continue
                if nsid == sid:
                    continue
                kind, nkind = groups[sid][0], groups[nsid][0]
                front_is_me = kind == "room" or (kind == "door" and nkind == "exit") or (kind == "door" and nkind == "door" and sid < nsid)
                if not front_is_me:
                    continue
                add_run(("two", sid, nsid, side, line_pos), start, end)

    for key in order:
        for start, end in runs[key]:
            v1, v2 = b.vertex(*start), b.vertex(*end)
            if key[0] == "wall":
                _, sid, _, _, switch = key
                if switch:
                    front = b.side(sid, middle="SW1EXIT")
                    b.linedefs.append(
                        f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {front}; blocking = true; dontpegbottom = true; special = 243; playeruse = true; }}"
                    )
                else:
                    front = b.side(sid, middle=wall_texture(sid))
                    b.linedefs.append(
                        f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {front}; blocking = true; dontpegbottom = true; }}"
                    )
                continue
            _, sid, nsid, _, _ = key
            kind, nkind = groups[sid][0], groups[nsid][0]
            if nkind == "door" and kind != "door":
                tex, lock = DOOR_STYLE[groups[nsid][1]]
                front = b.side(sid, top=tex)
                back = b.side(nsid)
                if lock:
                    b.linedefs.append(
                        f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {front}; sideback = {back}; twosided = true; special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {lock}; playeruse = true; monsteruse = false; repeatspecial = true; }}"
                    )
                else:
                    b.linedefs.append(
                        f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {front}; sideback = {back}; twosided = true; special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true; }}"
                    )
            elif nkind == "exit":
                front = b.side(sid)
                back = b.side(nsid)
                b.linedefs.append(
                    f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {front}; sideback = {back}; twosided = true; special = 243; playercross = true; }}"
                )
            else:
                front = b.side(sid)
                back = b.side(nsid)
                b.linedefs.append(
                    f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {front}; sideback = {back}; twosided = true; }}"
                )

    things = []
    for r in range(rows):
        for c in range(cols):
            ch = cells[r][c]
            if ch not in THINGS:
                continue
            x = c * CELL + CELL // 2
            y = (rows - 1 - r) * CELL + CELL // 2
            angle = player_angle if ch == "P" else 180
            things.append(
                f"thing {{ x = {x:.1f}; y = {y:.1f}; angle = {angle}; type = {THINGS[ch]}; skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }}"
            )

    verts = sorted(b.vertices.items(), key=lambda kv: kv[1])
    out = ['namespace = "zdoom";']
    out += [f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}" for (x, y), _ in verts]
    out += b.sectors + b.sidedefs + b.linedefs + things
    return "\n".join(out) + "\n"


def write_grid_wad(grid: str, path: str | Path, map_name: str = "MAP01", **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = compile_grid(grid, **kwargs)
    write_wad(path, [Lump(map_name, b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])
    return path
