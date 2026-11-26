import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from doom_env import make_env
from doom_env.mapinfo import load_mapinfo, parse_udmf
from doom_env.wad import list_maps, map_lumps, read_wad
from doomai.maps.grid import compile_grid, write_grid_wad

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from fixtures import fixture_wad  # noqa: E402

LAYOUTS = [
    "#####\n#P.z#\n#####",
    "####\n#P.#\n####",
    "#####\n#P..#\n#####",
    "#######\n#P...E#\n#######",
    "##########\n#P..D...E#\n##########",
    "#######\n#P..B.E#\n#######",
    "############\n#P.r..R...E#\n############",
    "#########\n#P..h..z#\n#########",
    "###########\n#z.P..D..z#\n###########",
    "##########\n#P......i#\n##########",
    "##########\n#P......c#\n##########",
    "#########\n#P.....z#\n#..D....#\n#########",
]

FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)


def canonical(text: str):
    m = parse_udmf(text)
    verts = [(float(v["x"]), float(v["y"])) for v in m["vertex"]]
    sectors = [tuple(sorted(s.items())) for s in m["sector"]]
    sides = m["sidedef"]

    def side(i):
        if i is None:
            return None
        s = dict(sides[int(i)])
        sec = sectors[int(s.pop("sector"))]
        return sec, tuple(sorted(s.items()))

    lines = Counter()
    for line in m["linedef"]:
        props = {k: v for k, v in line.items() if k not in ("v1", "v2", "sidefront", "sideback")}
        lines[(verts[int(line["v1"])], verts[int(line["v2"])], side(line["sidefront"]), side(line.get("sideback")), tuple(sorted(props.items())))] += 1
    things = Counter(tuple(sorted(t.items())) for t in m.get("thing", []))
    return Counter(sectors), lines, things


def fixture_text(grid: str) -> str:
    lumps = read_wad(fixture_wad(grid))
    return map_lumps(lumps, list_maps(lumps)[0])["TEXTMAP"].decode()


@pytest.mark.parametrize("grid", LAYOUTS)
def test_matches_template_fixture(grid):
    ours = canonical(compile_grid(grid))
    theirs = canonical(fixture_text(grid))
    assert ours[0] == theirs[0], "sectors differ"
    assert ours[2] == theirs[2], "things differ"
    assert ours[1] == theirs[1], "linedefs differ"


def test_walk_to_exit_pad(tmp_path):
    path = write_grid_wad("#######\n#P...E#\n#######", tmp_path / "m.wad")
    env = make_env(str(path), "MAP01", episode_ticks=400)
    env.reset()
    for _ in range(100):
        _, _, term, trunc, info = env.step(FWD)
        if term or trunc:
            break
    assert term and info["exit_reached"]
    env.close()


def test_exit_switch(tmp_path):
    path = write_grid_wad("######\n#P...X\n######", tmp_path / "m.wad")
    env = make_env(str(path), "MAP01", episode_ticks=400)
    env.reset()
    for i in range(100):
        _, _, term, trunc, info = env.step(USE if i % 2 else FWD)
        if term or trunc:
            break
    assert term and info["exit_reached"]
    env.close()


def test_multi_room_doors_detected(tmp_path):
    grid = "\n".join([
        "###########",
        "#P..#.....#",
        "#...D..y..#",
        "#...#.....#",
        "##R####Y###",
        "#....#...E#",
        "#.r..B....#",
        "#....#.b..#",
        "###########",
    ])
    path = write_grid_wad(grid, tmp_path / "m.wad", style_seed=3)
    info = load_mapinfo(path, "MAP01")
    assert sorted(info.door_sectors.values()) == [0, 129, 130, 131]
    env = make_env(str(path), "MAP01", episode_ticks=200)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    env.close()
