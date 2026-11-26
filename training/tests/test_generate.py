import re

import pytest

from doom_env import make_env
from doom_env.mapinfo import load_mapinfo
from doomai.maps.generate import TIERS, generate, seeds_for, solve, solve_path, split_of
from doomai.maps.grid import write_grid_wad
from doomai.maps.walker import walk

LOCK = {"R": 129, "B": 130, "Y": 131}


@pytest.mark.parametrize("tier", TIERS)
def test_levels_are_solvable_and_shaped_by_tier(tier):
    for seed in range(60):
        level = generate(tier, seed, difficulty=(seed % 5) / 4)
        lines = level.grid.splitlines()
        assert solve(lines) is not None
        assert sum(ch == "P" for ch in level.grid) == 1
        assert len(level.locks) == {1: 0, 2: 0, 3: 1, 4: 2}[tier]
        assert 2100 <= level.episode_ticks <= 10500


def test_generation_is_deterministic():
    assert generate(3, 42).grid == generate(3, 42).grid
    assert generate(3, 42).grid != generate(3, 43).grid


def test_splits_are_stable_and_disjoint():
    train, val, test = seeds_for("train", 200), seeds_for("val", 30), seeds_for("test", 30)
    assert not set(train) & set(val) and not set(val) & set(test) and not set(train) & set(test)
    assert all(split_of(s) == "val" for s in val)


@pytest.mark.parametrize("tier", TIERS)
def test_level_loads_with_matching_locks(tier, tmp_path):
    level = generate(tier, 5)
    path = write_grid_wad(level.grid, tmp_path / "m.wad", player_angle=level.player_angle)
    info = load_mapinfo(path, "MAP01")
    assert sorted(v for v in info.door_sectors.values() if v) == sorted(LOCK[c] for c in level.locks)
    env = make_env(str(path), "MAP01", episode_ticks=200)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("tier", TIERS)
def test_scripted_walker_reaches_exit_in_engine(tier, tmp_path):
    level = generate(tier, 3, difficulty=0.75)
    grid = re.sub("[zscidkl]", ".", level.grid)
    lines = grid.splitlines()
    env = make_env(str(write_grid_wad(grid, tmp_path / "m.wad")), "MAP01", episode_ticks=12000)
    env.reset(seed=0)
    result = walk(env, lines, solve_path(lines))
    env.close()
    assert result["exit_reached"]


def test_bug_hunt_regressions():
    generate(3, 1661, 0.75)
    assert len(generate(4, 255, 0.0).locks) == 2
    for seed in range(300):
        assert len(generate(4, seed, 0.0).locks) == 2


def test_unknown_grid_characters_are_rejected():
    from doomai.maps.grid import compile_grid

    with pytest.raises(ValueError):
        compile_grid("#####\n#P Q#\n#####")


def test_adjacent_doors_close_their_sectors():
    from doomai.maps.grid import compile_grid

    text = compile_grid("########\n#P.RB.E#\n########")
    lines = [l for l in text.splitlines() if "twosided" in l]
    assert len(lines) == 4 and sum("special" not in l for l in lines) == 1
