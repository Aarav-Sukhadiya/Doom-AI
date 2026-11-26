import numpy as np

from doom_env import make_env
from doomai.envs.shaped import ShapedDoomEnv
from doomai.maps.generate import solve_path
from doomai.maps.grid import write_grid_wad
from doomai.maps.walker import walk

NOOP = np.zeros(9, np.int8)
ATTACK = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], np.int8)


def run_walker(tmp_path, grid, ticks=2000):
    env = ShapedDoomEnv(str(write_grid_wad(grid, tmp_path / "m.wad")), "MAP01", episode_ticks=ticks)
    env.reset(seed=0)
    events = []
    step = env.step

    def logged(action):
        out = step(action)
        events.append(out[4])
        return out

    env.step = logged
    result = walk(env, grid.splitlines(), solve_path(grid.splitlines()))
    env.close()
    return result, events


def fired(events, term):
    return [e["reward_terms"][term] for e in events if abs(e["reward_terms"].get(term, 0.0)) > 1e-9]


def test_key_locked_door_exit_fire_once(tmp_path):
    result, events = run_walker(tmp_path, "############\n#P.r..R...E#\n############")
    assert result["exit_reached"]
    assert fired(events, "key") == [3.0]
    assert fired(events, "locked_door") == [3.0]
    assert fired(events, "door") == []
    assert len(fired(events, "exit")) == 1 and fired(events, "exit")[0] > 10.0
    assert events[-1]["episode_stats"]["steps"] == len(events)


def test_plain_door_and_sectors(tmp_path):
    result, events = run_walker(tmp_path, "##########\n#P..D...E#\n##########")
    assert result["exit_reached"]
    assert fired(events, "door") == [1.0]
    assert sum(fired(events, "new_sector")) == 0.4


def test_kill_and_damage_dealt(tmp_path):
    env = ShapedDoomEnv(str(write_grid_wad("#########\n#P.....z#\n#########", tmp_path / "m.wad")), "MAP01", episode_ticks=2000)
    env.reset(seed=0)
    kills = dealt = 0.0
    for i in range(150):
        _, _, term, trunc, info = env.step(ATTACK if i % 2 == 0 else NOOP)
        kills += info["reward_terms"]["kill"]
        dealt += info["reward_terms"]["damage_dealt"]
        if term or trunc or kills:
            break
    env.close()
    assert kills == 1.0
    assert dealt > 0.0


def test_death_penalty_scales_with_time_left(tmp_path):
    env = ShapedDoomEnv(str(write_grid_wad("###########\n#P.......c#\n#ccc......#\n###########", tmp_path / "m.wad")), "MAP01", episode_ticks=8000)
    env.reset(seed=0)
    for _ in range(2000):
        _, _, term, trunc, info = env.step(NOOP)
        if term or trunc:
            break
    env.close()
    assert term and info["episode_stats"]["dead"] == 1.0
    assert -5.0 <= info["reward_terms"]["death"] < -2.0


def test_observations_match_official_env(tmp_path):
    path = str(write_grid_wad("##########\n#P..D..z.E#\n##########", tmp_path / "m.wad"))
    a = make_env(path, "MAP01", episode_ticks=400)
    b = ShapedDoomEnv(path, "MAP01", episode_ticks=400)
    oa, _ = a.reset(seed=1)
    ob, _ = b.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(60):
        for key in ("vitals", "depth", "rgb"):
            np.testing.assert_array_equal(oa[key], ob[key])
        np.testing.assert_array_equal(oa["rays"]["distance"], ob["rays"]["distance"])
        action = (rng.random(9) < 0.3).astype(np.int8)
        oa, _, ta, tra, _ = a.step(action)
        ob, _, tb, trb, _ = b.step(action)
        if ta or tra:
            break
    a.close()
    b.close()


def _walk_with_subgoal(tmp_path, grid, scale=0.05):
    import re

    from doomai.envs.subgoals import Subgoals

    clean = re.sub("[zscidkl]", ".", grid)
    env = ShapedDoomEnv(str(write_grid_wad(clean, tmp_path / "m.wad")), "MAP01", episode_ticks=8000,
                        reward={"subgoal": scale}, subgoal_grid=clean)
    env.reset(seed=0)
    events = []
    step = env.step

    def logged(action):
        out = step(action)
        events.append(out[4])
        return out

    env.step = logged
    result = walk(env, clean.splitlines(), solve_path(clean.splitlines()))
    env.close()
    return result, events, Subgoals(clean)


def test_subgoal_shaping_rewards_progress_through_key_lock_exit(tmp_path):
    from doomai.maps.generate import generate

    for tier in (3, 4):
        level = generate(tier, 9, difficulty=0.5)
        result, events, sub = _walk_with_subgoal(tmp_path, level.grid)
        assert result["exit_reached"]
        shaping = [e["reward_terms"].get("subgoal", 0.0) for e in events]
        assert sum(shaping) > 0.05 * (len(sub.stages) - 1) * sub.d_max * 0.5
        assert sub.stages[-1][0] == "exit" and len(sub.stages) == 1 + 2 * (tier - 2)


def test_subgoal_shaping_penalises_walking_away(tmp_path):
    grid = "############\n#.....P..r.R...E#\n############"
    env = ShapedDoomEnv(str(write_grid_wad(grid, tmp_path / "m.wad")), "MAP01", episode_ticks=400,
                        reward={"subgoal": 0.05}, subgoal_grid=grid)
    env.reset(seed=0)
    back = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], np.int8)
    total = sum(env.step(back)[4]["reward_terms"]["subgoal"] for _ in range(20))
    env.close()
    assert total < 0


def test_subgoal_shaping_off_by_default(tmp_path):
    result, events = run_walker(tmp_path, "##########\n#P..D...E#\n##########")
    assert all("subgoal" not in e["reward_terms"] for e in events)
