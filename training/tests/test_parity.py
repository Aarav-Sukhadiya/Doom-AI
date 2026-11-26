import json

import numpy as np
import pytest
import torch

from doom_env import make_env
from doom_env import spaces as S
from doomai.maps.generate import generate
from doomai.maps.grid import write_grid_wad
from doomai.ppo import REPO

import doomnet
from doomnet import USE_COOLDOWN, DoomNet, decode_action, observe, sample, stack, to_tensors
from landmarks import LandmarkTracker


def trainer_actions(net, env, seed, steps):
    obs, _ = env.reset(seed=seed)
    tracker = LandmarkTracker(mapping=net.use_map)
    last = None
    h = net.initial_state(1)
    prev = np.zeros((1, 9), np.float32)
    use_cd = np.zeros(1, np.int64)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            x = to_tensors(stack([observe(obs, tracker, last)]), "cpu")
            allowed = torch.from_numpy((use_cd == 0).astype(np.float32))
            logits, _, h = net(x, torch.from_numpy(prev), h, allowed)
            action = decode_action(sample(logits, greedy=True).numpy())
            out.append(action[0].copy())
            last = action[0].copy()
            use_cd = np.where(action[:, 7] == 1, USE_COOLDOWN, np.maximum(use_cd - 1, 0))
            prev = action.astype(np.float32)
            obs, _, term, trunc, _ = env.step(action[0])
            if term or trunc:
                break
    return out


def agent_actions(agent, env, seed, steps):
    obs, _ = env.reset(seed=seed)
    agent.reset()
    out = []
    for _ in range(steps):
        action = agent.act(obs)
        out.append(np.asarray(action).copy())
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    return out


@pytest.mark.parametrize("tier,slots,crumbs,mapping,weapon", [(2, False, False, False, False), (3, True, False, False, False), (4, True, True, False, False), (4, True, True, True, False), (1, True, True, True, True)])
@pytest.mark.parametrize("space", [False, True])
def test_trainer_and_submitted_agent_choose_identical_actions(tier, slots, crumbs, mapping, weapon, space, tmp_path, monkeypatch):
    torch.manual_seed(7)
    net = DoomNet(slots=slots, crumbs=crumbs, mapping=mapping, weapon=weapon, space=space).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    weights = tmp_path / "w"
    weights.mkdir()
    torch.save(net.state_dict(), weights / "model.pt")
    (weights / "config.json").write_text(json.dumps({"greedy": True, "slots": slots, "crumbs": crumbs, "map": mapping, "weapon": weapon, "space": space, "fire_mask": True}))

    import importlib.util

    spec = importlib.util.spec_from_file_location("submission_agent_parity", REPO / "submission" / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WEIGHTS_DIR", weights)
    agent = module.DoomAgent(S.observation_space(), S.action_space(), "cpu")

    level = generate(tier, 123)
    path = str(write_grid_wad(level.grid, tmp_path / "m.wad", player_angle=level.player_angle))
    env = make_env(path, "MAP01", episode_ticks=level.episode_ticks)
    # Both sides see the same game step by step. Their outputs must agree to float rounding, and
    # their actions must agree wherever the choice is not a near-tie (two options within 1e-4,
    # where rounding in a different kernel path may pick either).
    obs, _ = env.reset(seed=5)
    agent.reset()
    tracker = LandmarkTracker(mapping=net.use_map)
    last, h = None, net.initial_state(1)
    prev, use_cd = np.zeros((1, 9), np.float32), np.zeros(1, np.int64)
    seen = {}
    orig = agent.net.forward
    agent.net.forward = lambda *a, **k: seen.__setitem__("out", orig(*a, **k)) or seen["out"]
    for _ in range(100):
        with torch.no_grad():
            logits, _, h = net(to_tensors(stack([observe(obs, tracker, last)]), "cpu"), torch.from_numpy(prev), h,
                               torch.from_numpy((use_cd == 0).astype(np.float32)))
        action = decode_action(sample(logits, greedy=True).numpy())
        agent_action = agent.act(obs)
        assert torch.allclose(seen["out"][0], logits, atol=1e-5)
        margins = [torch.topk(p, 2).values for p in torch.split(logits[0], doomnet.HEAD_SIZES)]
        if all(float(v[0] - v[1]) > 1e-4 for v in margins):
            assert np.array_equal(action[0], agent_action)
        last = action[0].copy()
        use_cd = np.where(action[:, 7] == 1, USE_COOLDOWN, np.maximum(use_cd - 1, 0))
        prev = action.astype(np.float32)
        agent.prev_action, agent.use_cooldown = prev[0].copy(), int(use_cd[0])
        obs, _, term, trunc, _ = env.step(action[0])
        if term or trunc:
            break
    env.close()
    assert doomnet.NUM_BUTTONS == len(S.BUTTONS)


def test_agent_survives_garbage_observations():
    import importlib.util

    spec = importlib.util.spec_from_file_location("submission_agent_fuzz", REPO / "submission" / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.DoomAgent(S.observation_space(), S.action_space(), "cpu")
    space = S.observation_space()
    good = space.sample()
    bad = [
        {},
        {"rgb": None},
        dict(good, vitals=np.full(17, np.nan, np.float32)),
        dict(good, rays={"distance": np.full(64, np.inf, np.float32), "hit_type": np.full(64, 99)}),
        dict(good, rgb=np.zeros((2, 2, 3), np.uint8)),
    ]
    for obs in [good, *bad, good]:
        action = np.asarray(agent.act(obs))
        assert action.shape == (9,) and set(np.unique(action)) <= {0, 1}


def test_slot_handover_is_exact():
    torch.manual_seed(3)
    base = DoomNet(slots=False).eval()
    with_slots = DoomNet(slots=True).eval()
    missing, unexpected = with_slots.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("slot_") for k in missing)
    x = {k: torch.from_numpy(v) for k, v in stack([observe(S.observation_space().sample(), LandmarkTracker(), None) for _ in range(16)]).items()}
    x["slots"] = torch.randn_like(x["slots"]).abs()
    prev, h, allowed = torch.zeros(16, 9), torch.randn(16, base.hidden), torch.ones(16)
    with torch.no_grad():
        a = base(x, prev, h, allowed)
        b = with_slots(x, prev, h, allowed)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_landmark_tracker_records_locked_door_relative_to_agent(tmp_path):
    env = make_env(str(write_grid_wad("#######\n#P..B.E#\n#######", tmp_path / "m.wad")), "MAP01", episode_ticks=400)
    obs, _ = env.reset(seed=0)
    tracker = LandmarkTracker()
    feats = tracker.observe(obs, None)
    env.close()
    locked = feats[2]
    assert locked[0] == 1.0
    assert abs(locked[1] * 1024 - (256 - 96)) < 8 and abs(locked[2]) < 0.02


@pytest.mark.parametrize("pattern", [[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 1, 1, 1]])
def test_turn_model_matches_engine_after_reset(pattern, tmp_path):
    from landmarks import Odometry

    env = make_env(str(write_grid_wad("#######\n#.....#\n#..P..#\n#.....#\n#######", tmp_path / "m.wad")), "MAP01", episode_ticks=400)
    env.reset(seed=0)
    odo = Odometry()
    a0 = env.privileged_state()["angle"]
    for turn in pattern + [1, 0, 1]:
        action = np.zeros(9, np.int8)
        action[4] = turn
        predicted = odo.turn(action)
        env.step(action)
        a1 = env.privileged_state()["angle"]
        assert abs(((a1 - a0 + 180) % 360 - 180) - predicted) < 1e-3
        a0 = a1
    env.close()


def test_fire_mask_blocks_attack_only_when_enabled_and_no_enemy():
    net = DoomNet().eval()
    x = {k: torch.from_numpy(v) for k, v in stack([observe(S.observation_space().sample(), LandmarkTracker(), None) for _ in range(8)]).items()}
    x["attack_ok"] = torch.zeros(8)
    with torch.no_grad():
        net.policy.bias[9:11] = torch.tensor([-5.0, 5.0])
        masked, _, _ = net(x, torch.zeros(8, 9), net.initial_state(8), torch.ones(8))
        x.pop("attack_ok")
        free, _, _ = net(x, torch.zeros(8, 9), net.initial_state(8), torch.ones(8))
    assert (sample(masked)[:, 3] == 0).all()
    assert (sample(free, greedy=True)[:, 3] == 1).all()


def test_far_entity_kept_at_its_distance_with_entity_range():
    import numpy as np

    from landmarks import RAY_RANGE

    for cap, expected in ((1.0, 1.0), (4.0, 3.0)):
        tracker = LandmarkTracker(cap)
        tracker.reset()
        tracker._write(1, 3.0, 0.0, 1.0, tracker.entity_range)
        assert abs(np.hypot(*tracker.pos[1]) - expected * RAY_RANGE) < 1e-6


def test_crumb_handover_is_exact():
    torch.manual_seed(5)
    base = DoomNet(slots=True).eval()
    with_crumbs = DoomNet(slots=True, crumbs=True).eval()
    missing, unexpected = with_crumbs.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("crumb_") for k in missing)
    x = {k: torch.from_numpy(v) for k, v in stack([observe(S.observation_space().sample(), LandmarkTracker(), None) for _ in range(8)]).items()}
    x["crumbs"] = torch.rand_like(x["crumbs"])
    prev, h, allowed = torch.zeros(8, 9), torch.randn(8, base.hidden), torch.ones(8)
    with torch.no_grad():
        a = base(x, prev, h, allowed)
        b = with_crumbs(x, prev, h, allowed)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_crumbs_mark_ground_already_covered(tmp_path):
    from landmarks import CRUMB_FEATURES, CRUMB_RADII

    env = make_env(str(write_grid_wad("##################\n#P..............E#\n##################", tmp_path / "m.wad")), "MAP01", episode_ticks=1000)
    obs, _ = env.reset(seed=0)
    tracker = LandmarkTracker()
    tracker.observe(obs, None)
    forward = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
    for _ in range(30):
        obs, *_ = env.step(forward)
        tracker.observe(obs, forward)
    env.close()
    c = tracker.crumbs()
    assert c.shape == (CRUMB_FEATURES,)
    n = len(CRUMB_RADII)
    ahead, behind = c[2 : 2 + n], c[2 + 4 * n : 2 + 5 * n]
    assert c[0] > 0 and ahead.max() == 0 and behind[:2].min() > 0


def test_map_handover_is_exact():
    torch.manual_seed(6)
    base = DoomNet(slots=True, crumbs=True).eval()
    with_map = DoomNet(slots=True, crumbs=True, mapping=True).eval()
    missing, unexpected = with_map.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("map_") for k in missing)
    x = {k: torch.from_numpy(v) for k, v in stack([observe(S.observation_space().sample(), LandmarkTracker(mapping=True), None) for _ in range(8)]).items()}
    x["map"] = torch.randint(0, 256, x["map"].shape, dtype=torch.uint8)
    prev, h, allowed = torch.zeros(8, 9), torch.randn(8, base.hidden), torch.ones(8)
    with torch.no_grad():
        a = base(x, prev, h, allowed)
        b = with_map(x, prev, h, allowed)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_level_map_shows_corridor_and_path(tmp_path):
    import time

    from landmarks import MAP_VIEW

    env = make_env(str(write_grid_wad("####################\n#P.................E#\n####################", tmp_path / "m.wad")), "MAP01", episode_ticks=1000)
    obs, _ = env.reset(seed=0)
    tracker = LandmarkTracker(mapping=True)
    tracker.observe(obs, None)
    forward = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
    t0 = time.perf_counter()
    for _ in range(20):
        obs, *_ = env.step(forward)
        tracker.observe(obs, forward)
        view = tracker.map_view()
    per_step = (time.perf_counter() - t0) / 20
    env.close()
    assert view.shape == (4, MAP_VIEW, MAP_VIEW) and view.dtype == np.uint8
    mid = MAP_VIEW // 2
    ahead = view[:, : mid - 1, mid - 1 : mid + 1]
    assert ahead[0].max() == 255
    assert view[1, mid - 3 : mid + 3, :].max() == 255
    assert view[3, mid:, mid - 1 : mid + 1].max() == 255
    assert view[3, : mid - 3].max() == 0
    assert per_step < 0.01


def test_weapon_head_handover():
    torch.manual_seed(8)
    base = DoomNet(slots=True, crumbs=True, mapping=True).eval()
    with_weapon = DoomNet(slots=True, crumbs=True, mapping=True, weapon=True).eval()
    missing, unexpected = with_weapon.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("weapon.") for k in missing)
    x = {k: torch.from_numpy(v) for k, v in stack([observe(S.observation_space().sample(), LandmarkTracker(mapping=True), None) for _ in range(8)]).items()}
    prev, h, allowed = torch.zeros(8, 9), torch.randn(8, base.hidden), torch.ones(8)
    with torch.no_grad():
        a = base(x, prev, h, allowed)
        b = with_weapon(x, prev, h, allowed)
    assert torch.equal(a[0][:, :13], b[0][:, :13]) and torch.equal(a[1], b[1])
    press = torch.softmax(b[0][:, 13:], -1)[:, 1]
    assert torch.all(press < 0.01)
    choice = doomnet.sample(a[0])
    assert torch.all(choice[:, 5] == 0)
    assert not decode_action(choice.numpy())[:, 8].any()


def test_space_handover_and_features():
    from doomobs import SPACE_FEATURES, space

    torch.manual_seed(9)
    base = DoomNet(slots=True, crumbs=True, mapping=True, weapon=True).eval()
    with_space = DoomNet(slots=True, crumbs=True, mapping=True, weapon=True, space=True).eval()
    missing, unexpected = with_space.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("space_") for k in missing)
    x = {k: torch.from_numpy(v) for k, v in stack([observe(S.observation_space().sample(), LandmarkTracker(mapping=True), None) for _ in range(8)]).items()}
    prev, h, allowed = torch.zeros(8, 9), torch.randn(8, base.hidden), torch.ones(8)
    with torch.no_grad():
        a = base(x, prev, h, allowed)
        b = with_space(x, prev, h, allowed)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    obs = S.observation_space().sample()
    d = np.ones(64, np.float32)
    d[0] = 0.05
    d[16] = 0.3
    obs["rays"]["distance"] = d
    f = space(obs)
    assert f.shape == (SPACE_FEATURES,)
    assert abs(f[0] - 0.05) < 1e-6 and abs(f[-1] - 0.05) < 1e-6
    assert abs(f[2] - 0.3) < 1e-6 and f[4] == 1.0
