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
    tracker = LandmarkTracker()
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


@pytest.mark.parametrize("tier,slots", [(2, False), (3, True)])
def test_trainer_and_submitted_agent_choose_identical_actions(tier, slots, tmp_path, monkeypatch):
    torch.manual_seed(7)
    net = DoomNet(slots=slots).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    weights = tmp_path / "w"
    weights.mkdir()
    torch.save(net.state_dict(), weights / "model.pt")
    (weights / "config.json").write_text(json.dumps({"greedy": True, "slots": slots}))

    import importlib.util

    spec = importlib.util.spec_from_file_location("submission_agent_parity", REPO / "submission" / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WEIGHTS_DIR", weights)
    agent = module.DoomAgent(S.observation_space(), S.action_space(), "cpu")

    level = generate(tier, 123)
    path = str(write_grid_wad(level.grid, tmp_path / "m.wad", player_angle=level.player_angle))
    env = make_env(path, "MAP01", episode_ticks=level.episode_ticks)
    a = trainer_actions(net, env, 5, 100)
    b = agent_actions(agent, env, 5, 100)
    env.close()
    assert len(a) == len(b)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
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
