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
from doomnet import USE_COOLDOWN, DoomNet, decode_action, preprocess, sample, stack, to_tensors


def trainer_actions(net, env, seed, steps):
    obs, _ = env.reset(seed=seed)
    h = net.initial_state(1)
    prev = np.zeros((1, 9), np.float32)
    use_cd = np.zeros(1, np.int64)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            x = to_tensors(stack([preprocess(obs)]), "cpu")
            allowed = torch.from_numpy((use_cd == 0).astype(np.float32))
            logits, _, h = net(x, torch.from_numpy(prev), h, allowed)
            action = decode_action(sample(logits, greedy=True).numpy())
            out.append(action[0].copy())
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


@pytest.mark.parametrize("tier", [2, 3])
def test_trainer_and_submitted_agent_choose_identical_actions(tier, tmp_path, monkeypatch):
    torch.manual_seed(7)
    net = DoomNet().eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    weights = tmp_path / "w"
    weights.mkdir()
    torch.save(net.state_dict(), weights / "model.pt")
    (weights / "config.json").write_text(json.dumps({"greedy": True}))

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
