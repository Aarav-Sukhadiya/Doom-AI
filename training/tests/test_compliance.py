"""Checks that the submission stays within RULES.md and EVALUATION.md (PRD tests K6, K7, K9, K10)."""

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from doom_env import make_env
from doom_env import spaces as S
from doomai.maps.generate import generate
from doomai.maps.grid import write_grid_wad

REPO = Path(__file__).resolve().parents[2]
SUBMISSION = REPO / "submission"
sys.path.insert(0, str(SUBMISSION))
FORBIDDEN_MODULES = {"vizdoom", "socket", "requests", "urllib", "http", "subprocess", "multiprocessing", "doomai", "heapq", "networkx", "ctypes"}
FORBIDDEN_NAMES = {"privileged_state", "sector_polygons", "atan2", "arctan2", "_game", "system", "popen", "write_text", "write_bytes", "save"}


def submission_sources():
    return sorted(SUBMISSION.glob("*.py"))


@pytest.mark.parametrize("path", submission_sources(), ids=lambda p: p.name)
def test_k6_rules_scan(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in FORBIDDEN_MODULES, f"{path.name}: import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod not in FORBIDDEN_MODULES, f"{path.name}: from {node.module} import"
            assert not (node.module or "").startswith("doom_env.wrapper"), f"{path.name}: imports the engine wrapper"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_NAMES, f"{path.name}:{node.lineno} uses .{node.attr}"
        elif isinstance(node, ast.Name):
            assert node.id not in FORBIDDEN_NAMES, f"{path.name}:{node.lineno} uses {node.id}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
            assert not set(str(mode)) & set("wax+"), f"{path.name}:{node.lineno} opens a file for writing"


def test_k6_actions_come_from_the_network():
    """act() must pick buttons only by sampling or arg-maxing the policy logits, then masking."""
    agent_src = (SUBMISSION / "agent.py").read_text()
    assert re.search(r"logits, _, self\.h = self\.net\(", agent_src)
    assert "sample(logits" in agent_src and "decode_action(choice)" in agent_src


def _agent(weights_dir=None):
    spec = importlib.util.spec_from_file_location("submission_agent_compliance", SUBMISSION / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if weights_dir is not None:
        module.WEIGHTS_DIR = weights_dir
    return module


def test_k7_validator_on_template_and_generated_maps(tmp_path):
    maps = [f"{p}:MAP01" for p in sorted((REPO / "tests" / "fixtures").glob("*.wad"))]
    for tier in (1, 2, 3, 4):
        level = generate(tier, 31, difficulty=1.0)
        maps.append(f"{write_grid_wad(level.grid, tmp_path / f't{tier}.wad', player_angle=level.player_angle)}:MAP01")
    cmd = [sys.executable, str(REPO / "validate.py"), "--device", "cpu", "--ticks", "1400"]
    for m in maps:
        cmd += ["--map", m]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=REPO)
    text = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout + out.stderr)
    assert out.returncode == 0, text[-3000:]
    assert "✗" not in text and "FAILED" not in text, text[-3000:]
    assert text.count("Episode on") == len(maps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_k9_runs_under_the_2gb_vram_cap(tmp_path):
    code = f"""
import sys, numpy as np, torch
sys.path.insert(0, {str(SUBMISSION)!r}); sys.path.insert(0, {str(REPO / 'training')!r})
import importlib.util
spec = importlib.util.spec_from_file_location('a', {str(SUBMISSION / 'agent.py')!r}); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from doom_env import make_env, spaces as S
from doomai.maps.generate import generate
from doomai.maps.grid import write_grid_wad
agent = m.DoomAgent(S.observation_space(), S.action_space(), 'cuda')
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / total)
torch.cuda.reset_peak_memory_stats()
level = generate(3, 77, 1.0)
env = make_env(str(write_grid_wad(level.grid, {str(tmp_path / 'm.wad')!r})), 'MAP01', episode_ticks=1200)
obs, _ = env.reset(seed=0); agent.reset()
while True:
    obs, _, te, tr, _ = env.step(agent.act(obs))
    if te or tr: break
print('PEAK', torch.cuda.max_memory_allocated())
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    peak = int(out.stdout.split("PEAK")[-1])
    assert peak < 2 * 1024**3


def test_k10_long_episode_stays_finite_and_errors_fall_back(tmp_path):
    module = _agent()
    agent = module.DoomAgent(S.observation_space(), S.action_space(), "cpu")
    level = max((generate(4, s, 1.0) for s in range(40, 46)), key=lambda lv: len(lv.grid))
    env = make_env(str(write_grid_wad(level.grid, tmp_path / "m.wad", player_angle=level.player_angle)), "MAP01", episode_ticks=level.episode_ticks)
    obs, _ = env.reset(seed=0)
    agent.reset()
    steps = 0
    while True:
        action = agent.act(obs)
        assert action.shape == (9,) and set(np.unique(action)) <= {0, 1}
        assert torch.isfinite(agent.h).all() and agent.h.abs().max() <= 1.0 + 1e-5
        feats = agent.tracker.features()
        assert np.isfinite(feats).all()
        steps += 1
        obs, _, te, tr, _ = env.step(action)
        if te or tr:
            break
    env.close()
    assert steps > 100

    def broken(*args, **kwargs):
        raise RuntimeError("simulated failure")

    agent.net.forward = broken
    assert np.array_equal(agent.act(obs), np.zeros(9, np.int8))


def test_k11_shipped_weights_load_strictly():
    config = json.loads((SUBMISSION / "weights" / "config.json").read_text())
    state = torch.load(SUBMISSION / "weights" / "model.pt", map_location="cpu", weights_only=True)
    sys.path.insert(0, str(SUBMISSION))
    from doomnet import DoomNet

    net = DoomNet(hidden=config.get("hidden", 256), slots=config.get("slots", False))
    net.load_state_dict(state, strict=True)
