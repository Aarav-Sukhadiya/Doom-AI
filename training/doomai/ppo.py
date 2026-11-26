"""Recurrent PPO for DoomNet, written for this project.

Game workers run in separate processes and step a few ShapedDoomEnv instances each; the learner
batches every environment's decision on the GPU, keeps the GRU state, and trains on sequence chunks.
Levels come from the procedural generator; each tier's difficulty rises on its own as the agent's
exit rate on that tier improves.

    python -m doomai.ppo --run v1 --tiers 1 2 3 --hours 2

Stop a run cleanly (checkpoint saved, every game closed) with: touch runs/<run>/STOP
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submission"))

from doomnet import USE_COOLDOWN, DoomNet, decode_action, log_prob_entropy, observe, sample, stack, to_tensors  # noqa: E402
from landmarks import LandmarkTracker  # noqa: E402

TIER_COUNT = 4
KL_BAND = (0.005, 0.03)

GUIDE = """
# How to read this dashboard

The x-axis of every chart is the **update number**: all games play 128 decisions each, then the
network learns once. Use the *Relative* horizontal axis for hours. Set smoothing to about 0.8.

Open the **CUSTOM SCALARS** tab. Panels, top to bottom:

1. **Is it winning?** *Validation score* is the most honest curve: every snapshot is scored on the same
   unseen full-difficulty levels (higher is better; the shipped agent is the bar to beat).
   *Exit rate* and *practice difficulty* are measured on the practice levels: when the curriculum
   makes a tier harder its exit rate dips, so read them together. Difficulty rising toward 1.0 is progress.
2. **How it plays:** deaths (lower is better), keys and doors (higher), areas explored (higher), and
   *time used*: the share of the time limit each episode takes. Near 1.0 means it runs out of time.
3. **Is learning healthy?** These should stay inside their bands rather than go up or down:
   entropy (how random the choices are) near its target; KL (how much one update changes the
   policy) between the two dashed lines; explained variance (how well it predicts its own future
   reward) between 0.5 and 1.
4. **Where reward comes from:** the exit should be the largest source; if exploring or killing
   dominates while exits stay flat, the agent is farming side rewards.
5. **Speed and stability:** decisions per second; worker restarts and skipped updates should stay flat.
"""

LAYOUT = {
    "1 Is it winning?": {
        "Validation score on fixed unseen levels (higher is better)": ["Multiline", [r"eval/score"]],
        "Validation exit rate per tier (unseen, full difficulty)": ["Multiline", [r"eval/tier\d_exit"]],
        "Practice exit rate per tier (higher is better)": ["Multiline", [r"tier\d/exit"]],
        "Practice difficulty per tier (rises as it improves, goal 1.0)": ["Multiline", [r"curriculum/difficulty_t\d"]],
    },
    "2 How it plays": {
        "Deaths per episode (lower is better)": ["Multiline", [r"tier\d/dead"]],
        "Keys picked up per episode (higher is better)": ["Multiline", [r"tier\d/keys"]],
        "Doors opened per episode (higher is better)": ["Multiline", [r"tier\d/doors_opened"]],
        "Areas explored per episode (higher is better)": ["Multiline", [r"tier\d/sectors"]],
        "Share of time limit used (near 1.0 = running out of time)": ["Multiline", [r"tier\d/time_used"]],
        "Enemies killed per episode": ["Multiline", [r"tier\d/kills"]],
    },
    "3 Is learning healthy? (stay inside the bands)": {
        "Entropy vs target (randomness of choices)": ["Multiline", [r"ppo/ent$", r"ref/entropy_target"]],
        "KL per update (between the dashed band)": ["Multiline", [r"ppo/kl$", r"ref/kl_(low|high)"]],
        "Explained variance (0.5 to 1 is healthy)": ["Multiline", [r"ppo/explained_variance", r"ref/ev_low"]],
        "Clip fraction (0.05 to 0.3)": ["Multiline", [r"ppo/clipfrac"]],
        "Value loss and gradient size": ["Multiline", [r"ppo/vf", r"ppo/grad"]],
        "Entropy coefficient (self-adjusting)": ["Multiline", [r"ppo/entropy_coef"]],
    },
    "4 Where reward comes from": {
        "Reward per episode by source": ["Multiline", [r"reward_terms/.*"]],
        "Total reward per episode by tier": ["Multiline", [r"tier\d/return"]],
    },
    "5 Speed and stability": {
        "Decisions per second": ["Multiline", [r"perf/decisions_per_s"]],
        "Share of time spent playing vs learning": ["Multiline", [r"perf/rollout_share"]],
        "Worker restarts and skipped updates (should stay flat)": ["Multiline", [r"health/.*"]],
    },
}


def worker_main(remote, worker_id: int, envs_per_worker: int, difficulty, weights, seed: int, map_dir: str):
    torch.set_num_threads(1)
    from doomai.envs.shaped import ShapedDoomEnv
    from doomai.maps.generate import generate, seeds_for
    from doomai.maps.grid import write_grid_wad

    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(seed)
    build_rng = np.random.default_rng(seed + 7919)
    train_seeds = seeds_for("train", 200000)
    envs = [None] * envs_per_worker
    trackers = [LandmarkTracker() for _ in range(envs_per_worker)]
    tiers = [0] * envs_per_worker
    plays = [0] * envs_per_worker
    counter = [0]
    builder = ThreadPoolExecutor(max_workers=1)

    def build(i):
        w = np.array(weights[:], np.float64)
        tier = int(build_rng.choice(TIER_COUNT, p=w / w.sum())) + 1
        d = float(np.clip(difficulty[tier - 1] + build_rng.normal(0, 0.1), 0.0, 1.0))
        level = generate(tier, int(build_rng.choice(train_seeds)), difficulty=round(d, 2))
        counter[0] += 1
        path = Path(map_dir) / f"w{worker_id}_e{i}_{counter[0]}.wad"
        write_grid_wad(level.grid, path, player_angle=level.player_angle)
        return ShapedDoomEnv(str(path), "MAP01", episode_ticks=level.episode_ticks), tier

    pending = [builder.submit(build, i) for i in range(envs_per_worker)]

    def new_level(i):
        env, tier = pending[i].result()
        pending[i] = builder.submit(build, i)
        if envs[i] is not None:
            old = Path(envs[i].wad)
            envs[i].close()
            old.unlink(missing_ok=True)
        envs[i] = env
        tiers[i] = tier
        plays[i] = 0

    def reset(i):
        if envs[i] is None or plays[i] >= 2:
            new_level(i)
        plays[i] += 1
        obs, _ = envs[i].reset(seed=int(rng.integers(1 << 30)))
        trackers[i].reset()
        return observe(obs, trackers[i], None)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send([reset(i) for i in range(envs_per_worker)])
            elif cmd == "step":
                out = []
                for i, action in enumerate(data):
                    obs, reward, term, trunc, info = envs[i].step(action)
                    done = term or trunc
                    stats = None
                    if done:
                        stats = dict(info["episode_stats"], tier=tiers[i], terms=info["episode_terms"], cap=envs[i].episode_ticks / 4)
                        pre = reset(i)
                    else:
                        pre = observe(obs, trackers[i], action)
                    out.append((pre, reward, done, stats))
                remote.send(out)
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        traceback.print_exc()
        remote.send(("error", traceback.format_exc()))
    finally:
        for env in envs:
            if env is not None:
                env.close()
        for f in pending:
            try:
                f.result()[0].close()
            except Exception:
                pass
        builder.shutdown(wait=False)


class Workers:
    def __init__(self, n: int, per: int, difficulty, weights, seed: int, map_dir: str):
        self.n, self.per = n, per
        self.difficulty, self.weights, self.seed, self.map_dir = difficulty, weights, seed, map_dir
        self.ctx = mp.get_context("forkserver")
        self.remotes, self.procs = [None] * n, [None] * n
        for i in range(n):
            self._start(i, seed + i)

    def _start(self, i: int, seed: int) -> None:
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=worker_main, args=(child, i, self.per, self.difficulty, self.weights, seed, self.map_dir), daemon=True)
        p.start()
        child.close()
        self.remotes[i], self.procs[i] = parent, p

    def reset(self) -> list:
        for r in self.remotes:
            r.send(("reset", None))
        return [o for r in self.remotes for o in r.recv()]

    def step(self, actions: np.ndarray) -> tuple[list, int]:
        ids = list(range(self.n))
        self.send_step(ids, actions)
        return self.recv_step(ids)

    def send_step(self, ids: list[int], actions: np.ndarray) -> None:
        for k, i in enumerate(ids):
            self.remotes[i].send(("step", actions[k * self.per : (k + 1) * self.per]))

    def recv_step(self, ids: list[int]) -> tuple[list, int]:
        results, restarts = [], 0
        for i in ids:
            r = self.remotes[i]
            try:
                out = r.recv()
                if isinstance(out, tuple) and out and out[0] == "error":
                    raise RuntimeError(out[1])
            except (EOFError, RuntimeError, ConnectionError) as err:
                print(f"worker {i} died ({str(err).splitlines()[-1] if str(err) else 'EOF'}); restarting", flush=True)
                self.procs[i].kill()
                self._start(i, self.seed + 1000 + i + int(time.time()) % 1000)
                restarts += 1
                self.remotes[i].send(("reset", None))
                obs = self.remotes[i].recv()
                out = [(o, 0.0, True, None) for o in obs]
            results.extend(out)
        return results, restarts

    def close(self) -> None:
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)


class RunningStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = float(x.mean()), float(x.var()), x.size
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return math.sqrt(self.var) + 1e-8


def save_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def gae(rewards, values, dones, last_value, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="dev")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--envs-per-worker", type=int, default=2)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--tiers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--tier-weights", type=float, nargs="+", default=None)
    p.add_argument("--start-difficulty", type=float, default=0.0)
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--max-updates", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--init", default="", help="start from these weights but fresh optimizer and counters")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--slots", action="store_true", help="enable the landmark slot memory read path")
    p.add_argument("--target-entropy", type=float, default=0.0, help="adapt the entropy coefficient to hold this policy entropy (0 = fixed --ent)")
    p.add_argument("--eval-every", type=int, default=200, help="score a snapshot on validation levels every N updates in the background (0 = off)")
    p.add_argument("--eval-procs", type=int, default=3)
    p.add_argument("--eval-per-tier", type=int, default=10)
    args = p.parse_args(argv)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = REPO / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(run_dir / "tb"))
    writer.add_custom_scalars(LAYOUT)
    writer.add_text("guide", GUIDE, 0)

    net = DoomNet(slots=args.slots).to(device)
    (run_dir / "config.json").write_text(json.dumps({"slots": args.slots, "hidden": net.hidden}))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    update, total_steps = 0, 0
    difficulty = mp.Array("d", [args.start_difficulty] * TIER_COUNT, lock=False)
    weights = mp.Array("d", [0.0] * TIER_COUNT, lock=False)
    tw = args.tier_weights or [1.0] * len(args.tiers)
    for t, w in zip(args.tiers, tw):
        weights[t - 1] = w
    ret_norm = RunningStd()
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        update, total_steps = ck["update"], ck["steps"]
        for i, d in enumerate(ck["difficulty"]):
            difficulty[i] = d
        ret_norm.mean, ret_norm.var, ret_norm.count = ck["ret_norm"]
        print(f"resumed {args.resume} at update {update}, {total_steps} steps", flush=True)
    elif args.init:
        state = torch.load(args.init, map_location=device, weights_only=False)
        missing, unexpected = net.load_state_dict(state["model"] if "model" in state else state, strict=False)
        if unexpected or any(not k.startswith("slot_") for k in missing):
            raise RuntimeError(f"--init mismatch: missing {missing}, unexpected {unexpected}")
        print(f"initialised from {args.init}; new parameters: {missing}", flush=True)

    map_dir = tempfile.mkdtemp(prefix=f"doomai_{args.run}_")
    workers = Workers(args.workers, args.envs_per_worker, difficulty, weights, args.seed * 1000, map_dir)
    B, T = args.workers * args.envs_per_worker, args.rollout
    obs_all = workers.reset()
    half = args.workers // 2 if args.workers > 1 else args.workers
    group_workers = [list(range(half)), list(range(half, args.workers))] if half < args.workers else [list(range(args.workers))]
    groups = []
    for ids in group_workers:
        lo, hi = ids[0] * args.envs_per_worker, (ids[-1] + 1) * args.envs_per_worker
        groups.append({
            "ids": ids, "lo": lo, "hi": hi, "n": hi - lo,
            "obs": obs_all[lo:hi],
            "h": net.initial_state(hi - lo, device),
            "prev": np.zeros((hi - lo, 9), np.float32),
            "cd": np.zeros(hi - lo, np.int64),
        })
    ret_acc = np.zeros(B, np.float64)
    disc_ret = np.zeros(B, np.float64)

    tier_hist = {t: deque(maxlen=100) for t in range(1, TIER_COUNT + 1)}
    recent = deque(maxlen=200)
    skipped, restarts_total = 0, 0
    start = time.time()
    deadline = start + args.hours * 3600

    ent_coef = args.ent
    evaluator = None
    while time.time() < deadline and (not args.max_updates or update < args.max_updates):
        if (run_dir / "STOP").exists():
            print("STOP file found: saving and shutting down", flush=True)
            break
        t0 = time.time()
        keys = ("obs", "prev", "allowed", "choice", "logp", "value", "reward", "done")
        gbuf = [{k: [] for k in keys} for _ in groups]
        L = args.chunk
        n_chunks = T // L
        gh_chunks = [torch.zeros(n_chunks, g["n"], net.hidden, device=device) for g in groups]
        net.eval()

        def act(gi: int, t: int) -> None:
            g = groups[gi]
            if t % L == 0:
                gh_chunks[gi][t // L] = g["h"]
            x = to_tensors(stack(g["obs"]), device)
            prev_t = torch.from_numpy(g["prev"]).to(device)
            allowed = torch.from_numpy((g["cd"] == 0).astype(np.float32)).to(device)
            logits, value, g["h"] = net(x, prev_t, g["h"], allowed)
            choice = sample(logits)
            logp, _ = log_prob_entropy(logits, choice)
            actions = decode_action(choice.cpu().numpy())
            workers.send_step(g["ids"], actions)
            g["actions"] = actions
            gb = gbuf[gi]
            for k, v in (("obs", x), ("prev", prev_t), ("allowed", allowed), ("choice", choice), ("logp", logp), ("value", value)):
                gb[k].append(v)

        def collect(gi: int) -> int:
            nonlocal ret_acc, disc_ret
            g = groups[gi]
            results, restarts = workers.recv_step(g["ids"])
            lo, hi = g["lo"], g["hi"]
            rewards = np.array([r[1] for r in results], np.float32)
            dones = np.array([r[2] for r in results], np.float32)
            for i, r in enumerate(results):
                if r[3] is not None:
                    st = r[3]
                    st["return"] = ret_acc[lo + i] + rewards[i]
                    tier_hist[st["tier"]].append(st)
                    recent.append(st)
            ret_acc[lo:hi] = (ret_acc[lo:hi] + rewards) * (1.0 - dones)
            disc_ret[lo:hi] = disc_ret[lo:hi] * args.gamma + rewards
            ret_norm.update(disc_ret[lo:hi])
            disc_ret[lo:hi] *= 1.0 - dones
            gbuf[gi]["reward"].append(torch.from_numpy(rewards).to(device))
            gbuf[gi]["done"].append(torch.from_numpy(dones).to(device))
            actions = g["actions"]
            g["cd"] = np.where(actions[:, 7] == 1, USE_COOLDOWN, np.maximum(g["cd"] - 1, 0))
            g["cd"] = np.where(dones > 0, 0, g["cd"])
            g["prev"] = actions.astype(np.float32) * (1.0 - dones)[:, None]
            g["h"] = g["h"] * torch.from_numpy(1.0 - dones).to(device).unsqueeze(-1)
            g["obs"] = [r[0] for r in results]
            return restarts

        with torch.no_grad():
            for t in range(T):
                for gi in range(len(groups)):
                    if t > 0:
                        restarts_total += collect(gi)
                    act(gi, t)
            last_values = []
            for gi, g in enumerate(groups):
                restarts_total += collect(gi)
                x = to_tensors(stack(g["obs"]), device)
                allowed = torch.from_numpy((g["cd"] == 0).astype(np.float32)).to(device)
                _, lv, _ = net(x, torch.from_numpy(g["prev"]).to(device), g["h"], allowed)
                last_values.append(lv)
            last_value = torch.cat(last_values)

        buf = {k: [torch.cat([gb[k][t] for gb in gbuf], dim=0) if k != "obs" else {kk: torch.cat([gb[k][t][kk] for gb in gbuf], dim=0) for kk in gbuf[0][k][t]} for t in range(T)] for k in keys}
        h_chunks = torch.cat(gh_chunks, dim=1)

        rewards = torch.stack(buf["reward"]) / ret_norm.std
        values = torch.stack(buf["value"])
        dones = torch.stack(buf["done"])
        adv, returns = gae(rewards, values, dones, last_value, args.gamma, args.lam)
        rollout_time = time.time() - t0

        obs_b = {k: torch.stack([o[k] for o in buf["obs"]]) for k in buf["obs"][0]}
        prev_b, allowed_b = torch.stack(buf["prev"]), torch.stack(buf["allowed"])
        choice_b, logp_b = torch.stack(buf["choice"]), torch.stack(buf["logp"])
        net.train()
        seqs = [(c, b) for c in range(n_chunks) for b in range(B)]
        mb_size = max(1, len(seqs) // args.minibatches)
        stats = {k: [] for k in ("pg", "vf", "ent", "kl", "clipfrac", "grad")}
        bad_update = False
        for _ in range(args.epochs):
            order = np.random.permutation(len(seqs))
            for m in range(0, len(seqs), mb_size):
                idx = order[m : m + mb_size]
                cs = torch.tensor([seqs[i][0] for i in idx], device=device)
                bs = torch.tensor([seqs[i][1] for i in idx], device=device)
                t_idx = (cs[None, :] * L + torch.arange(L, device=device)[:, None])
                b_idx = bs[None, :].expand(L, -1)
                xt = {key: v[t_idx, b_idx] for key, v in obs_b.items()}
                logits, val = net.forward_sequence(xt, prev_b[t_idx, b_idx], h_chunks[cs, bs], allowed_b[t_idx, b_idx], dones[t_idx, b_idx])
                flat_choice = choice_b[t_idx, b_idx].reshape(-1, choice_b.shape[-1])
                lp, ent = log_prob_entropy(logits, flat_choice)
                a = adv[t_idx, b_idx].reshape(-1)
                ret = returns[t_idx, b_idx].reshape(-1)
                old = logp_b[t_idx, b_idx].reshape(-1)
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(lp - old)
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vf = 0.5 * (val - ret).pow(2).mean()
                loss = pg + args.vf * vf - ent_coef * ent.mean()
                if not torch.isfinite(loss):
                    bad_update = True
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
                if not torch.isfinite(grad):
                    bad_update = True
                    opt.zero_grad(set_to_none=True)
                    break
                opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.item())
                    stats["vf"].append(vf.item())
                    stats["ent"].append(ent.mean().item())
                    stats["kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean().item())
                    stats["grad"].append(grad.item())
            if bad_update:
                break
        if bad_update:
            skipped += 1
            print(f"update {update}: non-finite loss or gradient, update skipped ({skipped} so far)", flush=True)
            if args.resume or (run_dir / "latest.pt").exists():
                ck = torch.load(run_dir / "latest.pt", map_location=device, weights_only=False)
                net.load_state_dict(ck["model"])
                opt.load_state_dict(ck["optimizer"])
            writer.add_scalar("health/skipped_updates", skipped, update)
            continue

        update += 1
        total_steps += T * B
        if args.target_entropy > 0 and stats["ent"]:
            step = 1.05 if np.mean(stats["ent"]) < args.target_entropy else 1 / 1.05
            ent_coef = float(np.clip(ent_coef * step, 1e-4, 0.05))
        for tier in args.tiers:
            hist = tier_hist[tier]
            if len(hist) >= 30:
                rate = float(np.mean([s["exit"] for s in hist]))
                if rate > 0.6 and difficulty[tier - 1] < 1.0:
                    difficulty[tier - 1] = min(1.0, difficulty[tier - 1] + 0.1)
                    hist.clear()
                elif rate < 0.15 and difficulty[tier - 1] > 0.0:
                    difficulty[tier - 1] = max(0.0, difficulty[tier - 1] - 0.05)
                    hist.clear()

        elapsed = time.time() - start
        sps = T * B / (time.time() - t0)
        with torch.no_grad():
            var_ret = returns.var()
            ev = float(1 - (returns - values).var() / var_ret) if var_ret > 0 else 0.0
        w = writer
        w.add_scalar("perf/decisions_per_s", sps, update)
        w.add_scalar("perf/rollout_share", rollout_time / (time.time() - t0), update)
        w.add_scalar("perf/total_decisions", total_steps, update)
        for k, v in stats.items():
            if v:
                w.add_scalar(f"ppo/{k}", float(np.mean(v)), update)
        w.add_scalar("ppo/explained_variance", ev, update)
        w.add_scalar("ppo/entropy_coef", ent_coef, update)
        w.add_scalar("ref/kl_low", KL_BAND[0], update)
        w.add_scalar("ref/kl_high", KL_BAND[1], update)
        w.add_scalar("ref/ev_low", 0.5, update)
        if args.target_entropy > 0:
            w.add_scalar("ref/entropy_target", args.target_entropy, update)
        w.add_scalar("ppo/return_std", ret_norm.std, update)
        w.add_scalar("health/worker_restarts", restarts_total, update)
        for tier in range(1, TIER_COUNT + 1):
            hist = list(tier_hist[tier])
            if weights[tier - 1] > 0:
                w.add_scalar(f"curriculum/difficulty_t{tier}", difficulty[tier - 1], update)
            if hist:
                for key in ("exit", "dead", "kills", "sectors", "keys", "doors_opened", "steps", "return"):
                    w.add_scalar(f"tier{tier}/{key}", float(np.mean([s[key] for s in hist])), update)
                w.add_scalar(f"tier{tier}/time_used", float(np.mean([s["steps"] / max(1.0, s.get("cap", s["steps"])) for s in hist])), update)
        if recent:
            for key in recent[-1]["terms"]:
                w.add_scalar(f"reward_terms/{key}", float(np.mean([s["terms"].get(key, 0.0) for s in recent])), update)

        if update % 5 == 0 or update == 1:
            line = " ".join(
                f"t{t}:exit={np.mean([s['exit'] for s in tier_hist[t]]):.2f}/d={difficulty[t-1]:.1f}"
                for t in args.tiers if tier_hist[t]
            )
            print(
                f"[{elapsed/60:6.1f} min] update {update} steps {total_steps} {sps:5.0f}/s "
                f"kl {np.mean(stats['kl']):.4f} clip {np.mean(stats['clipfrac']):.2f} ent {np.mean(stats['ent']):.2f} ev {ev:.2f} | {line}",
                flush=True,
            )
        if update % args.save_every == 0:
            ck = {
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "update": update,
                "steps": total_steps,
                "difficulty": list(difficulty[:]),
                "ret_norm": (ret_norm.mean, ret_norm.var, ret_norm.count),
                "args": vars(args),
            }
            save_atomic(ck, run_dir / "latest.pt")
            if update % (args.save_every * 10) == 0:
                save_atomic(net.state_dict(), run_dir / f"model_u{update}.pt")
        if args.eval_every and update % args.eval_every == 0 and evaluator is None:
            snap = run_dir / f"model_u{update}.pt"
            if not snap.exists():
                save_atomic(net.state_dict(), snap)
            out = run_dir / f"eval_u{update}.json"
            evaluator = (update, out, subprocess.Popen(
                ["nice", "-n", "10", sys.executable, "-m", "doomai.evaluate", "--weights", str(snap), "--split", "val",
                 "--per-tier", str(args.eval_per_tier), "--procs", str(args.eval_procs), "--out", str(out)],
                cwd=str(REPO), env=dict(os.environ, PYTHONPATH=str(REPO / "training")),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if evaluator is not None and evaluator[2].poll() is not None:
            ev_update, ev_out, _ = evaluator
            evaluator = None
            if ev_out.exists():
                summary = json.loads(ev_out.read_text())["summary"]
                w.add_scalar("eval/score", summary["score"], ev_update)
                for tier in range(1, TIER_COUNT + 1):
                    if f"tier{tier}" in summary:
                        w.add_scalar(f"eval/tier{tier}_exit", summary[f"tier{tier}"]["exit"], ev_update)
                print(f"validation score at update {ev_update}: {summary['score']}", flush=True)

    ck = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "update": update,
        "steps": total_steps,
        "difficulty": list(difficulty[:]),
        "ret_norm": (ret_norm.mean, ret_norm.var, ret_norm.count),
        "args": vars(args),
    }
    save_atomic(ck, run_dir / "latest.pt")
    save_atomic(net.state_dict(), run_dir / "model.pt")
    workers.close()
    writer.close()
    print(f"done: {update} updates, {total_steps} decisions in {(time.time()-start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
