"""Recurrent PPO for DoomNet, written for this project.

Game workers run in separate processes and step a few ShapedDoomEnv instances each; the learner
batches every environment's decision on the GPU, keeps the GRU state, and trains on sequence chunks.
Levels come from the procedural generator; each tier's difficulty rises on its own as the agent's
exit rate on that tier improves.

    python -m doomai.ppo --run v1 --tiers 1 2 3 --hours 2
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
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
        path = Path(map_dir) / f"w{worker_id}_e{i}_{counter[0] % 8}.wad"
        write_grid_wad(level.grid, path, player_angle=level.player_angle)
        return ShapedDoomEnv(str(path), "MAP01", episode_ticks=level.episode_ticks), tier

    pending = [builder.submit(build, i) for i in range(envs_per_worker)]

    def new_level(i):
        env, tier = pending[i].result()
        pending[i] = builder.submit(build, i)
        if envs[i] is not None:
            envs[i].close()
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
                        stats = dict(info["episode_stats"], tier=tiers[i], terms=info["episode_terms"])
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
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.per : (i + 1) * self.per]))
        results, restarts = [], 0
        for i, r in enumerate(self.remotes):
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
    args = p.parse_args(argv)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = REPO / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(run_dir / "tb"))

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
    obs = workers.reset()
    h = net.initial_state(B, device)
    prev_action = np.zeros((B, 9), np.float32)
    use_cd = np.zeros(B, np.int64)
    ret_acc = np.zeros(B, np.float64)

    tier_hist = {t: deque(maxlen=100) for t in range(1, TIER_COUNT + 1)}
    recent = deque(maxlen=200)
    skipped, restarts_total = 0, 0
    start = time.time()
    deadline = start + args.hours * 3600

    while time.time() < deadline and (not args.max_updates or update < args.max_updates):
        t0 = time.time()
        buf = {k: [] for k in ("obs", "prev", "allowed", "choice", "logp", "value", "reward", "done")}
        L = args.chunk
        n_chunks = T // L
        h_chunks = torch.zeros(n_chunks, B, net.hidden, device=device)
        net.eval()
        with torch.no_grad():
            for t in range(T):
                if t % L == 0:
                    h_chunks[t // L] = h
                x = to_tensors(stack(obs), device)
                prev_t = torch.from_numpy(prev_action).to(device)
                allowed = torch.from_numpy((use_cd == 0).astype(np.float32)).to(device)
                logits, value, h = net(x, prev_t, h, allowed)
                choice = sample(logits)
                logp, _ = log_prob_entropy(logits, choice)
                choice_np = choice.cpu().numpy()
                actions = decode_action(choice_np)
                results, restarts = workers.step(actions)
                restarts_total += restarts
                next_obs = [r[0] for r in results]
                rewards = np.array([r[1] for r in results], np.float32)
                dones = np.array([r[2] for r in results], np.float32)
                for i, r in enumerate(results):
                    if r[3] is not None:
                        s = r[3]
                        s["return"] = ret_acc[i] + rewards[i]
                        tier_hist[s["tier"]].append(s)
                        recent.append(s)
                ret_acc = (ret_acc + rewards) * (1.0 - dones)
                buf["obs"].append(x)
                buf["prev"].append(prev_t)
                buf["allowed"].append(allowed)
                buf["choice"].append(choice)
                buf["logp"].append(logp)
                buf["value"].append(value)
                buf["reward"].append(torch.from_numpy(rewards).to(device))
                buf["done"].append(torch.from_numpy(dones).to(device))
                use_cd = np.where(actions[:, 7] == 1, USE_COOLDOWN, np.maximum(use_cd - 1, 0))
                prev_action = actions.astype(np.float32)
                done_mask = torch.from_numpy(1.0 - dones).to(device).unsqueeze(-1)
                h = h * done_mask
                prev_action *= (1.0 - dones)[:, None]
                use_cd = np.where(dones > 0, 0, use_cd)
                obs = next_obs
            x = to_tensors(stack(obs), device)
            _, last_value, _ = net(x, torch.from_numpy(prev_action).to(device), h, torch.from_numpy((use_cd == 0).astype(np.float32)).to(device))

        rewards = torch.stack(buf["reward"])
        ret_norm.update(rewards.cpu().numpy())
        rewards = rewards / ret_norm.std
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
                loss = pg + args.vf * vf - args.ent * ent.mean()
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
        w.add_scalar("ppo/return_std", ret_norm.std, update)
        w.add_scalar("health/worker_restarts", restarts_total, update)
        for tier in range(1, TIER_COUNT + 1):
            hist = list(tier_hist[tier])
            if weights[tier - 1] > 0:
                w.add_scalar(f"curriculum/difficulty_t{tier}", difficulty[tier - 1], update)
            if hist:
                for key in ("exit", "dead", "kills", "sectors", "keys", "doors_opened", "steps", "return"):
                    w.add_scalar(f"tier{tier}/{key}", float(np.mean([s[key] for s in hist])), update)
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
