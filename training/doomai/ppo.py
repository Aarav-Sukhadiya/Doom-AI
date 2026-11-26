"""Recurrent PPO for DoomNet, written for this project.

Game workers run in separate processes and step a few ShapedDoomEnv instances each; the learner
batches every environment's decision on the GPU, keeps the GRU state, and trains on sequence chunks.
Levels come from the procedural generator; each tier's difficulty rises on its own as the agent's
exit rate on that tier improves.

    python -m doomai.train --run v1 --tiers 1 2 3 --hours 2   (doomai.train keeps worker processes small)

Stop a run cleanly (checkpoint saved, every game closed) with: touch runs/<run>/STOP
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import deque
from multiprocessing.connection import wait
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submission"))

from doomnet import USE_COOLDOWN, DoomNet, decode_action, log_prob_entropy, sample, stack, to_tensors  # noqa: E402

from doomai.worker import worker_main  # noqa: E402

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
        "Validation score on fixed unseen levels (higher is better)": ["Multiline", [r"eval/score", r"ref/shipped_score"]],
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


class Workers:
    def __init__(self, n: int, per: int, difficulty, weights, seed: int, map_dir: str, reward: dict | None = None, hard_share: float = 0.0, freedoom_share: float = 0.0, entity_range: float = 1.0, mapping: bool = False):
        self.n, self.per = n, per
        self.entity_range, self.mapping = entity_range, mapping
        self.reward, self.hard_share, self.freedoom_share = reward, hard_share, freedoom_share
        self.difficulty, self.weights, self.seed, self.map_dir = difficulty, weights, seed, map_dir
        self.ctx = mp.get_context("forkserver")
        self.ctx.set_forkserver_preload(["doomai.worker"])
        self.remotes, self.procs = [None] * n, [None] * n
        for i in range(n):
            self._start(i, seed + i)

    def _start(self, i: int, seed: int) -> None:
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=worker_main, args=(child, i, self.per, self.difficulty, self.weights, seed, self.map_dir, self.reward, self.hard_share, self.freedoom_share, self.entity_range, self.mapping), daemon=True)
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
                obs = self._restart(i)
                restarts += 1
                out = [(o, 0.0, True, None) for o in obs]
            results.extend(out)
        return results, restarts

    def _restart(self, i: int, attempts: int = 10) -> list:
        """Restart worker i until its reset returns one observation per env (a failed reset sends an error tuple)."""
        for attempt in range(attempts):
            self.procs[i].kill()
            self.remotes[i].close()
            self._start(i, self.seed + 1000 * (attempt + 1) + i + int(time.time()) % 1000)
            try:
                self.remotes[i].send(("reset", None))
                obs = self.remotes[i].recv()
            except (EOFError, ConnectionError, BrokenPipeError):
                obs = None
            if isinstance(obs, list) and len(obs) == self.per:
                return obs
            print(f"worker {i} restart attempt {attempt + 1} failed; retrying in 5 s", flush=True)
            time.sleep(5)
        raise RuntimeError(f"worker {i} could not be restarted after {attempts} attempts")

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


class Decide:
    """One rollout decision for a batch of games: forward pass, sampled choice, its log-prob, value.

    At rollout batch sizes the eager forward is bound by kernel-launch overhead (about 4 ms for 28 games),
    so on CUDA each batch size is recorded once as a CUDA graph and replayed (about 0.5 ms). Replay runs
    the same kernels on the same weights (optimizer steps update them in place), so logits, values and
    GRU states match eager mode bit for bit; only the sampling noise comes from a different random draw.
    """

    def __init__(self, net: torch.nn.Module, graphs: bool):
        self.net, self.graphs, self.cache, self.pool = net, graphs, {}, None

    def _run(self, x, prev, h, allowed):
        logits, value, h_new = self.net(x, prev, h, allowed)
        choice = sample(logits)
        logp, _ = log_prob_entropy(logits, choice)
        return choice, logp, value, h_new

    def _capture(self, x, prev, h, allowed):
        static = ({k: v.clone() for k, v in x.items()}, prev.clone(), h.clone(), allowed.clone())
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up (also lets cudnn pick its kernels for this shape)
            for _ in range(3):
                self._run(*static)
        torch.cuda.current_stream().wait_stream(side)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._run(*static)
        return static, out, graph

    def __call__(self, x, prev, h, allowed):
        """Returns (choice, logp, value, h_new); with graphs these are reused buffers, valid until the next call."""
        if not self.graphs:
            return self._run(x, prev, h, allowed)
        n = prev.shape[0]
        if n not in self.cache:
            try:
                self.cache[n] = self._capture(x, prev, h, allowed)
            except Exception as err:  # never let an optimisation stop a run
                print(f"CUDA graph capture failed ({type(err).__name__}: {err}); using eager mode", flush=True)
                self.graphs = False
                return self._run(x, prev, h, allowed)
        (sx, sprev, sh, sallowed), out, graph = self.cache[n]
        for k, v in x.items():
            sx[k].copy_(v)
        sprev.copy_(prev)
        sh.copy_(h)
        sallowed.copy_(allowed)
        graph.replay()
        return out


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
    p.add_argument("--crumbs", action="store_true", help="breadcrumb memory input: how often the agent has been around here")
    p.add_argument("--map", action="store_true", help="level map memory input: 32x32 map built from rays and odometry")
    p.add_argument("--weapon", action="store_true", help="learned next-weapon action head")
    p.add_argument("--space", action="store_true", help="room-around-me input: nearest wall and mean distance in 8 directions")
    p.add_argument("--entity-range", type=float, default=4.0, help="landmark distance cap for entities, in 1024-unit steps (1.0 before main10)")
    p.add_argument("--target-entropy", type=float, default=0.0, help="adapt the entropy coefficient to hold this policy entropy (0 = fixed --ent)")
    p.add_argument("--eval-every", type=int, default=200, help="score a snapshot on validation levels every N updates in the background (0 = off)")
    p.add_argument("--eval-procs", type=int, default=3)
    p.add_argument("--eval-per-tier", type=int, default=10)
    p.add_argument("--shipped-score", type=float, default=0.0, help="validation score of the shipped agent, drawn as a reference line")
    p.add_argument("--subgoal", type=float, default=0.0, help="subgoal shaping scale per grid cell of progress (training only; 0 = off)")
    p.add_argument("--raise-at", type=float, default=0.6, help="raise a tier's difficulty when its rolling exit rate passes this")
    p.add_argument("--hard-share", type=float, default=0.0, help="share of practice levels generated at full difficulty")
    p.add_argument("--fire-mask", action="store_true", help="block ATTACK when no enemy is in view (action mask from observation fields)")
    p.add_argument("--freedoom-share", type=float, default=0.0, help="share of practice episodes on real Freedoom 2 levels MAP01-MAP20 (logged as tier5)")
    p.add_argument("--eval-freedoom", action="store_true", help="background validation also plays held-out Freedoom 2 MAP21-MAP32")
    p.add_argument("--no-cuda-graphs", action="store_true", help="run rollout decisions in eager mode (same results, slower)")
    args = p.parse_args(argv)
    if args.rollout % args.chunk:
        p.error(f"--rollout ({args.rollout}) must be a multiple of --chunk ({args.chunk})")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    shuffle_rng = np.random.default_rng(args.seed)  # minibatch order, so --seed makes CPU runs repeatable
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    run_dir = REPO / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(run_dir / "tb"))
    writer.add_custom_scalars(LAYOUT)
    writer.add_text("guide", GUIDE, 0)

    net = DoomNet(slots=args.slots, crumbs=args.crumbs, mapping=args.map, weapon=args.weapon, space=args.space).to(device)
    (run_dir / "config.json").write_text(json.dumps({"slots": args.slots, "hidden": net.hidden, "fire_mask": args.fire_mask, "entity_range": args.entity_range, "crumbs": args.crumbs, "map": args.map, "weapon": args.weapon, "space": args.space}))

    def batch(observations: list) -> dict:
        arrays = stack(observations)
        if not args.fire_mask:
            arrays.pop("attack_ok", None)
        return to_tensors(arrays, device)
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
        if unexpected or any(not k.startswith(("slot_", "crumb_", "map_", "weapon.", "space_")) for k in missing):
            raise RuntimeError(f"--init mismatch: missing {missing}, unexpected {unexpected}")
        print(f"initialised from {args.init}; new parameters: {missing}", flush=True)

    map_dir = str(run_dir / "levels")
    Path(map_dir).mkdir(parents=True, exist_ok=True)
    for stale in Path(map_dir).glob("*.wad"):
        stale.unlink(missing_ok=True)
    workers = Workers(args.workers, args.envs_per_worker, difficulty, weights, args.seed * 1000, map_dir,
                      reward={"subgoal": args.subgoal}, hard_share=args.hard_share, freedoom_share=args.freedoom_share,
                      entity_range=args.entity_range, mapping=args.map)
    W, P = args.workers, args.envs_per_worker
    B, T, L = W * P, args.rollout, args.chunk
    n_chunks = T // L
    obs_all = workers.reset()
    # Per-env state carried across decisions and updates; env e lives on worker e // P.
    cur_obs = [obs_all[w * P : (w + 1) * P] for w in range(W)]
    h_state = net.initial_state(B, device)
    prev = np.zeros((B, 9), np.float32)
    cd = np.zeros(B, np.int64)
    ret_acc = np.zeros(B, np.float64)
    disc_ret = np.zeros(B, np.float64)
    # Rollout storage (T, B, ...): GPU buffers made on the first decision and overwritten every update.
    rb: dict = {"obs": {}, "h": torch.zeros(n_chunks, B, net.hidden, device=device)}
    decide = Decide(net, graphs=device.type == "cuda" and not args.no_cuda_graphs)
    rew_np = np.zeros((T, B), np.float32)
    done_np = np.zeros((T, B), np.float32)

    tier_hist = {t: deque(maxlen=100) for t in range(1, TIER_COUNT + 2)}
    recent = deque(maxlen=200)
    skipped, restarts_total = 0, 0
    start = time.time()
    deadline = start + args.hours * 3600

    ent_coef = float(ck.get("ent_coef", args.ent)) if args.resume else args.ent
    evaluator = None
    while time.time() < deadline and (not args.max_updates or update < args.max_updates):
        if (run_dir / "STOP").exists():
            print("STOP file found: saving and shutting down", flush=True)
            break
        t0 = time.time()
        net.eval()
        step_of = [0] * W  # decisions worker w has taken this rollout
        sent: list = [None] * W  # the actions worker w is executing

        def store(name: str, t_t, e_t, v: torch.Tensor, into: dict | None = None) -> None:
            into = rb if into is None else into
            if name not in into:
                into[name] = torch.empty((T, B, *v.shape[1:]), dtype=v.dtype, device=device)
            into[name][t_t, e_t] = v

        def act(ws: list[int]) -> None:
            """One decision for every env of the workers in ws (each may be at a different step t)."""
            envs = np.concatenate([np.arange(w * P, (w + 1) * P) for w in ws])
            ts = np.repeat([step_of[w] for w in ws], P)
            e_t, t_t = torch.from_numpy(envs).to(device), torch.from_numpy(ts).to(device)
            starts = ts % L == 0
            if starts.any():  # first decision of a training chunk: keep the GRU state it starts from
                s_env = torch.from_numpy(envs[starts]).to(device)
                rb["h"][torch.from_numpy(ts[starts] // L).to(device), s_env] = h_state[s_env]
            x = batch([o for w in ws for o in cur_obs[w]])
            prev_t = torch.from_numpy(prev[envs]).to(device)
            allowed = torch.from_numpy((cd[envs] == 0).astype(np.float32)).to(device)
            choice, logp, value, h_new = decide(x, prev_t, h_state[e_t], allowed)
            h_state[e_t] = h_new
            actions = decode_action(choice.cpu().numpy())
            workers.send_step(ws, actions)
            for k, w in enumerate(ws):
                sent[w] = actions[k * P : (k + 1) * P]
            for key, v in x.items():
                store(key, t_t, e_t, v, rb["obs"])
            for name, v in (("prev", prev_t), ("allowed", allowed), ("choice", choice), ("logp", logp), ("value", value)):
                store(name, t_t, e_t, v)

        def collect(w: int) -> int:
            results, restarts = workers.recv_step([w])
            lo, hi, t = w * P, (w + 1) * P, step_of[w]
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
            rew_np[t, lo:hi] = rewards
            done_np[t, lo:hi] = dones
            actions = sent[w]
            cd[lo:hi] = np.where(dones > 0, 0, np.where(actions[:, 7] == 1, USE_COOLDOWN, np.maximum(cd[lo:hi] - 1, 0)))
            prev[lo:hi] = actions.astype(np.float32) * (1.0 - dones)[:, None]
            if dones.any():
                h_state[lo:hi] *= torch.from_numpy(1.0 - dones).to(device).unsqueeze(-1)
            cur_obs[w] = [r[0] for r in results]
            step_of[w] = t + 1
            return restarts

        # Asynchronous rollout: whenever any workers have replied, batch all of them into one forward pass
        # and send their next actions at once, so no worker waits for a slower one between decisions.
        # Every env still takes exactly T decisions with the same frozen policy, as in a lockstep rollout.
        with torch.no_grad():
            ready, flying = list(range(W)), {}
            while ready or flying:
                if ready:
                    act(ready)
                    for w in ready:
                        flying[workers.remotes[w]] = w
                    ready = []
                for conn in wait(list(flying)):
                    w = flying.pop(conn)
                    restarts_total += collect(w)
                    if step_of[w] < T:
                        ready.append(w)
            x = batch([o for w in range(W) for o in cur_obs[w]])
            allowed = torch.from_numpy((cd == 0).astype(np.float32)).to(device)
            _, last_value, _ = net(x, torch.from_numpy(prev).to(device), h_state, allowed)

        h_chunks = rb["h"]
        rewards = torch.from_numpy(rew_np).to(device) / ret_norm.std
        values = rb["value"]
        dones = torch.from_numpy(done_np).to(device)
        adv, returns = gae(rewards, values, dones, last_value, args.gamma, args.lam)
        rollout_time = time.time() - t0

        obs_b = rb["obs"]
        prev_b, allowed_b = rb["prev"], rb["allowed"]
        choice_b, logp_b = rb["choice"], rb["logp"]
        net.train()
        seqs = [(c, b) for c in range(n_chunks) for b in range(B)]
        mb_size = max(1, len(seqs) // args.minibatches)
        stats = {k: [] for k in ("pg", "vf", "ent", "kl", "clipfrac", "grad")}
        bad_update = False
        before = {"model": copy.deepcopy(net.state_dict()), "optimizer": copy.deepcopy(opt.state_dict())}
        for _ in range(args.epochs):
            order = shuffle_rng.permutation(len(seqs))
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
            net.load_state_dict(before["model"])
            opt.load_state_dict(before["optimizer"])
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
                if rate > args.raise_at and difficulty[tier - 1] < 1.0:
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
        for tier in range(1, TIER_COUNT + 2):
            hist = list(tier_hist[tier])
            if tier <= TIER_COUNT and weights[tier - 1] > 0:
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
                (f"t{t}:exit={np.mean([s['exit'] for s in tier_hist[t]]):.2f}/d={difficulty[t-1]:.1f}" if t <= TIER_COUNT
                 else f"freedoom:exit={np.mean([s['exit'] for s in tier_hist[t]]):.2f}/areas={np.mean([s['sectors'] for s in tier_hist[t]]):.0f}")
                for t in list(args.tiers) + ([TIER_COUNT + 1] if args.freedoom_share > 0 else []) if tier_hist[t]
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
                "ent_coef": ent_coef,
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
                 "--per-tier", str(args.eval_per_tier), "--procs", str(args.eval_procs), "--out", str(out)]
                + (["--freedoom-maps", "21", "32"] if args.eval_freedoom else []),
                cwd=str(REPO), env=dict(os.environ, PYTHONPATH=str(REPO / "training")),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if evaluator is not None and evaluator[2].poll() is not None:
            ev_update, ev_out, _ = evaluator
            evaluator = None
            if ev_out.exists():
                summary = json.loads(ev_out.read_text())["summary"]
                w.add_scalar("eval/score", summary["score"], ev_update)
                if args.shipped_score:
                    w.add_scalar("ref/shipped_score", args.shipped_score, ev_update)
                for tier in range(1, TIER_COUNT + 1):
                    if f"tier{tier}" in summary:
                        w.add_scalar(f"eval/tier{tier}_exit", summary[f"tier{tier}"]["exit"], ev_update)
                if "freedoom2" in summary:
                    for key in ("exit", "dead", "kills", "sectors", "doors_opened"):
                        w.add_scalar(f"eval/freedoom_{key}", summary["freedoom2"][key], ev_update)
                print(f"validation score at update {ev_update}: {summary['score']}", flush=True)

    ck = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "update": update,
        "steps": total_steps,
        "difficulty": list(difficulty[:]),
        "ret_norm": (ret_norm.mean, ret_norm.var, ret_norm.count),
        "ent_coef": ent_coef,
        "args": vars(args),
    }
    save_atomic(ck, run_dir / "latest.pt")
    save_atomic(net.state_dict(), run_dir / "model.pt")
    workers.close()
    writer.close()
    print(f"done: {update} updates, {total_steps} decisions in {(time.time()-start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
