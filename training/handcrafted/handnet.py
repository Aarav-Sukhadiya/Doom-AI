"""HandNet: a Doom-playing neural network whose weights are all written by hand. Nothing is trained.

EXPERIMENT ONLY. docs/RULES.md section 1 requires the submitted agent to act through *trained*
parameters. This network is hand-designed logic in network form, so it must never be copied into
submission/. It exists only to compare against the trained DoomNet.

The network is ordinary layers: 1x1 convolutions and a pixel-to-column dense layer for vision, then a
stack of dense layers y = act(W x + b) for decisions. Every weight and bias is set by the builder below
from a written rule. For example the rule

    door_close[i] = clamp01(hit_door[i] + near[i] - 1)

puts weight 1 on hit_door[i], weight 1 on near[i] and bias -1 in that neuron's row; every other weight
in the row is 0. Activations are relu, clamp01 (hardtanh from 0 to 1) and identity. The dense layers
are densely connected (each layer reads every earlier neuron, as in DenseNet), and a memory vector is fed
back each step, which makes the whole thing a recurrent network.

What the rules do, in order of priority:
  1. combat   an enemy is visible: turn onto the nearest one, fire when it is centred, strafe side to
              side, back away from close melee enemies
  2. escape   pushing forward without moving for several steps: back off and turn
  3. search   hurt with no enemy in view: turn around to find the attacker
  4. key      a key is visible: walk to it
  5. exit     the red exit pad is visible on the floor: walk onto it
  6. item     a weapon, or health/ammo when low, is visible: walk to it
  7. explore  right-hand wall following over the 64 rays (doors count as open, locked doors count as
              open while holding a key unless the agent already failed to open it recently)
  USE is pressed at doors, at locked doors while holding a key, at a green exit switch, and when stuck,
  with a 12-step cooldown so a door is not bounced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submission"))
from doomobs import observe  # noqa: E402
from landmarks import LandmarkTracker  # noqa: E402

LIN, RELU, CLAMP = 0, 1, 2
NUM_RAYS = 64
RANGE = 1024.0
THETA = [i * 360.0 / NUM_RAYS if i <= NUM_RAYS // 2 else i * 360.0 / NUM_RAYS - 360.0 for i in range(NUM_RAYS)]
FRONT = (62, 63, 0, 1, 2)  # doors and locks within ~11 deg of straight ahead
AHEAD = (63, 0, 1)  # walls within ~6 deg (wider rays would hit the wall being followed)
R60, R90, R112, R120, R135 = 53, 48, 44, 43, 40  # rays about 60, 90, 112, 118 and 135 degrees to the right
WALL_RAYS = FRONT + (R60, R90, R112, R120, R135)
KEEP_NEAR, KEEP_FAR = 40.0, 80.0  # strafe to stay this far from the right wall
FRONT_STOP = 96.0  # a wall closer than this straight ahead blocks the way
MEMORY = ("use", "lock", "hp", "search", "stuck", "esc", "osc", "sw", "x", "tl", "tr")
H, W = 60, 80  # half-resolution image from doomobs.preprocess
BINS = 8


class Builder:
    """Collects hand-written neurons layer by layer and turns them into dense weight matrices."""

    def __init__(self):
        self.index: dict[tuple[str, int], int] = {}
        self.size = 0
        self.layers: list[list[tuple[tuple[str, int], dict[int, float], float, int]]] = []
        self.current: list | None = None

    def input(self, name: str, n: int) -> None:
        for i in range(n):
            self.index[(name, i)] = self.size
            self.size += 1

    def __call__(self, name: str, i: int = 0) -> int:
        return self.index[(name, i)]

    def layer(self) -> None:
        if self.current:
            self._close()
        self.current = []

    def neuron(self, name: str, i: int, terms: dict[int, float], bias: float = 0.0, act: int = LIN) -> None:
        self.current.append(((name, i), terms, bias, act))

    def _close(self) -> None:
        start = self.size
        for k, (key, _, _, _) in enumerate(self.current):
            self.index[key] = start + k
        self.layers.append((start, self.current))
        self.size += len(self.current)
        self.current = None

    def build(self) -> tuple[nn.ModuleList, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self.current:
            self._close()
        mods, acts = nn.ModuleList(), []
        for start, rows in self.layers:
            lin = nn.Linear(start, len(rows))
            with torch.no_grad():
                lin.weight.zero_()
                lin.bias.zero_()
                for r, (_, terms, bias, _) in enumerate(rows):
                    for col, w in terms.items():
                        lin.weight[r, col] += w
                    lin.bias[r] = bias
            lin.requires_grad_(False)
            mods.append(lin)
            act = torch.tensor([a for _, _, _, a in rows])
            acts.append((act == RELU, act == CLAMP))
        return mods, acts


def _vision() -> tuple[nn.Module, nn.Module, nn.Module, nn.Linear]:
    """Hand-set layers that find the red exit pad and the green exit switch, then count them per column bin."""
    red = nn.Conv2d(3, 1, 1)  # redness = clamp01((r - 3g - 3b - 50) / 10)
    green1 = nn.Conv2d(3, 3, 1)  # a = g - r - 25, b = g - b - 35, relu(a - b)
    green2 = nn.Conv2d(3, 1, 1)  # greenness = clamp01(min(a, b) / 5) = clamp01((a - relu(a - b)) / 5)
    bins = nn.Linear(2 * H * W, 2 * BINS)  # per column bin: count of red floor pixels, count of green pixels
    with torch.no_grad():
        red.weight[:] = torch.tensor([1.0, -3.0, -3.0]).view(1, 3, 1, 1) / 10.0
        red.bias[:] = -5.0
        green1.weight[:] = torch.tensor([[-1.0, 1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]]).view(3, 3, 1, 1)
        green1.bias[:] = torch.tensor([-25.0, -35.0, 10.0])
        green2.weight[:] = torch.tensor([1.0, 0.0, -1.0]).view(1, 3, 1, 1) / 5.0
        green2.bias[:] = 0.0
        bins.weight.zero_()
        bins.bias.zero_()
        for row in range(H):
            for col in range(W):
                gun = row >= 36 and 28 <= col < 52  # the weapon sprite sits bottom-centre
                b = col * BINS // W
                if row >= H // 2 + 1 and not gun:  # the pad is on the floor, below the horizon
                    bins.weight[b, row * W + col] = 1.0
                if not gun:
                    bins.weight[BINS + b, H * W + row * W + col] = 1.0
    for m in (red, green1, green2, bins):
        m.requires_grad_(False)
    return red, green1, green2, bins


def _and(b: Builder, *refs: int) -> dict[int, float]:
    return {r: 1.0 for r in refs}


def build_policy() -> Builder:
    b = Builder()
    b.input("d", NUM_RAYS)
    for t in range(6):
        b.input(f"hit{t}", NUM_RAYS)
    for k in ("esin", "ecos", "edist", "emask"):
        b.input(k, 32)
    for t in range(12):
        b.input(f"et{t}", 32)
    b.input("v", 17)
    b.input("pa", 9)
    b.input("m", len(MEMORY))
    b.input("red", BINS)
    b.input("grn", BINS)
    M = {k: b("m", i) for i, k in enumerate(MEMORY)}

    # ---- layer 1: per-ray, per-entity and scalar facts
    b.layer()
    for i in range(NUM_RAYS):
        b.neuron("near", i, {b("d", i): -RANGE / 8}, 96 / 8, CLAMP)  # something within 96 units
    for i in range(32):
        b.neuron("en", i, {b("emask", i): 1, b("et1", i): 1, b("et2", i): 1, b("et3", i): 1}, -1, CLAMP)
        b.neuron("key", i, {b("emask", i): 1, b("et9", i): 1}, -1, CLAMP)
        b.neuron("weapon", i, {b("emask", i): 1, b("et8", i): 1}, -1, CLAMP)
        b.neuron("health", i, {b("emask", i): 1, b("et5", i): 1}, -1, CLAMP)
        b.neuron("ammo", i, {b("emask", i): 1, b("et7", i): 1}, -1, CLAMP)
    b.neuron("hp_low", 0, {b("v", 0): -20}, 0.5 * 20, CLAMP)  # health < 50
    b.neuron("ammo_low", 0, {b("v", 3): -50 * 2}, 10 * 2 / 1, CLAMP)  # selected ammo < 10 (vitals[3] is /50)
    b.neuron("anykey", 0, {b("v", k): 1 for k in range(8, 14)}, 0, CLAMP)
    b.neuron("use_ok", 0, {M["use"]: -20}, 1, CLAMP)
    b.neuron("lock_bad", 0, {M["lock"]: 20}, -10, CLAMP)
    b.neuron("sw_bad", 0, {M["sw"]: 20}, -10, CLAMP)
    b.neuron("x_bad", 0, {M["x"]: 20}, -10, CLAMP)
    b.neuron("search_on", 0, {M["search"]: 20}, -1, CLAMP)
    b.neuron("tl_on", 0, {M["tl"]: 20}, -1, CLAMP)
    b.neuron("tr_on", 0, {M["tr"]: 20}, -1, CLAMP)
    b.neuron("esc_on", 0, {M["esc"]: 20}, -1, CLAMP)
    b.neuron("esc_go", 0, {M["stuck"]: 20}, -12, CLAMP)  # stuck counter past 0.6
    b.neuron("osc_wrap", 0, {M["osc"]: 100}, -99, CLAMP)
    b.neuron("osc_left", 0, {M["osc"]: 100}, -50, CLAMP)
    b.neuron("hurt", 0, {M["hp"]: 100, b("v", 0): -100}, -1, CLAMP)  # lost 2+ health since last step
    b.neuron("vpos", 0, {b("v", 14): 1}, 0, RELU)
    b.neuron("vneg", 0, {b("v", 14): -1}, 0, RELU)
    b.neuron("red_tot", 0, {b("red", k): 1 for k in range(BINS)})
    b.neuron("red_l", 0, {b("red", k): 1 for k in range(3)})
    b.neuron("red_c", 0, {b("red", k): 1 for k in (3, 4)})
    b.neuron("red_r", 0, {b("red", k): 1 for k in range(5, 8)})
    b.neuron("grn_c", 0, {b("grn", k): 1 for k in (3, 4)})
    b.neuron("grn_l", 0, {b("grn", k): 1 for k in range(3)})
    b.neuron("grn_r", 0, {b("grn", k): 1 for k in range(5, 8)})

    # ---- layer 2: pick the nearest of each kind (entities arrive sorted nearest first), combine facts
    b.layer()
    for kind in ("en", "key"):
        for i in range(32):
            terms = {b(kind, i): 1.0} | {b(kind, j): -1.0 for j in range(i)}
            b.neuron(f"{kind}1", i, terms, 0, CLAMP)
    for i in range(32):
        b.neuron("hw", i, _and(b, b("health", i), b("hp_low")), -1, CLAMP)
        b.neuron("aw", i, _and(b, b("ammo", i), b("ammo_low")), -1, CLAMP)
    for i in FRONT:
        b.neuron("door_n", i, _and(b, b("hit2", i), b("near", i)), -1, CLAMP)
        b.neuron("lock_n", i, _and(b, b("hit3", i), b("near", i)), -1, CLAMP)
    b.neuron("lockok", 0, {b("anykey"): 1, b("lock_bad"): -1}, 0, CLAMP)
    b.neuron("E", 0, {b("en", i): 1 for i in range(32)}, 0, CLAMP)
    b.neuron("Kvis", 0, {b("key", i): 1 for i in range(32)}, 0, CLAMP)
    b.neuron("slow", 0, {b("vpos"): -40, b("vneg"): -40}, 0.05 * 40, CLAMP)  # under ~0.8 units/tic
    b.neuron("exit_vis", 0, {b("red_tot"): 0.5, b("x_bad"): -10}, -2, CLAMP)  # 6+ red floor pixels
    b.neuron("exit_l", 0, {b("red_l"): 1, b("red_c"): -1, b("red_r"): -1}, 0, CLAMP)
    b.neuron("exit_r", 0, {b("red_r"): 1, b("red_c"): -1, b("red_l"): -1}, 0, CLAMP)
    b.neuron("sw_l", 0, {b("grn_l"): 1, b("grn_c"): -1, b("grn_r"): -1}, 0, CLAMP)
    b.neuron("sw_r", 0, {b("grn_r"): 1, b("grn_c"): -1, b("grn_l"): -1}, 0, CLAMP)
    b.neuron("sw_seen", 0, {b("grn_l"): 0.1, b("grn_c"): 0.1, b("grn_r"): 0.1}, -2, CLAMP)
    b.neuron("osc_next", 0, {M["osc"]: 1, b("osc_wrap"): -2}, 1 / 6, CLAMP)

    # ---- layer 3: passable rays, item wants, features of the chosen enemy / key
    b.layer()
    for i in range(NUM_RAYS):
        b.neuron("lockpass", i, _and(b, b("hit3", i), b("lockok")), -1, CLAMP)
    for i in range(32):
        b.neuron("want", i, {b("weapon", i): 1, b("hw", i): 1, b("aw", i): 1}, 0, CLAMP)
        for kind in ("en", "key"):
            sel = b(f"{kind}1", i)
            b.neuron(f"{kind}_sp", i, {b("esin", i): 1, sel: 1}, -1, RELU)
            b.neuron(f"{kind}_sn", i, {b("esin", i): -1, sel: 1}, -1, RELU)
            b.neuron(f"{kind}_cp", i, {b("ecos", i): 1, sel: 1}, -1, RELU)
            b.neuron(f"{kind}_d", i, {b("edist", i): 1, sel: 4}, -4, RELU)
        b.neuron("en_mel", i, _and(b, b("et1", i), b("en1", i)), -1, CLAMP)
    b.neuron("sw_vis", 0, {b("sw_seen"): 1, b("sw_bad"): -1}, 0, CLAMP)
    b.neuron("door_front", 0, {b("door_n", i): 1 for i in FRONT}, 0, CLAMP)
    b.neuron("lock_front", 0, {b("lock_n", i): 1 for i in FRONT}, 0, CLAMP)

    # ---- layer 4: rays that do not block walking (doors, pickups, openable locks); target sums
    b.layer()
    for i in WALL_RAYS:
        b.neuron("passable", i, {b("hit2", i): 1, b("hit5", i): 1, b("lockpass", i): 1}, 0, CLAMP)
    for i in range(32):
        b.neuron("want1", i, {b("want", i): 1} | {b("want", j): -1 for j in range(i)}, 0, CLAMP)
    for kind in ("en", "key"):
        b.neuron(f"{kind}_s", 0, {b(f"{kind}_sp", i): 1 for i in range(32)} | {b(f"{kind}_sn", i): -1 for i in range(32)})
        b.neuron(f"{kind}_c", 0, {b(f"{kind}_cp", i): 1 for i in range(32)})
        b.neuron(f"{kind}_dist", 0, {b(f"{kind}_d", i): 1 for i in range(32)})
    b.neuron("melee", 0, {b("en_mel", i): 1 for i in range(32)}, 0, CLAMP)
    b.neuron("Wvis", 0, {b("want", i): 1 for i in range(32)}, 0, CLAMP)
    b.neuron("sw_seek", 0, {b("sw_vis"): 1}, 0, CLAMP)
    b.neuron("stuck_c", 0, {b("pa", 0): 1, b("slow"): 1, b("door_front"): -1, b("E"): -1}, -1, CLAMP)

    # ---- layer 5: effective wall distance (passable rays read as nothing hit); item target; combat facts
    b.layer()
    for i in WALL_RAYS:
        b.neuron("eff", i, {b("d", i): 1, b("passable", i): 1}, 0, CLAMP)
    for i in range(32):
        sel = b("want1", i)
        b.neuron("w_sp", i, {b("esin", i): 1, sel: 1}, -1, RELU)
        b.neuron("w_sn", i, {b("esin", i): -1, sel: 1}, -1, RELU)
        b.neuron("w_cp", i, {b("ecos", i): 1, sel: 1}, -1, RELU)
    b.neuron("en_abs", 0, {b("en_s"): 1}, 0, RELU)
    b.neuron("en_absn", 0, {b("en_s"): -1}, 0, RELU)
    b.neuron("c_l", 0, {b("en_s"): 100}, -5, CLAMP)  # enemy more than ~3 deg left
    b.neuron("c_r", 0, {b("en_s"): -100}, -5, CLAMP)
    b.neuron("c_back", 0, {b("melee"): 10, b("en_dist"): -20}, 6 - 10, CLAMP)  # melee enemy within ~260 units
    b.neuron("k_l", 0, {b("key_s"): 100}, -8, CLAMP)
    b.neuron("k_r", 0, {b("key_s"): -100}, -8, CLAMP)
    b.neuron("k_f", 0, {b("key_c"): 20}, -10, CLAMP)  # key within ~60 deg of straight ahead
    b.neuron("hn", 0, {b("hurt"): 1, b("E"): -1}, 0, CLAMP)
    b.neuron("lock_use", 0, {b("lock_front"): 1, b("anykey"): 1}, -1, CLAMP)
    b.neuron("sw_use", 0, {b("sw_vis"): 1, b("near", 0): 1, b("sw_bad"): -1}, -1, CLAMP)

    def below(i, units, soft):  # clamp01((units - distance) / soft)
        return {b("eff", i): -RANGE / soft}, units / soft

    def above(i, units, soft):  # clamp01((distance - units) / soft)
        return {b("eff", i): RANGE / soft}, -units / soft

    # ---- layer 6: wall-following facts from single rays
    b.layer()
    for i in AHEAD:
        b.neuron("fb", i, *below(i, FRONT_STOP, 8), CLAMP)  # wall right ahead
    # the right wall ends just ahead: the 60 deg ray sees past it while the 120 deg ray still hits it. Doom
    # players slide ~80 units after letting go, so the turn starts here and finishes beside the opening.
    b.neuron("open_r", 0, *above(R60, 200, 16), CLAMP)
    b.neuron("wall_rbb", 0, *below(R120, 160, 16), CLAMP)
    b.neuron("far_r", 0, *above(R90, KEEP_FAR, 8), CLAMP)
    b.neuron("close_r", 0, *below(R90, KEEP_NEAR, 8), CLAMP)
    b.neuron("wall_r", 0, *below(R90, 250, 8), CLAMP)  # a wall on the right within 250
    b.neuron("wall_60", 0, *below(R60, 160, 8), CLAMP)
    b.neuron("wall_120", 0, *below(R120, 160, 8), CLAMP)
    # parallel to the wall: the rays 60 and 120 degrees right read the same distance
    b.neuron("conv", 0, {b("eff", R120): 0.88 * RANGE / 8, b("eff", R60): -RANGE / 8}, 0, CLAMP)  # heading into it
    b.neuron("div", 0, {b("eff", R60): RANGE / 8, b("eff", R120): -1.14 * RANGE / 8}, 0, CLAMP)  # heading away
    b.neuron("c_fire", 0, {b("en_abs"): -100, b("en_absn"): -100}, 8, CLAMP)  # centred within ~4.5 deg
    b.neuron("w_s", 0, {b("w_sp", i): 1 for i in range(32)} | {b("w_sn", i): -1 for i in range(32)})
    b.neuron("w_c", 0, {b("w_cp", i): 1 for i in range(32)})

    # ---- layer 7: wall-following situations; item steering
    b.layer()
    b.neuron("blocked", 0, {b("fb", i): 1 for i in AHEAD}, 0, CLAMP)
    b.neuron("opening", 0, {b("open_r"): 1, b("wall_rbb"): 1}, -1, CLAMP)
    b.neuron("al_l", 0, {b("conv"): 1, b("wall_120"): 1}, -1, CLAMP)
    b.neuron("al_r", 0, {b("div"): 1, b("wall_60"): 1, b("wall_120"): 1}, -2, CLAMP)
    b.neuron("s_r", 0, {b("far_r"): 1, b("wall_r"): 1}, -1, CLAMP)
    b.neuron("w_l", 0, {b("w_s"): 100}, -8, CLAMP)
    b.neuron("w_r", 0, {b("w_s"): -100}, -8, CLAMP)
    b.neuron("w_f", 0, {b("w_c"): 20}, -10, CLAMP)

    # ---- layer 8: follower rule by priority (timed turns first); one-hot mode gates in priority order
    b.layer()
    bl, op, tl, tr = b("blocked"), b("opening"), b("tl_on"), b("tr_on")
    b.neuron("tl_go", 0, {bl: 1, tl: -1, tr: -1}, 0, CLAMP)  # wall ahead: start a 90 deg left turn
    b.neuron("tr_go", 0, {op: 1, bl: -1, tl: -1, tr: -1}, 0, CLAMP)  # right wall ended: 90 deg right turn
    b.neuron("idle", 0, {bl: -1, op: -1, tl: -1, tr: -1}, 1, CLAMP)  # otherwise walk along the wall
    E, esc, srch, K, X = b("E"), b("esc_on"), b("search_on"), b("Kvis"), b("exit_vis")
    S, I = b("sw_seek"), b("Wvis")
    higher = []
    for gate, own in (("gE", E), ("gEsc", esc), ("gS", srch), ("gK", K), ("gX", X), ("gW", S), ("gI", I)):
        b.neuron(gate, 0, {own: 1} | {r: -1 for r in higher}, 0, CLAMP)
        higher.append(own)
    b.neuron("gF", 0, {r: -1 for r in higher}, 1, CLAMP)

    # ---- layer 9: follower buttons
    b.layer()
    idle = b("idle")
    b.neuron("f_l", 0, {tl: 2, b("tl_go"): 2, b("al_l"): 1, idle: 1}, -1, CLAMP)  # timed turn OR (align AND idle)
    b.neuron("f_r", 0, {tr: 2, b("tr_go"): 2, b("al_r"): 1, idle: 1}, -1, CLAMP)
    b.neuron("f_sl", 0, _and(b, b("close_r"), idle), -1, CLAMP)
    b.neuron("f_sr", 0, {b("s_r"): 1, idle: 1, b("close_r"): -1}, -1, CLAMP)
    b.neuron("f_f", 0, {idle: 1}, 0, CLAMP)

    # ---- layer 10: each mode's buttons, gated by its mode
    b.layer()
    pairs = {
        "L": [("gE", "c_l"), ("gK", "k_l"), ("gX", "exit_l"), ("gW", "sw_l"), ("gI", "w_l"), ("gF", "f_l")],
        "R": [("gE", "c_r"), ("gK", "k_r"), ("gX", "exit_r"), ("gW", "sw_r"), ("gI", "w_r"), ("gF", "f_r")],
        "F": [("gK", "k_f"), ("gI", "w_f"), ("gF", "f_f")],
        "B": [("gE", "c_back")],
        "A": [("gE", "c_fire")],
        "SL": [("gE", "osc_left"), ("gF", "f_sl")],
        "SR": [("gF", "f_sr")],
    }
    for out, items in pairs.items():
        for k, (gate, dec) in enumerate(items):
            b.neuron(f"{out}_part", k, _and(b, b(gate), b(dec)), -1, CLAMP)
    b.neuron("SR_part", 1, {b("gE"): 1, b("osc_left"): -1}, 0, CLAMP)
    b.neuron("use_want", 0, {b("door_front"): 1, b("lock_use"): 1, b("stuck_c"): 1, b("sw_use"): 1}, 0, CLAMP)

    # ---- layer 11: buttons
    b.layer()
    b.neuron("turn_l", 0, {b("L_part", k): 1 for k in range(6)} | {b("gS"): 1, b("gEsc"): 1}, 0, CLAMP)
    b.neuron("turn_r", 0, {b("R_part", k): 1 for k in range(6)}, 0, CLAMP)
    b.neuron("back", 0, {b("B_part", 0): 1, b("gEsc"): 1}, 0, CLAMP)
    b.neuron("forward", 0, {b("F_part", k): 1 for k in range(3)} | {b("gX"): 1, b("gW"): 1, b("B_part", 0): -1,
                                                                    b("gEsc"): -1}, 0, CLAMP)
    b.neuron("attack", 0, {b("A_part", 0): 1}, 0, CLAMP)
    b.neuron("strafe_l", 0, {b("SL_part", 0): 1, b("SL_part", 1): 1}, 0, CLAMP)
    b.neuron("strafe_r", 0, {b("SR_part", 0): 1, b("SR_part", 1): 1}, 0, CLAMP)
    b.neuron("use", 0, {b("use_want"): 1, b("use_ok"): 1}, -1, CLAMP)

    # ---- layer 12: the next memory
    b.layer()
    b.neuron("m_use", 0, {M["use"]: 1, b("use"): 1}, -1 / 12, CLAMP)
    b.neuron("m_lock", 0, {M["lock"]: 1, b("lock_front"): 0.04}, -0.004, CLAMP)
    b.neuron("m_hp", 0, {b("v", 0): 1})
    b.neuron("m_search", 0, {M["search"]: 1, b("hn"): 1}, -0.1, CLAMP)
    b.neuron("m_stuck", 0, {M["stuck"]: 1, b("stuck_c"): 0.2, b("esc_on"): -0.5}, -0.1, CLAMP)
    b.neuron("m_esc", 0, {M["esc"]: 1, b("esc_go"): 2}, -1 / 6, CLAMP)
    b.neuron("m_osc", 0, {b("osc_next"): 1})
    # chasing red for ~50 steps without exiting means it was blood, not the pad: ignore red ~250 steps
    b.neuron("m_x", 0, {M["x"]: 1, b("gX"): 0.012}, -0.002, CLAMP)
    # pressing USE at green that is not a switch builds frustration; past 0.5 green is ignored ~60 steps
    b.neuron("m_sw", 0, {M["sw"]: 1, b("sw_use"): 0.1}, -0.008, CLAMP)
    # a 90 degree turn takes 7 steps of holding the turn button (1 slow step, then fast)
    b.neuron("m_tl", 0, {M["tl"]: 1, b("tl_go"): 1}, -1 / 7, CLAMP)
    b.neuron("m_tr", 0, {M["tr"]: 1, b("tr_go"): 1}, -1 / 7, CLAMP)
    return b


class HandNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.red, self.green1, self.green2, self.bins = _vision()
        builder = build_policy()
        self.index = builder.index
        self.dense, acts = builder.build()
        for k, (relu, clamp) in enumerate(acts):
            self.register_buffer(f"relu{k}", relu)
            self.register_buffer(f"clamp{k}", clamp)
        self.out = {name: self.index[(name, 0)] for name in
                    ("forward", "back", "strafe_l", "strafe_r", "turn_l", "turn_r", "attack", "use")}
        self.mem_out = [self.index[(f"m_{k}", 0)] for k in MEMORY]

    def vision(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb.float()
        red = (self.red(x)).clamp(0, 1)
        g1 = self.green1(x)
        g1 = torch.cat([g1[:, :2], g1[:, 2:].relu()], dim=1)
        green = self.green2(g1).clamp(0, 1)
        return self.bins(torch.cat([red.flatten(1), green.flatten(1)], dim=1))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        for k, lin in enumerate(self.dense):
            z = lin(v)
            z = torch.where(getattr(self, f"relu{k}"), z.relu(), z)
            z = torch.where(getattr(self, f"clamp{k}"), z.clamp(0, 1), z)
            v = torch.cat([v, z], dim=1)
        return v


class HandAgent:
    """Same interface as submission DoomAgent, driven by HandNet."""

    def __init__(self, observation_space=None, action_space=None, device: str = "cpu"):
        torch.set_num_threads(1)
        self.net = HandNet().eval()
        self.tracker = LandmarkTracker()
        self.last = None
        self.reset()

    def reset(self) -> None:
        self.memory = np.zeros(len(MEMORY), np.float32)
        self.memory[MEMORY.index("hp")] = 1.0
        self.prev = np.zeros(9, np.float32)
        self.tracker.reset()
        self.first = True

    @torch.inference_mode()
    def act(self, obs: dict) -> np.ndarray:
        x = observe(obs, self.tracker, None if self.first else self.prev)
        self.first = False
        rays = x["rays"]
        et = np.zeros((12, 32), np.float32)
        et[x["ent_type"], np.arange(32)] = 1.0
        feats = x["ent_feat"]
        vec = np.concatenate([
            rays.reshape(-1), feats[:, 1], feats[:, 2], feats[:, 0], x["ent_mask"], et.reshape(-1),
            x["vitals"], self.prev, self.memory,
        ]).astype(np.float32)
        img = torch.from_numpy(x["image"][None, :3])
        v = torch.cat([torch.from_numpy(vec)[None], self.net.vision(img)], dim=1)
        out = self.net(v)[0]
        self.last = out
        o = {k: float(out[i]) > 0.5 for k, i in self.net.out.items()}
        self.memory = out[self.net.mem_out].numpy().astype(np.float32)
        a = np.zeros(9, np.int8)
        a[0] = o["forward"] and not o["back"]
        a[1] = o["back"]
        a[2] = o["strafe_l"]
        a[3] = o["strafe_r"] and not o["strafe_l"]
        a[4] = o["turn_l"]
        a[5] = o["turn_r"] and not o["turn_l"]
        a[6] = o["attack"]
        a[7] = o["use"]
        self.prev = a.astype(np.float32)
        return a

    def probe(self, *names) -> dict:
        return {n: float(self.last[self.net.index[(n, 0)]]) for n in names}


def count_parameters() -> int:
    return sum(p.numel() for p in HandNet().parameters())
