# Doom-AI architecture

A detailed reference for people who want to understand, check or extend the project. For a plain overview read
`README.md`; for the two-page summary read `brief/brief.pdf`; for AI coding agents read `AGENTS.md`.

## 1. The task

The organisers run our `submission/agent.py` (class `DoomAgent`) on unseen DOOM maps in four tiers: combat and
survival, multi-room exploration with doors, keycard progression, and a full gauntlet. Each map ends at the exit,
on death or at a tick cap. Points come from four categories: exit (largest, plus a bonus for health and time left),
progression (keys, doors, locked doors), exploration (unique areas) and combat (kills, minus self-damage and early
death). Harder tiers weigh more. The official rules, contract and limits are in `docs/` and `env/` (unchanged
template). Limits per map: 40 ms of agent compute per decision on average, 10 s for the first `act()`, 5 s for any
other, 8 GB RAM, 2 GB VRAM, 1 CPU.

## 2. What the agent sees and does

Each decision covers 4 game tics. The official observation (`docs/CONTRACT.md`) is turned into network inputs in
`submission/doomobs.py` (`preprocess`, `observe`):

| Input | Shape | Source |
| --- | --- | --- |
| Image | 4 x 60 x 80, uint8 | RGB screen and depth buffer, downsampled 2x |
| Rays | 7 x 64 | distance plus one-hot hit type (none, wall, door, locked door, enemy, item) for 64 rays round the player |
| Objects | 32 x 5 + type | visible entities: distance, direction (sin, cos), relative velocity; 12 coarse types |
| Vitals | 17 | health, armour, weapon, ammo, six key flags, velocity |
| Previous action | 9 | the last buttons pressed |
| Landmarks | 8 x 7 | see section 3 |
| Breadcrumbs | 26 | see section 3 |
| Level map | 4 x 32 x 32, uint8 | see section 3 |
| Room around me | 17 | nearest hit and mean distance in 8 slices of the rays, plus straight ahead |
| `attack_ok` | 1 | an enemy is in view (used only as a mask) |

Actions are six categorical heads decoded into the 9 official buttons (`decode_action`): move (none, forward,
back), strafe (none, left, right), turn (none, left, right), fire (off, on), USE (off, on) and next weapon (off, on).
Two masks are built from observation fields, which the rules allow: fire is masked off when no enemy is in view,
and USE is masked for 8 decisions after a press because pressing USE on a moving door reverses it.

## 3. Memories (`submission/landmarks.py`)

The game never gives the agent its position. `Odometry` dead-reckons heading from the turn buttons (DOOM's turn
table, including that rotation starts on tic 8 after reset and slows for the first held tics) and position from
the observed player-frame velocity. About 150 units of drift after 400 decisions on a real map.

- **Landmarks** (`LandmarkTracker`): 8 typed slots (start, key, locked door, door, health, ammo, weapon, armour).
  When rays or objects show one, its position is stored in the start frame; every decision each slot is reported
  relative to the agent (valid, forward and left offset, distance, bearing, age). Objects are kept up to 4096 units
  away (`entity_range` 4.0).
- **Breadcrumbs** (`crumbs()`): visit counts per 128-unit cell of the dead-reckoned path; the input is visits here
  and at 1, 2 and 4 cells in 8 directions round the heading, plus how much ground has been covered.
- **Level map** (`LevelMap`): every decision the 64 rays mark cells they crossed as open floor and the cell they
  stopped at as wall, door or locked door; the cell underfoot is marked walked. A 32 x 32 crop (64 units per cell)
  centred on the agent and turned to its heading is the input. Doors that open are cleared by later rays.

## 4. Network (`submission/doomnet.py`, `DoomNet`)

CNN on the image (3 convolutions, 256 out), circular 1D convolutions on the rays (128 out), an embedding plus MLP
on objects pooled by mean and max (128 out), an MLP on vitals and previous action (64 out), concatenated into a
256-unit trunk. Optional branches add into the trunk as residuals whose output layer starts at zero, so new inputs
can be added to a trained agent without changing its behaviour at first: breadcrumbs (`crumb_*`), room around me
(`space_*`), level map (`map_*`, small CNN) and landmarks (`slot_*`, attention with the trunk as query). A GRU
(256) carries memory between decisions. Heads: one linear layer for the five original action heads, a separate
zero-initialised layer for next weapon, and a value head used only in training. The submitted config enables every
branch (`submission/weights/config.json`). Inference takes about 2 ms per decision on one CPU thread.

## 5. Training (`training/doomai`)

- **Algorithm** (`ppo.py`): recurrent PPO with GAE. 14 worker processes x 4 games each, rollouts of 128 decisions,
  sequences of 32 for back-propagation through the GRU, 3 epochs, 4 minibatches, clip 0.2, value loss 0.5,
  entropy 0.005, gamma 0.998, lambda 0.95, gradient clip 1.0. Returns are scaled by a running standard deviation
  of discounted returns. Workers (`worker.py`) import no PyTorch so they stay small; `train.py` is the launcher.
  The learning rate was 2e-4, then 1e-4 and 5e-5 for the last hours (a resumed run uses the `--lr` given).
- **Levels** (`maps/`): `grid.py` compiles ASCII layouts into UDMF maps with doors, coloured locked doors, keys,
  enemies, items, weapons and both exit kinds. `generate.py` makes tiers 1 to 4 at a difficulty from 0 to 1,
  checks every level is solvable with a key-aware search, and splits seeds 80/10/10 into training, validation and
  locked test levels. The agent also practises on Freedoom 2 MAP01-20 and Freedoom 1 E1M1-E4M9 (15% of games,
  sometimes at easier skill); Freedoom 2 MAP21-32 are only ever used for testing.
- **Curriculum**: per-tier difficulty rises by 0.1 when the recent exit rate passes 0.45 and falls by 0.05 below
  0.15; 35% of games are forced to full difficulty. `focus.py` shifts the tier mix from spec-test results (the
  submitted agent was trained with weights 1.25 / 1 / 1 / 1.25 for tiers 1 to 4).
- **Rewards** (`envs/shaped.py`, training only):

| Event | Reward |
| --- | --- |
| Exit | 25, plus up to 2 for health and 5 for time left |
| Key / door / locked door | 3 / 1 / 3 |
| New area / new 64-unit cell | 0.2 / 0.02 (decays 0.998 per cell; x3 once only the exit is left) |
| Kill / damage dealt | 1 / 0.005 per point |
| Pickups | health 0.02 per point gained, armour 0.005, ammo 0.01 per round, new weapon 1.5 |
| Dodging | up to 0.02 per decision moving sideways while a fireball closes in |
| Progress hint | 0.05 per cell closer to the next key, its door, then the exit (x2 on the last stretch) |
| Damage taken | -0.01 per point |
| Death | -2, minus up to 3 more the earlier it happens |
| Wasted ammo / firing at nothing | -0.02 per round / -0.02 |
| USE on a locked door without its key | -0.1 |
| Fighting while boxed in | -0.003 per decision |
| Time | -0.001 per decision |

Privileged game state (position, door locks, sector heights) is used only to compute these rewards.

## 6. Testing and shipping

- **Validation** (`evaluate.py`): every 200 updates a snapshot plays 20 validation levels per tier plus
  `two_rooms.wad` through the real `DoomAgent`, scored with a simple proxy.
- **Spec test** (`spec_test.py`): our copy of the official evaluation. 25 locked test levels per tier (100 for
  the final pick), the official tick caps and compute limits, points in the four official categories with tier
  weights 1 to 4 (point values are our estimate). Freedoom 2 MAP21-32 are reported beside it.
- **Auto-ship** (`autoship.py`): a snapshot replaces `submission/weights` only if it beats the shipped validation
  score by 0.3, is within 0.5 of the shipped spec score, and `validate.py` passes; then it is committed and pushed.
- **Final pick** (`final_pick.py`): the leading snapshots on 100 locked levels per tier. Main20 update 4200 won
  with 63.7 points and was submitted.
- **Tests** (`training/tests`, 130 with the template's): the level compiler and generator, every reward, the rules
  checks (no banned imports, limits, strict weight loading) and that the trainer and the submitted agent produce the
  same outputs and choices for the same game.
- **Tools**: `watch.py` and `live.py` show the agent playing; `best_attempts.py` keeps the best runs on the hardest
  levels with video and a written review; `dashboard_data.py` feeds the progress dashboard.

## 7. Results of the submitted agent

On held-out levels (validation, 25 and 100 locked test levels per tier): tier 1 exits 50-55% with 12-25% deaths,
tier 2 68-80%, tier 3 72-84%, tier 4 32-36% with about half dying, `two_rooms.wad` every time. Real Freedoom maps
are not yet finished. About 115 million training decisions in total; the last run (main20) trained about 44 million.

## 8. Decisions and lessons

- The exit reward went from 10 to 25 after the exit was found to be only 26% of all reward; this was the largest
  single gain.
- Penalties for standing still in fights made the policy steadily more random and lowered the score; they were
  removed and a positive dodge reward used instead.
- Cutting the keycard tier's practice share made it slip unnoticed by validation; the spec test caught it, and
  shipping now needs both tests.
- The agent could not change weapons until a next-weapon head was added; Doom also switches automatically on pickup.
- Nothing in the observation marks the exit, so finding it relies on the image and on exploration; this is the
  main remaining weakness, with fights against several enemies and very large real maps.
