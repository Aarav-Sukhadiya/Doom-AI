# Doom-AI: an agent that learns to play DOOM levels it has never seen

**Yugaantar 2026 · DevForge · Track 1: "Can It Play DOOM?"**

## 1. Mission

For decades, *"Can it run DOOM?"* was the classic test for hardware. This hackathon flips the question:
**can a neural network actually play DOOM?**

Most game bots break the moment a wall moves or an enemy appears somewhere new, because they memorise a fixed
sequence of button presses. Our goal is an agent that plays **levels it has never seen before**. It should survive
ambushes, find its way through rooms connected by doors, pick up keycards, remember which door each key opens,
and reach the exit.

Nobody gives the agent a dataset or a map. It learns only by playing, again and again, and finding out what works.

## 2. What makes our agent different (our USP)

**It remembers places, not button sequences.**

A person playing DOOM for the first time remembers things like *"there was a red locked door back near the start"*.
Once they find the red key, they head back to that door. Our agent does the same. It has a small **landmark memory**:

- When it **sees** something important, such as a locked door, a keycard or the exit, it writes a note into a memory slot.
- Each note records **where the thing was**. The agent works out its own position from its movements, the way you
  can count your steps in a dark room, because the game never tells it its coordinates.
- Later, when it picks up a key, it can **look up** where the matching door was and head back there.

The memory is part of the neural network and is learned, not hand-written rules. That matters because the
competition only allows decisions made by trained networks.

## 3. How it works, in plain words

| Part | What it does | Everyday comparison |
| --- | --- | --- |
| **Senses** | Each step the agent gets 64 "feelers" around it that report what they touch (wall, door, locked door, enemy, item), a list of visible enemies and items, its health, ammo and keys, and a small screen image. | Like walking with your eyes open and your arms out |
| **Memory** | A short-term memory carries what just happened. The landmark memory keeps the important places it has seen. | Remembering *"the red door is behind me, to the left"* |
| **Brain** | A compact neural network turns the senses and the memory into button presses: move, turn, shoot, open doors. | Deciding what to do next |
| **Learning** | The agent plays thousands of practice levels. Exploring new rooms, collecting keys, opening locked doors and reaching the exit earn points; dying loses points. Over time it keeps what earns points. | Practising a game until you get good at it |
| **Practice levels** | We built our own level maker. It generates endless new layouts, from easy (one room, walk to the exit) to hard (several rooms, locked doors, keys hidden down side branches, enemies). | A coach setting harder drills each week |

Because the agent practises on thousands of different generated levels, it can't memorise any one of them. It has to
learn skills that carry over to levels it has never seen, and the competition tests exactly that.

## 4. Progress

| Step | Status |
| --- | --- |
| Project plan and test plan | ✅ Done |
| Set-up checks (official tests, submission validator, GPU) | ✅ All passing |
| Level maker: turns simple text drawings into real DOOM levels, matches the organisers' test levels exactly, supports doors, coloured locked doors, keys, enemies, items and both kinds of exit | ✅ Done |
| Step counting: the agent can estimate where it is from its own movements (about 1% error after 500 steps) | ✅ Done |
| Speed check: the game runs about 16,000 agent decisions per second on our laptop | ✅ Done |
| Random level generator for all four tiers (combat arenas, multi-room mazes with doors, keys behind locked doors, two-lock gauntlets). A scripted test walker finished 48 out of 48 generated levels in the real game | ✅ Done |
| Scoring for practice: points for exploring, keys, doors, kills and the exit; penalties for damage and dying. Each event is tested to count exactly once | ✅ Done |
| Neural network, with its senses, short-term memory and landmark memory | ✅ Built (decides in under 2 ms per move) |
| Learning loop, with difficulty that rises by itself as the agent improves | ✅ Running at about 2,300 decisions per second |
| Check that practice and evaluation behave identically (same moves from the same situation) | ✅ Passing |
| Training, testing on unseen levels, final agent | 🔄 Training now, overnight run next |

## 5. Submission

| Required | Where |
| --- | --- |
| Public GitHub repository with the full source code, trained weights and the track entry point | This repository. The agent the organisers run is `submission/agent.py`. |
| 2-page project brief (PDF): USP, system architecture, technical approach | Added to this repository before the deadline |

## 6. Playing by the rules

- Every move the agent makes comes from its **trained neural network**. There are no hand-written "if you see X,
  do Y" rules, no path-finding algorithms and no auto-aim.
- During **practice** we may change anything (custom levels, extra hints for scoring). During **evaluation** the
  agent sees only what any player sees, through the organisers' official setup, on levels it has never seen.
- All of the agent's design and training code was written during the hackathon and lives in this repository.

## 7. Where things are

```
submission/agent.py     the agent the organisers run
submission/weights/     the trained network (stored with Git LFS)
training/               everything used to build and train it (level maker, step counting, learning loop)
maps/                   the organisers' small test level
env/, docs/, tests/     the organisers' official game wrapper, rules and tests (unchanged)
validate.py             the organisers' pre-submission checker
```

## 8. For developers

```bash
uv sync --group train          # Python 3.13; installs ViZDoom, the official wrapper, torch and tensorboard
uv run pytest                  # official tests plus ours
uv run python validate.py      # the organisers' submission checks
```

The official rules, scoring and the technical contract are in [docs/](docs/). This repository was created from the
organisers' starter template.
