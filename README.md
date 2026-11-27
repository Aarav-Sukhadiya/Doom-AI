# Doom-AI: an agent that learns to play DOOM levels it has never seen

**Yugaantar 2026, DevForge, Track 1: "Can It Play DOOM?"**

**Team Screen Breakers:** Aarav Sukhadiya (team leader) and Adit Krishnadas

## 1. Mission

For decades, "Can it run DOOM?" was the classic test for hardware. This hackathon flips the question: can a
neural network actually play DOOM?

Most game bots fall apart the moment a wall moves or an enemy shows up somewhere new, because they repeat a fixed
sequence of button presses. Our goal is an agent that plays levels it has never seen. It should survive fights,
find its way through rooms joined by doors, pick up keycards, remember which door each key opens, and reach the
exit.

Nobody gives the agent a dataset or a map. It learns only by playing, over and over, and finding out what works.

## 2. What makes our agent different

**It remembers places, not button sequences.**

A person playing a new DOOM level remembers things like "there was a red locked door back near the start". Once
they find the red key, they go back to that door. Our agent does the same, with three kinds of memory that it
builds as it plays:

- **Landmarks.** When it sees something important, such as a locked door, a keycard, a door, health, ammo, armour
  or a weapon, it notes where that thing was. Later, holding the key, it can head back to the right door.
- **Breadcrumbs.** It keeps count of where it has already walked, so it can tell old ground from new ground.
- **A map it draws itself.** Its 64 distance sensors mark open floor, walls and doors as it moves, and it reads
  that map like a small picture centred on itself.

The game never tells the agent where it is. It works out its own position from its movements, the way you can
count your steps in a dark room. Everything it decides comes from the trained network; the memories only give it
better information to decide with.

## 3. How it works, in plain words

| Part | What it does | Everyday comparison |
| --- | --- | --- |
| **Senses** | Each step the agent gets 64 feelers around it that report what they touch (wall, door, locked door, enemy, item), a list of what is in view (enemies, items, keys, incoming fireballs), its health, ammo and keys, and a small picture of the screen. | Walking with your eyes open and your arms out |
| **Memory** | A short-term memory of what just happened, plus the landmarks, breadcrumbs and self-drawn map above. | Remembering "the red door is behind me, to the left" |
| **Brain** | A compact neural network turns all of that into button presses: move, strafe, turn, shoot, open doors and switch weapons. | Deciding what to do next |
| **Learning** | The agent plays millions of practice moves. Reaching the exit earns the most; keys, doors, new rooms, kills and pickups earn smaller amounts; taking damage and dying cost points. Over time it keeps what earns points. | Practising a game until you get good at it |
| **Practice levels** | We built a level maker that creates endless new layouts in four tiers, from a single combat room to multi-room levels with two locked doors. The agent also practises on 56 real Freedoom levels. | A coach setting harder drills each week |

Because it practises on so many different levels, it cannot memorise any one of them. It has to learn skills that
carry over to new levels, and that is exactly what the competition tests.

## 4. Where it stands now

The agent in this repository was trained overnight on a laptop, about 115 million decisions in all. On levels
it has never seen:

| Test | Reaches the exit | Dies |
| --- | --- | --- |
| The organisers' test level (`maps/two_rooms.wad`) | Every time | Never |
| Combat rooms (tier 1) | About half the time | 12 to 25% of the time |
| Multi-room levels with doors (tier 2) | 68 to 80% | Never |
| Keycard levels (tier 3) | 72 to 84% | Never |
| Two-lock gauntlets (tier 4) | About a third of the time | About half the time |
| Real Freedoom levels it never practised on | Not yet | Most of the time |

Each range covers three separate sets of unseen levels, the largest with 100 levels per tier. Real Freedoom maps are far larger than anything it
practised on; it explores and fights there but does not yet find the exit.

It has finished each of our three hardest test levels at least once, collecting both keys and opening both
locked doors on the way. Over the night its score on our copy of the official test rose from about 40 to 64.

What it still finds hard: several enemies at once, finding an exit that is far from everything else, and very
large real maps.

## 5. How we check it

- **Validation levels.** Every 10 to 15 minutes the newest version plays 80 levels it has never trained on. A new
  version only replaces the one in this repository if it clearly beats it, and only after the organisers'
  checker passes.
- **A copy of the official test.** We built a test that follows the organisers' rules as closely as we can: four
  tiers of locked levels that are never used for training, the same time limits per map and the same compute
  limits per decision. Points follow the four scoring categories (exit, progression, exploration, combat).
- **Videos.** We record the agent's best runs on the hardest levels, and we can watch the current agent play at
  normal speed, level after level.

## 6. Submission

| Required | Where |
| --- | --- |
| Public GitHub repository with the source code, trained weights and the track entry point | This repository. The agent the organisers run is `submission/agent.py`, with its trained network in `submission/weights/` |
| 2-page project brief (PDF): USP, system architecture, technical approach | [brief/brief.pdf](brief/brief.pdf) |

## 7. Playing by the rules

- Every move comes from the trained neural network. There are no hand-written "if you see X, do Y" rules, no
  path-finding and no auto-aim.
- The only fixed checks are two simple masks taken from what the agent can see: it cannot fire when no enemy is
  in view, and it cannot press USE again for a moment after pressing it (pressing USE on a moving door would
  reverse it). The rules allow masks like these.
- During practice we can change anything: custom levels, extra reward hints, easier difficulty. During the real
  test the agent sees only what any player sees, through the organisers' official setup.
- All design and training code was written during the hackathon and is in this repository.

## 8. Where things are

```
submission/agent.py      the agent the organisers run
submission/weights/      the trained network (stored with Git LFS) and its settings
training/                everything used to build, train and test it (see training/README.md)
maps/                    the organisers' small test level
env/, docs/, tests/      the organisers' official game wrapper, rules and tests (unchanged)
validate.py              the organisers' pre-submission checker (unchanged)
```

## 9. For developers

```bash
uv sync --group train                                  # Python 3.13: ViZDoom, the official wrapper, PyTorch
uv run pytest                                          # the organisers' tests plus ours
uv run python validate.py                              # the organisers' submission checks
PYTHONPATH=training uv run python -m doomai.live       # watch the current agent play, level after level
```

How training works and how to run it is described in [training/README.md](training/README.md). The official
rules, scoring and technical contract are in [docs/](docs/). This repository was created from the organisers'
starter template.
