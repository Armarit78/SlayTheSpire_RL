# SlayTheSpire_RL

Research prototype for learning **Slay the Spire combat decisions** with a custom
PyTorch implementation of **Proximal Policy Optimization (PPO)**.

> [!IMPORTANT]
> The current model is **not** the flat MLP described by the original v1 README.
> The code now uses structured observations, masked self-attention,
> card-to-enemy cross-attention, a factorized policy, and a 126-action space.
> The implementation under `slay_rl/` is the source of truth.

## Current scope

The project currently covers:

- Ironclad combat in a custom, simplified simulator;
- structured encoding of the player, combat context, card piles, relics, hand,
  enemies, and potions;
- legal-action masking for card plays, targets, potions, end turn, and combat
  choice screens;
- a shared actor-critic network with structured policy heads;
- PPO training with GAE, clipped policy updates, entropy regularization, and
  gradient clipping;
- curriculum sampling for normal, elite, and boss encounters;
- deterministic evaluation on fixed seed ranges;
- a CommunicationMod adapter for live combat inference.

This is a **combat agent**, not a complete autonomous run planner. In live mode,
map routing, events, shops, rewards, campfires, and other non-combat decisions
remain manual. The macro observation/action configuration is present as future
scaffolding but is not connected to a trained macro policy.

## Architecture

```text
game state
   |
   v
CombatEncoder
   |-- player scalars                     [B, 25]
   |-- combat context                     [B, 26]
   |-- deck/discard/exhaust + relics      [B, 429]
   |-- hand slots                         [B, 10, 99]
   |-- enemy slots                        [B,  5, 57]
   `-- potion slots                       [B,  5, 21]
   |
   v
slot encoders (192 dimensions) + positional embeddings
   |-- hand self-attention    (4 heads)
   |-- enemy self-attention   (4 heads)
   |-- potion self-attention  (4 heads)
   `-- cards -> enemies cross-attention (4 heads)
   |
   v
masked mean + max pooling
   |
   v
shared fusion MLP (384 dimensions)
   |-- factorized policy heads -> 126 masked logits
   `-- value head             -> V(s)
```

The attention blocks skip batch rows whose slots are entirely masked, avoiding
the undefined softmax/NaN behavior produced by fully masked attention inputs.
Invalid action logits are set to `-1e9`; if a row has no valid action, the model
falls back to the `end_turn` index.

### Factorized action policy

The flat 126-logit output is assembled from heads that share parameters across
equivalent slots:

| Action family | Count | Representation used |
|---|---:|---|
| Play card without target | 10 | card slot + global state |
| Play card on enemy | 50 | card/enemy pair + global state |
| End turn | 1 | global state |
| Use potion without target | 5 | potion slot + global state |
| Use potion on enemy | 25 | potion/enemy pair + global state |
| Choose hand card | 10 | card slot + global state |
| Choose option | 5 | global state |
| Choose discard target | 10 | card slot + global state |
| Choose exhaust target | 10 | card slot + global state |
| **Total** | **126** | |

See [`slay_rl/models/combat_model.py`](slay_rl/models/combat_model.py) and
[`slay_rl/features/combat_encoder.py`](slay_rl/features/combat_encoder.py).

## PPO training

The trainer is implemented directly in PyTorch; it does not use Stable-Baselines3
or RLlib.

Current default configuration:

| Parameter | Value |
|---|---:|
| Discount `gamma` | `0.995` |
| GAE `lambda` | `0.97` |
| PPO clip epsilon | `0.2` |
| Learning rate | `1.5e-4` |
| Entropy coefficient | `0.016` |
| Value coefficient | `0.35` |
| Rollout steps | `8192` |
| Minibatch size | `1024` |
| PPO epochs per rollout | `4` |
| Gradient norm limit | `0.5` |
| Environments | `8` in-process environments |

The training loop records reward, episode length, win rate, truncations, policy
loss, value loss, entropy, approximate KL, and clip fraction. Checkpoints are
selected first by deterministic evaluation win rate, then by average reward.

See [`slay_rl/train/train_combat.py`](slay_rl/train/train_combat.py) and
[`slay_rl/config.py`](slay_rl/config.py).

## Simulator and reward model

`slay_rl/sts_env.py` implements a custom Ironclad combat simulator with cards,
potions, relics, powers, normal encounters, Act 1 elites, and Act 1 bosses. It is
intentionally a simplified approximation of the game, not a frame-perfect clone.

The shaped reward includes combat outcome, damage dealt/taken, enemy kills,
illegal actions, tempo, energy usage, useful block, threat reduction, buffs,
debuffs, setup sequencing, lethal opportunities, hand pollution, and potion
timing. Dedicated regression tests cover reward interactions and common
reward-hacking patterns.

## Repository layout

```text
.
|-- slay_rl/
|   |-- agents/combat_agent.py       # action decoding and rule baseline
|   |-- features/combat_encoder.py   # structured observation + action mask
|   |-- models/combat_model.py       # attention actor-critic + PPO loss
|   |-- rewards/combat_reward.py     # tactical reward shaping
|   |-- train/train_combat.py        # rollouts, GAE, PPO, evaluation
|   |-- test/                         # simulator/model/reward regression tests
|   |-- config.py                     # vocabularies and hyperparameters
|   |-- sts_env.py                    # simplified combat simulator
|   |-- run_controller.py             # rule/random/model execution helpers
|   `-- main.py                       # local train/evaluation mode selector
|-- t_spire.py                        # live CommunicationMod combat adapter
|-- requirements.txt
`-- README.md
```

## Installation

Python 3.10 or newer is required by the current type syntax.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch numpy tqdm pytest
```

Install PyTorch with the command recommended for your OS, accelerator, and CUDA
version on the official PyTorch website. The committed `requirements.txt` is a
snapshot of the original Windows/CUDA 12.8 environment, not a portable lockfile;
use it only when that target matches your machine.

## Usage

### Train

`slay_rl/main.py` currently uses an explicit `mode` variable rather than a CLI.
Its default is `mode = "train"`:

```bash
python -m slay_rl.main
```

Training artifacts are written under:

- `slay_rl/checkpoints/<run_name>/`;
- `slay_rl/logs/<run_name>/train_metrics.csv`.

### Evaluate rule, random, or model policies

Edit `mode` near the top of [`slay_rl/main.py`](slay_rl/main.py):

```python
mode = "play_rule"   # or "play_random", "play_model", "train"
```

For `play_model`, also set `checkpoint_path` to a compatible checkpoint. Model
loading is strict, so a checkpoint produced by the obsolete flat-MLP version is
not compatible with the current attention architecture.

### Run the test suite

```bash
python -m pytest slay_rl/test
```

The suite covers encoder shapes, action masks, factorized action indices,
attention behavior, PPO loss, cards, potions, powers, relics, elites, bosses,
turn flow, reward shaping, anti-hacking invariants, and seeded fuzz cases.

### Live combat adapter

[`t_spire.py`](t_spire.py) reads CommunicationMod state messages from standard
input and writes game commands to standard output. Before using it:

1. install and configure Slay the Spire CommunicationMod;
2. set `CHECKPOINT_PATH` to a checkpoint from the current architecture;
3. verify `POTION_SLOT_OFFSET` and `TARGET_INDEX_OFFSET` for the mod version;
4. launch the adapter as required by the CommunicationMod setup.

The adapter defaults to deterministic inference, can fall back to the rule-based
agent, and intentionally waits for manual input outside combat.

## Evaluation status

The repository does **not** currently contain a versioned benchmark result for
the current attention model. Historical result files belonged to the obsolete
flat-MLP prototype and must not be attributed to this architecture.

To compare architectures correctly, evaluate them with the same simulator,
reward configuration, encounter distribution, fixed evaluation seeds, and
multiple independent training seeds. Report at least win rate, remaining HP,
damage taken, reward, episode length, and truncation rate.

## Known limitations

- The simulator is incomplete relative to the original game and can create a
  simulator-to-game gap.
- Training covers combat tactics, not complete run planning or deck building.
- The eight training environments run sequentially in one process; they are not
  parallel workers.
- Reward shaping is extensive and may still admit unintended strategies.
- The legal-action mask does not replace simulator-side rule validation.
- Time-limit truncation and bootstrap handling require care when interpreting
  returns.
- There is no controlled, versioned ablation proving that attention outperforms
  the former MLP architecture.
- Live behavior depends on CommunicationMod state fidelity and index conventions.

## Roadmap

- controlled MLP vs structured-attention ablations over multiple training seeds;
- harder and broader encounter distributions, including robust boss evaluation;
- corrected terminal-state bootstrap for time-limit truncations;
- true parallel rollout workers and simulator profiling;
- learned macro policy for map, rewards, shops, events, and campfires;
- tighter simulator/live parity and end-to-end run evaluation.

## Disclaimer

This is an independent research project and is not affiliated with Mega Crit.
Slay the Spire and its assets belong to their respective owners.
