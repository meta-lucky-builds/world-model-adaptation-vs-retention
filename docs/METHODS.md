# Methods

## Setup

- **Task:** Two-Room, where an agent starts in the left room and must pass a doorway to reach a fixed
  goal. The score is `exp(-D(s, g) / 64)`, with `D` measured through the doorway while the agent is
  left of the wall.
- **Model:** the public LeWorldModel checkpoint, with a ViT-Tiny encoder and a 6-layer predictor.
- **Planner:** CEM with 256 candidates, 6 iterations, 32 elites and a 5-block horizon. Each block is
  5 raw actions. It executes one block, then re-plans.
- **Reward head:** a sigmoid MLP fitted on 3,000 episodes of the public dataset, so that planning
  needs no goal image or online reward.
- **Deployment:** 500 episodes under one shift, then 50 more after a silent return to the original
  dynamics. Evaluation uses 100 fixed held-out episodes, every 10 episodes up to 200, then every 50.

## Shifts

Both rotate the command by a state-dependent angle before it moves the agent, from the paper's
Eqs. A7 and A8:

```
wave: alpha(s) = 75 deg * sin(2 pi (x - 20) / 88)
grid: alpha(s) = 75 deg * sin(2 pi (x - 20) / 96) * cos(2 pi (y - 20) / 96)
```

## Adaptation

Each window has 4 keyframes 5 raw steps apart, with one window starting at every raw step. Training
minimizes the one-step latent error of the paper's Eq. 4 with AdamW, in evaluation mode, with
batch 64 and gradient clipping at 1.0. `xN` makes N updates per 64 new windows, so x1 makes 250
updates in 500 episodes and x16 makes 4,000. Only the action encoder, predictor and prediction head
are trained.

## Action calibration

Let `F` be the frozen predictor, and `a_i` the i-th raw command in a block starting at state `s_0`.
A Two-Room shift step is:

```
a_applied_i = clip(R(alpha(s_i)) a_i)
s_{i+1}     = s_i + v a_applied_i        (away from walls)
```

**Premise.** A shifted step equals a base step given `a_applied_i`, so the shifted world is the
training world with translated actions. `test_applied_actions_reproduce_the_shifted_world_exactly`
checks this for exact equality of positions.

**Function class.** Since `s_i` depends only on `s_0` and earlier commands, a matrix `D_i` exists
such that:

```
a_applied_i = clip((I + D_i(s_0, a_0, ..., a_{i-1})) a_i)
```

Here, `D_i = R(alpha(s_i)) - I`. The calibrator learns `D_i` from the block's start latent, compressed
to 8 dimensions, the earlier commands, and a one-hot of position `i`. It assumes linearity in the
current action, not rotation specifically. It can only approximate the true map, and needs the latent to reveal `s_0`.

**Identity at the start.** The last layer is zero, so every `D_i` is 0 and the calibrated block
equals the input exactly, including for out-of-range planner candidates.

**Causality.** A mask stops `D_i` from seeing commands at or after position `i`.

**Objective.** With `F` frozen, training minimizes Eq. 4 over only the calibrator's weights `phi`:

```
L(phi) = mean || F(z_{0:j}, C_phi(z_{0:j}, b_{0:j})) - z_{j+1} ||^2
```

**What this doesn't prove.** Being able to express the map doesn't mean training finds it. Different
action sequences with the same displacement predict about the same thing, so the learned actions
need not equal the true ones. Training only sees encoded latents, while planning also feeds it
predicted ones.

## The five further ideas

| Arm | Design | Fixed settings |
|---|---|---|
| `calibration_before_now` | At each replan, plans with the frozen model alone if it had under half the calibrated error on the last 3 blocks. Doesn't store windows finishing while off, if the frozen error is also below 0.1 | Margin 0.5, threshold 0.1 |
| `calibration_fast` | Adds a 2x2 matrix per environment, adapted within each episode and then reset, including during evaluation | 3 normalized gradient steps of 0.1 per replan |
| `latent_correction` | Adds a zero-start residual to the frozen predictor's next latent, clamping out-of-range candidates first | 16 updates per 64 windows, learning rate 1e-3 |
| `rollout_loss_x4` | Adds a free-running loss from each window's first latent, as an analog of V-JEPA 2.1's dense loss | Weight 0.5 |
| `anchored_x4` | Penalizes squared drift from the pretrained weights, summed over all trainable weights | Weight 0.01 |

## How results were protected

- **Fixed thresholds:** rules live in the code, written before the results they judge. The git history
  starts after the runs, so it can't independently prove that order.
- **Held-out separation:** deployment, evaluation and trajectory seeds are disjoint, and evaluation
  data never reaches a replay buffer.
- **Data gates before any run:** rendered frames must match the dataset (difference 0.00004 on a
  0-255 scale), dataset speed must match the environment (5.0), and action timing must be right. The
  checkpoint must predict real data far better with real actions than shuffled ones (0.064 vs 1.64).
- **Baseline gates:** a shift is dropped if it doesn't damage the frozen model.
- **Shared comparison files:** the five ideas reuse the first run's reward head, baselines and test
  trajectories.
- **Independent reviews:** separate code reviews changed the rules before results existed. For
  example, the calibrator had to beat x16 as well as x4, because it makes as many updates. A further
  review checked these documents against the saved evidence.

## Differences from the paper

- **Checkpoint:** the public LeWorldModel checkpoint, where the paper trained its own.
- **Reward head data:** the public dataset, which probably overlaps the checkpoint's training data.
- **Starts:** uniform across the left room, which the paper doesn't specify exactly.
- **Window count:** 32 windows per 50-step episode, counting the final observation. This is my reading
  of Appendix A1.2, which isn't fully explicit.
- **Actions:** rollouts score unclipped candidates, while the environment receives clipped actions.
  As in upstream's solver, each CEM iteration's first candidate is the current mean.
- **Rotation clamp:** the base environment clamps each axis after rotation, where the paper states no
  clip for Two-Room.
- **Scope:** PPO-TTT, AdaJEPA and the sparse and dense-stream ablations aren't included, and
  `reset_local` is the paper's Appendix A6.1 control, not AdaJEPA.

## Reading the numbers

- **Calibration's offline results are rough:** `results/offline/` used a small random-policy dataset,
  and the bottleneck was chosen on grid alone from the same test episodes that report it.
- **Switch-back is dilution-limited:** 10 episodes after the switch, about 2% of replay data is new
  for most arms, so it's reported rather than judged for them.
- **Fast steps have a fixed size:** each replan moves the matrix by up to 0.3, even near the best value.
- **Latent error is partial for two arms:** `latent_mse` leaves out the gate and the fast matrix, so
  judge those by score.
- **Repeat runs vary a little:** x4's AUC was 0.729 and 0.722 in stage 1, and 0.740 and 0.717 in the
  calibration run, which also used a new reward head and schedule. The frozen wave score moved from
  0.499 to 0.483 with a refit reward head.

## Engineering notes

- **`transformers` must stay below 5:** version 5 renames the ViT weights, so the checkpoint won't
  load.
- **Limit threads for CPU tests:** with one thread per reported host core (192 on one machine), small
  CPU tests appeared to hang. The run scripts now limit them, and the tests then took 98 seconds.
- **Runs per card:** about 3.5 GB per run, measured at 2–3 runs per card. Six per card used about 21 GB
  of a 32 GB RTX 5090, with all cards at 100%.
