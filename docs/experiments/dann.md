# DANN on the IDM (c): protocol

23 November 2026, Myriam. Written before any DANN run: no result exists yet. The settings
below are fixed in advance and are not tuned on target labels (R1).

## Method

DANN (Ganin et al., 2016), not ADDA: one IDM trained end to end with a gradient reversal
layer, no separate target encoder. Code: `src/action_free/idm.py` (`train_idm` with
`cfg["target"]`), configs `configs/idm_dann_{A,D,E}.toml`, otherwise as `idm_turned.toml`.

- Every step passes 64 source pairs and 64 target pairs through the encoder in one batch,
  so batch norm sees both domains. The action loss uses the source pairs only.
- The discriminator (MLP 1536 -> 256 -> 1, binary cross-entropy) reads the concatenated
  camera features through the gradient reversal layer. Its optimizer is Adam (lr 3e-4,
  constant); the IDM keeps AdamW with the cosine schedule of (b).
- The target is read as images only (`labeled=False`). Its actions are used only to measure
  label error after training.

## Fixed in advance (R1)

| Setting | Value |
|---|---|
| `dann_lambda` (maximum weight of the reversed gradient) | 0.5 |
| `dann_gamma` (schedule `lambda_max * (2 / (1 + exp(-gamma p)) - 1)`) | 10 |
| Checkpoint | `best.pt` = last epoch (20); no selection on target labels |

## Control

`configs/idm_dann_D_lambda0.toml` is `idm_dann_D.toml` with `dann_lambda = 0`: same mixed
batches and the discriminator still trains (TSR/TTR measure how separable the domains are),
but no gradient reaches the features. It separates the adversarial effect (DANN vs control)
from batch norm seeing target images (control vs `idm_turned`).

## Target episodes

Adaptation uses the train split of the target (160 of 200 episodes, `data.split_of`); its 40
validation episodes are used only for TTR. Pseudo-labels are exported for all 200 and the
policies train on all 200.

Option, not adopted: adapt on all 200 unlabeled episodes. R1 allows it (images only) and it
would cover every episode that gets labeled, but it leaves no held-out target episode for TTR.

## Decision rule (23 November, before measuring target label error)

Seed 0 of the four configs ran on Modal (Brik's approval). Only their source validation
error had been seen: sMAE 0.032–0.033 for DANN, 0.024 for the control, 0.023 for
`idm_turned`. It is reported, not used to decide (R5 concerns the policy).

- Metric: sMAE of the pseudo-labels on all 200 target episodes against the true actions
  (evaluation only, R1). Wrist3, gripper and the episodes the 2 rad label filter drops are
  reported alongside.
- DANN improves on a target if its sMAE is at least 5% below the lowest of the three
  `idm_turned` seeds: A below 0.0278 (0.0293), D below 0.0332 (0.0349), E below 0.0618
  (0.0650).
- Where it improves: seeds 1 and 2, then policies (c) with the recipe of (b), label filter
  included, once the team approves the runs. Where it does not: (c) is reported by its
  label error only.
- The control is reported with D: DANN against control is the adversarial effect, control
  against `idm_turned` is batch norm seeing target images.

## Results, seed 0 (23 November)

Label error of `best.pt` (epoch 20) on all 200 target episodes, true actions read for
evaluation only; source = the 50 held-out demos (`scripts/evaluate_idm.py`). Dropped = episodes
whose labels move an arm joint more than 2 rad in one step (the label filter of (b)).

| IDM | Target | Source sMAE | Target sMAE | Wrist3 (rad) | Gripper | Dropped |
|---|---|---|---|---|---|---|
| `idm_turned` (b) | A | 0.020 | 0.030 | 0.043 | 0.021 | 5 |
| DANN | A | 0.030 | 0.114 | 0.521 | 0.026 | 29 |
| `idm_turned` (b) | D | 0.020 | 0.065 | 0.031 | 0.113 | 3 |
| Control (`dann_lambda = 0`) | D | 0.021 | 0.607 | 2.684 | 0.134 | 198 |
| DANN | D | 0.028 | 0.220 | 0.690 | 0.125 | 145 |
| `idm_turned` (b) | E | 0.020 | 0.065 | 0.278 | 0.016 | 21 |
| DANN | E | 0.028 | 0.198 | 1.054 | 0.024 | 28 |

- By the rule, DANN improves on no target: its target sMAE is 3 to 4 times that of (b).
  No seeds 1 and 2. After this result the team still ran the policies (c) on seed 0 (Brik's
  approval, 23 November), a departure from the rule, so that (c) also has a success rate
  ([Policies (c), seed 0](#policies-c-seed-0-24-november)).
- The control is the worst, which points to the mixed batches (batch norm statistics of
  source and target together) as what breaks the target labels; the adversarial term
  recovers part of it (0.607 to 0.220), not up to (b).
- The error is on the wrist, in every phase of the episode (idm.md, label error per phase);
  the other arm joints stay within 0.06 rad. It is a constant offset, not the full turns of
  (b) on E ([Why the labels fail](#why-the-labels-fail-24-november)).
- The discriminator's loss stays at ln 2 in most epochs while TSR and TTR swing between 0
  and 1: the adversarial game does not settle. At epoch 20: A 0.467 (TSR 0.90, TTR 0.87),
  D 0.693 (TSR 0.00, TTR 1.00: every pair called target, not chance), E 0.653 (0.78, 0.72).
  The control ends at 0.000 (TSR and TTR 1.00): without the reversed gradient the
  discriminator separates the domains perfectly.

## Why the labels fail (24 November)

`scripts/dann_diagnosis.py` on seed 0; true target actions for evaluation only (R1).

**A constant wrist offset, not full turns.** Wrist3 error averaged over each of the 200
episodes (mean ± std across episodes), share of the steps off by more than 0.5 rad, share of
those that are a full turn (2π) away, and correlation of the labels with the true wrist3.
(b) is the range over its 3 seeds.

| IDM | Target | Offset per episode (rad) | Steps > 0.5 rad | Of which full turns | Correlation |
|---|---|---|---|---|---|
| (b) | A | −0.01 to +0.01 | 0.6% | 13–23% | 0.85–0.87 |
| DANN | A | −0.49 ± 0.22 | 22.4% | 1% | 0.32 |
| (b) | D | −0.01 to +0.01 | 0.1–0.3% | 0% | 0.97–0.99 |
| DANN | D | −0.69 ± 0.09 | 36.0% | 0% | 0.58 |
| Control | D | −2.67 ± 0.48 | 70.4% | 6% | 0.09 |
| (b) | E | +0.16 to +0.17 (± 0.7) | 5.6–6.1% | 36–44% | 0.33–0.36 |
| DANN | E | +1.00 ± 0.60 | 64.0% | 4% | 0.27 |

- On A and D the grasp is the source's, yet DANN shifts the wrist the same way in nearly
  every episode. The 2 rad filter drops jumps, not a constant offset, so most of these
  labels reach ACT.
- Not missing data: DANN trains on the same 400 source demos as (b), turned grasps
  included; the only difference is the 64 target pairs in each batch.

**The weights, not the batch-norm statistics.** Wrist3 error (rad) on the target after
re-estimating the batch-norm running statistics (weights unchanged) on source pairs, mixed
64 + 64 batches, or target pairs:

| IDM | Target | As trained | Source | Mixed | Target |
|---|---|---|---|---|---|
| (b) seed 0 | A | 0.043 | 0.045 | 0.043 | 0.048 |
| DANN | A | 0.521 | 0.665 | 0.519 | 0.412 |
| (b) seed 0 | D | 0.031 | 0.032 | 0.032 | 0.050 |
| DANN | D | 0.690 | 0.846 | 0.680 | 0.551 |
| Control | D | 2.684 | 8.094 | 2.707 | 1.770 |
| (b) seed 0 | E | 0.278 | 0.281 | 0.309 | 0.366 |
| DANN | E | 1.054 | 1.029 | 1.016 | 1.014 |

- No statistics bring DANN or the control back to (b): training with target images in the
  batch changed the weights. The control with source statistics fails on the source too
  (sMAE 0.256 against 0.021): its weights rely on the mixed statistics.
- Wrist3 takes the damage: of the arm joints it has the largest spread (std 0.40–0.59 rad
  against at most 0.31) and, likely, the faintest cue in the image (the gripper turning
  about its own axis), so a small shift of the features is a large angle.
- Side result, not a method: mixed statistics on (b) lower D's gripper error (0.113 to
  0.058; sMAE 0.065 to 0.045), raise E's sMAE (0.065 to 0.075) and leave A unchanged.
  Measured with target actions, so it cannot be chosen per target (R1).

## Policies (c), seed 0 (24 November)

ACT with the recipe of (b), label filter included, on the DANN labels
(`configs/act_dann_{A,D,E}.toml`; target episodes kept: A 171, D 55, E 172). Checkpoint
chosen on 100 source validation rollouts (`scripts/select_checkpoint.py`), then tested once
on 100 test seeds of the target and of the source. Wilson 95% intervals in brackets.

| Target | Selected | Target success | Source success | (b) seed 0, target / source |
|---|---|---|---|---|
| A | best, epoch 1833 | 6% [2.8, 12.5] | 94% [87.5, 97.2] | 63% / 88% |
| D | last, epoch 1999 | 0% [0, 3.7] | 84% [75.6, 89.9] | 75% / 96% |
| E | best, epoch 1487 | 0% [0, 3.7] | 63% [53.2, 71.8] | 98% / 43% |

- On every target (c) is at the level of (a), which never sees the target (A 0.3%, D 0%,
  E 0%): the DANN labels add nothing the policy can use.
- The policies fail at the grasp: on 50 validation rollouts per target
  (`scripts/diagnose_failures.py`) the object is never lifted in 45 (A), 50 (D) and 50 (E).
  This is consistent with the wrist offset of the labels; the wrist angle in the rollouts
  was not measured.
- Seeds 1 and 2 of (c) are not run.
- Checkpoints: [ecse552-act-dann-A](https://huggingface.co/Autobrik/ecse552-act-dann-A),
  [-D](https://huggingface.co/Autobrik/ecse552-act-dann-D),
  [-E](https://huggingface.co/Autobrik/ecse552-act-dann-E).

## Not in this task

- The colour-jitter IDM (the comparison arm of the next step): owner to be agreed by the team.
