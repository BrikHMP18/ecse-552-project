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
  No seeds 1 and 2, no policies (c).
- The control is the worst, which points to the mixed batches (batch norm statistics of
  source and target together) as what breaks the target labels; the adversarial term
  recovers part of it (0.607 to 0.220), not up to (b).
- The error is on the wrist, full turns as in [idm.md](idm.md); the other arm joints stay
  within 0.06 rad.
- The discriminator's loss stays at ln 2 in most epochs while TSR and TTR swing between 0
  and 1 (A ends at 0.90 and 0.87): the adversarial game does not settle.

## Not in this task

- The colour-jitter IDM (the comparison arm of the next step): owner to be agreed by the team.
