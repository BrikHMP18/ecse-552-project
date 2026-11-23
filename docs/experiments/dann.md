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

## Not in this task

- The colour-jitter IDM (the comparison arm of the next step): owner to be agreed by the team.
