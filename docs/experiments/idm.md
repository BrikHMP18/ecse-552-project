# Inverse dynamics model (b): label error

17 November 2026, Gonzalo. First results, one training seed. No policy has been trained on
the labels yet.

## What we tested

Method (b) labels the target videos with an inverse dynamics model (IDM): from two
consecutive frames of the three cameras, it predicts the action ACT trains on (six joint
targets and the gripper opening). It learns from labeled source demos only. Before training
any policy on its labels, we measured how wrong those labels are, using the targets' true
actions for evaluation only.

We trained two IDMs that differ only in their data:

- **`idm`**: the 300 source demos.
- **`idm_turned`**: the same plus the 100 source demos with a turned grasp
  ([target_E.md](target_E.md)), so that it sees the wrist motion E needs.

Both reach the same validation error on the source (standardized MAE 0.024, best epoch 19
of 20).

## Results

Mean label error over all transitions of each collection. Joints in radians, gripper opening
in [0, 1]. The other joints stay below 0.03 everywhere.

|                            | wrist3, `idm` | wrist3, `idm_turned` | gripper, `idm` | gripper, `idm_turned` |
| -------------------------- | ------------: | -------------------: | -------------: | --------------------: |
| Source (50 held-out demos) |         0.013 |                0.032 |          0.014 |                 0.013 |
| A, blue cube               |         0.046 |                0.041 |          0.019 |                 0.017 |
| D, blue 9x7x6 block        |         0.017 |                0.035 |          0.039 |                 0.113 |
| E, turned grasp            |     **0.835** |            **0.275** |          0.047 |                 0.017 |

## Findings

**A change of colour or shape costs little.** On A and D, without any adaptation, the error
is about twice the source error and stays small. An unadapted IDM is likely good enough for
these targets.

**A new motion breaks the IDM.** On E, `idm` gets the wrist turn wrong by 0.84 rad (48
degrees) on average, and all 200 episodes are mislabeled. This is the risk predicted in
[target_E.md](target_E.md): the source demos almost never hold the wrist at E's angle, and
an IDM cannot label a motion it has never seen. The other joints remain accurate.

**The turned source demos fix most of it.** With `idm_turned`, the typical E episode is
labeled as well as the source (median wrist3 error per episode 0.024 rad). The average stays
at 0.28 rad because of a few bad episodes: 21 of 200 exceed 0.2 rad, and 13 exceed 0.5 rad.
They may be the 14 episodes whose wrist leaves the range of the source demos; this has not
been checked.

**But they cost something on D.** The gripper error on D triples (0.039 → 0.113). We do not
know why yet, and with one training seed it may partly be noise.

**In short:** `idm_turned` is the IDM for E; for D the choice is open.

## Pseudo-labels

`scripts/export_labels.py` labeled A, D and E with both IDMs, in the format of the README
(Pseudo-labels), under `data/labels/<target>-<idm>-seed0/`. Every file matches its episode's
length, and the errors recomputed from the files equal the table above. They are not yet
uploaded to the dataset repository.

## Limits

- One training seed per IDM.
- Label error is not policy success: only a policy trained on the labels shows whether they
  are good enough.

## Next

1. Choose the IDM for each target (team).
2. Train IDM seeds 1 and 2 to check that the differences hold.
3. Check whether E's worst episodes leave the source range, and why D's gripper error grows.
4. Upload the chosen labels and train (b) with 3 seeds: A first as a pipeline check, then D and E.

## Reproduction with 3 seeds (19 November 2026, Brik)

Same code and configs (commit `bf5d734`), seeds 0, 1, 2 of both IDMs trained again on Modal
(about one hour each on an A10G), evaluated and exported locally. Validation sMAE on the
source: `idm` 0.025, 0.023, 0.026; `idm_turned` 0.023, 0.023, 0.023.

Mean label error, seeds 0 / 1 / 2:

| | wrist3, `idm` | wrist3, `idm_turned` | gripper, `idm` | gripper, `idm_turned` |
|---|---:|---:|---:|---:|
| Source (50 held-out demos) | 0.013 / 0.013 / 0.012 | 0.025 / 0.028 / 0.033 | 0.012 / 0.013 / 0.012 | 0.012 / 0.013 / 0.013 |
| A | 0.053 / 0.054 / 0.044 | 0.043 / 0.046 / 0.041 | 0.020 / 0.017 / 0.015 | 0.021 / 0.014 / 0.015 |
| D | 0.020 / 0.019 / 0.015 | 0.031 / 0.026 / 0.026 | 0.110 / 0.080 / 0.058 | 0.113 / 0.033 / 0.041 |
| E | 0.839 / 0.769 / 0.749 | 0.278 / 0.273 / 0.271 | 0.039 / 0.033 / 0.035 | 0.016 / 0.016 / 0.016 |

- **The table above is reproduced**, on the source, A and E, and across seeds.
- **The gripper error on D is run-to-run noise, not a cost of the turned demos**: it ranges
  from 0.058 to 0.110 for `idm` and from 0.033 to 0.113 for `idm_turned`.
- **E's bad episodes are a full-turn ambiguity of the wrist.** In 14 of E's 200 episodes the
  true `wrist3` target is below -4.5 rad in more than 10% of the frames. They are exactly
  the episodes that `idm_turned` labels with an error above 0.5 rad (14, 13, 13 per seed),
  and on those frames the label is the true target plus 2 pi (median signed error 6.14,
  6.14, 5.75 rad). An image cannot tell an angle from the same angle one turn away, and
  only 3 of the 100 turned source demos go there. No E episode leaves the range of the 400
  source demos. Without these 20 episodes (any frame below -4.5 rad) the mean error of
  `idm_turned` on E is 0.035 to 0.038 rad.
- Inside such an episode the label jumps by about 2 pi (largest step between consecutive
  labels: 7 rad). Unwrapping each episode's `wrist3` labels in time (`numpy.unwrap`, which
  needs no target action) lowers E's mean error per episode from 0.21 to 0.07-0.09 rad and
  the episodes above 0.5 rad from 13-14 to 4-6, but it does not remove them and it adds two
  bad episodes on D for seed 2. Not applied: how to handle the ambiguity is open (team).

## Label filter and first policies (b) (19 November 2026, Brik)

Decided before training any policy on the labels:

- One IDM for every target: `idm_turned` (never chosen per target by target label error, R1).
  The policy with training seed k uses the labels of the IDM with seed k
  (`labels/<target>-deterministic-idm_turned-seed<k>/` in the dataset repository).
- Filter (`max_label_step = 2.0` in `[[targets]]`, `act_original.plausible`): a target
  episode is dropped when its labels move an arm joint more than 2 rad in one step. It reads
  the labels only. The true targets stay below 1.1 rad per step in the 300 source demos and
  in 99 of the 100 turned ones (one has a 3.0 rad step), and below 1.4 on A, D and E.
- Check of the filter with the true targets (diagnostic only; "bad" = some `wrist3` label
  more than 3 rad from the truth; ranges over the 3 IDM seeds, 200 episodes per target):

| Target | Bad | Dropped | Bad kept | Good dropped | `wrist3` error per episode, all -> kept (rad) |
|---|---:|---:|---:|---:|---:|
| A | 4-5 | 4-5 | 0 | 0 | 0.033 -> 0.023 |
| D | 0-2 | 0-3 | 0 | 0-1 | 0.028 -> 0.026 |
| E | 20-23 | 21-23 | 1-2 | 0-2 | 0.21 -> 0.033 |

- Limit: the dropped E episodes are the ones that need the wrist below about -4.5 rad, so
  the policy sees fewer of those block poses. The test rollouts are not filtered.
- Policies, same recipe as the oracle (300 source demos + 200 target videos, from scratch):
  `act_idm_A.toml` (pipeline check), `act_idm_E.toml`, and `act_idm_E_nofilter.toml` (the
  filter as a measured ablation). Seeds 0 and 1 first.

