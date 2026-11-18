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
