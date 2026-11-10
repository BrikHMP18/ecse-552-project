# Method (e): source only with image augmentation

10 November 2026. Implemented and launched by Brik's session so that the comparison is not
blocked. **Gonzalo: review the settings below; they are a first choice, not a tuned one.**

## What it is

The baseline (a) with one change: every training sample is augmented. No target videos, no
target actions. It answers whether target videos are needed at all; since the gap of the
baseline is colour (A and D at 0%, B and C transfer), it is the strongest alternative.

| | |
|---|---|
| Config | [`configs/act_augment.toml`](../../configs/act_augment.toml): `act_deterministic.toml` plus `augment = true` |
| Code | `augment_images` and `AUGMENT` in [`src/action_free/act_original.py`](../../src/action_free/act_original.py); upstream ACT is not edited, the training dataset is wrapped |
| Colour | brightness, contrast and saturation ± 30%; hue over the full circle (± 0.5), so the red cube takes every colour |
| Geometry | shift of up to 4 pixels (of 224), edge pixels repeated |
| Per sample | one random draw, applied to the three cameras alike |
| Not augmented | validation loss, validation and test rollouts |
| Runs | 300 source demos, seeds 0, 1, 2; same selection and test as (a): every object, 100 rollouts |

![Augmented samples](../augmentation.png)

First row: the original frame (front, side, wrist cameras). Other rows: five random draws.

## Open choices (Gonzalo)

- **Hue range.** The full circle is the favourable case for (e): it assumes the target may
  have any colour. A narrow range would not reach blue from red.
- **Shift size.** Kept small because the grasp needs millimetres; RAD uses larger crops.
- **Robot and background change colour too.** The jitter is applied to the whole image.
- No other augmentation (blur, cutout, noise) and no tuning: one setting, fixed before the
  results.

To change a setting: edit `AUGMENT`, commit, and launch
`modal run --detach scripts/modal_train.py --config configs/act_augment.toml --seeds 0,1,2`
(it overwrites the runs in `Autobrik/ecse552-act-augment`; test seeds are used once per
selected checkpoint).

## Results

Pending: launched on 10 November 2026.
