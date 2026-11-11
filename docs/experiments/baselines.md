# Baseline (a) and oracle (d) on the project data

10 November 2026. Original ACT (`third_party/act`, unmodified), images only, 300 source
demos (R2). Baseline (a) trains on the source cube only; oracle (d) adds 200 episodes of one
target with their true actions. These two bound the methods that use target videos, (b) and
(c).

## Setup

| | |
|---|---|
| Data | [Autobrik/ecse552-ocbench-block](https://huggingface.co/datasets/Autobrik/ecse552-ocbench-block), `[hub] revision` in [`configs/data.toml`](../../configs/data.toml): `deterministic` collections re-recorded on 10 November without 90° grasp turns (OCBench fork `148a7b8`) |
| Configs | `configs/act_deterministic.toml` (a), `configs/act_oracle_{A,B,C}.toml` (d), `configs/act_only_C.toml` (diagnosis) |
| Training | ACT README settings, chunk 20, 2,000 epochs, upstream 80/20 split; one A10G per run on Modal |
| Selection | `policy_best` vs `policy_last`, 100 source validation rollouts each (seeds 900,000+), ties to best |
| Test | each selected checkpoint once, 100 rollouts per object (seeds 1,000,000+), temporal ensembling (k = 0.01) |

## Results

Success over 100 test rollouts per object. All runs finished on 10 November 2026; model
repos `Autobrik/ecse552-act-deterministic`, `Autobrik/ecse552-act-oracle-{A,B,C}` and
`Autobrik/ecse552-act-only-C`.

### Baseline (a): source only, 3 seeds

| Test object | Seed 0 | Seed 1 | Seed 2 | Mean | Std | 95% Wilson, 300 rollouts |
|---|---:|---:|---:|---:|---:|---:|
| source (red cube) | 77% | 100% | 95% | 90.7% | 12.1 | [86.8, 93.5] |
| A (blue cube) | 1% | 0% | 0% | 0.3% | 0.6 | [0.1, 1.9] |
| B (red 9x6x6 block) | 67% | 99% | 81% | 82.3% | 16.0 | [77.6, 86.2] |
| C (red 9x7x6 block) | 42% | 81% | 54% | 59.0% | 20.0 | [53.4, 64.4] |
| D (blue 9x7x6 block) | 0% | 0% | 0% | 0.0% | 0.0 | [0.0, 1.3] |

### Oracle (d): source + one target with true actions, 3 seeds

| Target | Tested on | Seed 0 | Seed 1 | Seed 2 | Mean | Std | 95% Wilson, 300 rollouts | Baseline (a), mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | A | 70% | 97% | 90% | 85.7% | 14.0 | [81.2, 89.2] | 0.3% |
| A | source | 77% | 97% | 87% | 87.0% | 10.0 | [82.7, 90.3] | 90.7% |
| B | B | 86% | 84% | 95% | 88.3% | 5.9 | [84.2, 91.5] | 82.3% |
| B | source | 94% | 91% | 82% | 89.0% | 6.2 | [85.0, 92.1] | 90.7% |
| C | C | **3%** | 72% | 63% | 46.0% | 37.5 | [40.4, 51.7] | 59.0% |
| C | source | 67% | 100% | 94% | 87.0% | 17.6 | [82.7, 90.3] | 90.7% |

Oracle D is not trained yet.

### Diagnosis: ACT on C alone (`configs/act_only_C.toml`, seed 0)

200 C episodes with true actions, no source demos; not a method. 100 validation rollouts
(seeds 900,000+) per checkpoint, no test seeds.

| Checkpoint | On C | On source |
|---|---:|---:|
| `policy_best` (epoch 1604) | 87% | 97% |
| `policy_last` (epoch 1999) | 18% | 60% |

### Checkpoint selection (100 source validation rollouts each)

| Run | `policy_best` (epoch) | `policy_last` (epoch 1999) | Selected |
|---|---:|---:|---|
| (a) seed 0 | 63% (1616) | 52% | best |
| (a) seed 1 | 100% (1823) | 21% | best |
| (a) seed 2 | 94% (1456) | 95% | last |
| (d) A seed 0 | 36% (1635) | 82% | last |
| (d) A seed 1 | 85% (1421) | 97% | last |
| (d) A seed 2 | 81% (1456) | 89% | last |
| (d) B seed 0 | 94% (1905) | 95% | last |
| (d) B seed 1 | 91% (1734) | 93% | last |
| (d) B seed 2 | 86% (1776) | 84% | best |
| (d) C seed 0 | 31% (1688) | 60% | last |
| (d) C seed 1 | 98% (1578) | 82% | best |
| (d) C seed 2 | 94% (1868) | 97% | last |

## Reading

- **Colour is the gap, not shape.** The source-only policy fails on both blue objects (A, D:
  it never lifts them; 1 of 199 failures lifted the object) and transfers to the red blocks
  (B 82%, C 59%). A and D are the targets where methods (b) and (c) can show a gain.
- **The oracle works for A** (0.3% to 85.7%) and keeps the source (87.0%, within R5).
- **B has little room**: 82.3% baseline, 88.3% oracle.
- **C has no usable gap**: the oracle's mean (46.0%) is below the baseline (59.0%), so R4 is
  inconclusive there. Without seed 0 the oracle reaches 72% and 63% against a baseline of
  81% and 54% on the same seeds.
- **Oracle C seed 0 was a bad run, not a property of C** (see below).
- **High variance.** Success moves by tens of points across seeds and between checkpoints
  of one run ((a) seed 1: 100% vs 21%; C alone: 87% vs 18% on C). The Wilson interval pools
  the three seeds' rollouts and does not include that spread; compare methods per seed and
  report the spread.
- **Augmentation (e) changes the picture**: source-only training with colour jitter reaches
  98.7% on A and 66.7% on D ([augmentation.md](augmentation.md)).

## Why oracle C seed 0 fails

Checked on 10 validation rollouts per case (seeds 900,000+): position of the gripper pinch
point relative to the object centre when the gripper first closes. The gripper closes along
y; the opening is 8.5 cm, so a 7 cm block leaves 0.75 cm per side and a 6 cm one 1.25 cm.

| Policy (seed) | Object | y offset, mean ± std | Height above centre | Success |
|---|---|---:|---:|---:|
| Expert demos | C | 0.0 ± 0.1 cm | 1.5 cm | - |
| Oracle C (0), selected | C | **+0.9 ± 0.3 cm** | 1.5 cm | 0/10 |
| Oracle C (0), `policy_best` | C | +1.0 ± 0.4 cm | 1.5 cm | 0/10 |
| Oracle C (0), selected | source | +1.1 ± 0.2 cm | 1.5 cm | 6/10 |
| Oracle B (0) | B | +0.6 ± 0.3 cm | 0.9 cm | 9/10 |
| Baseline (1) | C | +0.1 ± 0.4 cm | 0.1 cm | 8/10 |
| Baseline (1) | source | -0.2 ± 0.3 cm | -0.7 cm | 10/10 |

- Oracle C reaches the block at the right height but about 1 cm off-centre in y, more than
  the 0.75 cm a 7 cm block allows: a finger lands on the block, the gripper closes empty and
  pushes the block about 0.8 cm per attempt. The same offset on the source cube still fits
  (6/10).
- Both of its checkpoints have the offset, so selecting on target rollouts would not fix it.
- It also uses C's grasp height on the source cube: it does not tell the two red objects apart.
- The baseline is centred and grasps C like the cube (low), which works in 59% of tests.
- **Conclusion (seeds 1, 2 and C alone): seed 0 was a bad run.** Seeds 1 and 2 of the same
  config reach 72% and 63% on C, and ACT on C alone reaches 87% (`policy_best`), so neither
  mixing source and C nor the 0.75 cm margin prevents learning C. Seed 0 was weak everywhere
  (60% on source validation, 97% and 98% for the other seeds). C and D stay as targets.
- The margin still makes C fragile: one run goes from 87% to 18% between two checkpoints.
- The grasp offsets of seeds 1 and 2 were not measured.

## Limits

- Oracle D is missing; the C-alone diagnosis has one seed and validation rollouts only.
- Every checkpoint, the oracle's too, is selected on source validation rollouts; this may
  understate the oracle on its target.
- The expert lost its 90° grasp turns after the 100-demo result (R2); R2 stays at 300. The
  earlier data gave 99% on source and 7% on A with one seed ([act_original.md](act_original.md)).
