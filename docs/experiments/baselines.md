# Baseline (a) and oracle (d) on the project data

10 November 2026. Original ACT (`third_party/act`, unmodified), images only, 300 source
demos (R2). Baseline (a) trains on the source cube only; oracle (d) adds 200 episodes of one
target with their true actions. These two bound the methods that use target videos, (b) and
(c).

## Setup

| | |
|---|---|
| Data | [Autobrik/ecse552-ocbench-block](https://huggingface.co/datasets/Autobrik/ecse552-ocbench-block), `[hub] revision` in [`configs/data.toml`](../../configs/data.toml): `deterministic` collections re-recorded on 10 November without 90° grasp turns (OCBench fork `148a7b8`) |
| Configs | `configs/act_deterministic.toml` (a), `configs/act_oracle_{A,B,C}.toml` (d) |
| Training | ACT README settings, chunk 20, 2,000 epochs, upstream 80/20 split; one A10G per run on Modal |
| Selection | `policy_best` vs `policy_last`, 100 source validation rollouts each (seeds 900,000+), ties to best |
| Test | each selected checkpoint once, 100 rollouts per object (seeds 1,000,000+), temporal ensembling (k = 0.01) |

## Results

Success over 100 test rollouts per object. All runs finished on 10 November 2026; model
repos `Autobrik/ecse552-act-deterministic` and `Autobrik/ecse552-act-oracle-{A,B,C}`.

### Baseline (a): source only, 3 seeds

| Test object | Seed 0 | Seed 1 | Seed 2 | Mean | Std | 95% CI (t, n = 3) |
|---|---:|---:|---:|---:|---:|---:|
| source (red cube) | 77% | 100% | 95% | 90.7% | 12.1 | ± 30.1 |
| A (blue cube) | 1% | 0% | 0% | 0.3% | 0.6 | ± 1.4 |
| B (red 9x6x6 block) | 67% | 99% | 81% | 82.3% | 16.0 | ± 39.9 |
| C (red 9x7x6 block) | 42% | 81% | 54% | 59.0% | 20.0 | ± 49.6 |
| D (blue 9x7x6 block) | 0% | 0% | 0% | 0.0% | 0.0 | ± 0.0 |

### Oracle (d): source + one target with true actions, seed 0 (provisional)

| Target | On the target | On source | Baseline (a) on that target, mean |
|---|---:|---:|---:|
| A | 70% | 77% | 0.3% |
| B | 86% | 94% | 82.3% |
| C | **3%** | 67% | 59.0% |

### Checkpoint selection (100 source validation rollouts each)

| Run | `policy_best` (epoch) | `policy_last` (epoch 1999) | Selected |
|---|---:|---:|---|
| (a) seed 0 | 63% (1616) | 52% | best |
| (a) seed 1 | 100% (1823) | 21% | best |
| (a) seed 2 | 94% | 95% | last |
| (d) A | 36% | 82% | last |
| (d) B | 94% | 95% | last |
| (d) C | 31% | 60% | last |

## Reading

- **Colour is the gap, not shape.** The source-only policy fails on both blue objects (A, D:
  it never lifts them; 1 of 199 failures lifted the object) and transfers to the red blocks
  (B 82%, C 59%). A and D are the targets where methods (b) and (c) can show a gain; B has
  little room (82% vs 86% oracle).
- **The oracle works for A** (0.3% to 70%) at some cost on source (77%).
- **Oracle C fails at the grasp, by a lateral offset** (see below): 3% on C, below the
  baseline's 59%, and none of the 97 failures lifted the block. It is not an upper bound.
- **High variance.** Source success goes from 77% to 100% across seeds, and between
  checkpoints of one run (seed 1: 100% vs 21%). With 3 seeds the intervals are wide; compare
  methods per seed and report the spread.

## Why oracle C fails

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
- Not known: why the oracles (C, and less so B) learn the offset, and whether it is this
  seed. Training losses are normal (validation L1 0.059; baseline 0.062). Seeds 1 and 2
  would tell; the 0.75 cm margin makes C and D sensitive to any policy.

## Limits

- The oracle has one seed.
- Every checkpoint, the oracle's too, is selected on source validation rollouts; this may
  understate the oracle on its target.
- The expert lost its 90° grasp turns after the 100-demo result (R2); R2 stays at 300. The
  earlier data gave 99% on source and 7% on A with one seed ([act_original.md](act_original.md)).
