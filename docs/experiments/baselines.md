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
- **Oracle C is broken, not an upper bound**: 3% on C, below the baseline's 59%, and none of
  the 97 failures lifted the block. Not diagnosed. One hypothesis: the expert grasps C 2 cm
  higher than the source cube, and the two red objects look alike, so the policy mixes the
  two grasp heights. Check before running more oracle seeds for C or D.
- **High variance.** Source success goes from 77% to 100% across seeds, and between
  checkpoints of one run (seed 1: 100% vs 21%). With 3 seeds the intervals are wide; compare
  methods per seed and report the spread.

## Limits

- The oracle has one seed.
- Every checkpoint, the oracle's too, is selected on source validation rollouts; this may
  understate the oracle on its target.
- The expert lost its 90° grasp turns after the 100-demo result (R2); R2 stays at 300. The
  earlier data gave 99% on source and 7% on A with one seed ([act_original.md](act_original.md)).
