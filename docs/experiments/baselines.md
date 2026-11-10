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

Pending: the runs were launched on 10 November 2026.
