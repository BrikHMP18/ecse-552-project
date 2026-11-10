# Baseline (a) with the original ACT: data variants and robot state

9 November 2026. Original ACT (`third_party/act`, unmodified) trained on 300 source demos,
seed 0, then selected on validation rollouts and tested once on the source cube and target A.

## Setup

| | |
|---|---|
| Code | `main` at `5c46c01`; ACT `742c753`; OCBench fork `54783cb` |
| Data | [Autobrik/ecse552-ocbench-block](https://huggingface.co/datasets/Autobrik/ecse552-ocbench-block) at `3d0229e` |
| Training | ACT README settings, chunk 20 (2 s at 10 Hz), 2,000 epochs, upstream 80/20 split; one A10G, ~2 h per run |
| Selection | `policy_best` (lowest validation loss) vs `policy_last`, 100 source validation rollouts (seeds 900,000+) |
| Test | 100 rollouts per object (seeds 1,000,000+), temporal ensembling (k = 0.01), 95% Wilson intervals |

| Run | Data | Robot state | Checkpoints | Curves |
|---|---|---|---|---|
| `act_original` | `source-n300-seed0`: random cube yaw, OCBench expert (detours, random speed) | no | [HF](https://huggingface.co/Autobrik/ecse552-act-original) | [W&B](https://forge.coreweave.com/wandb/mu-robotics/ecse552/runs/vunir7pa) |
| `act_clean` | `source-clean-n300-seed0`: yaw fixed at 0, deterministic expert (like ACT's sim) | no | [HF](https://huggingface.co/Autobrik/ecse552-act-clean) | [W&B](https://forge.coreweave.com/wandb/mu-robotics/ecse552/runs/86jkbauq) |
| `act_clean_proprio` | same as `act_clean` | **yes** (reference only; breaks R3) | [HF](https://huggingface.co/Autobrik/ecse552-act-clean-proprio) | [W&B](https://forge.coreweave.com/wandb/mu-robotics/ecse552/runs/thyrbqin) |

## Results

| Run | Validation: best / last | Selected | **Test source** | **Test A** | Latency p95 |
|---|---|---|---:|---:|---:|
| `act_original` | 40% / 53% | last (epoch 1999) | **53%** [43, 62] | **0%** [0, 4] | 10 ms |
| `act_clean` | 61% / 98% | last (epoch 1999) | **99%** [95, 100] | **7%** [3, 14] | 10 ms |
| `act_clean_proprio` | 44% / 66% | last (epoch 1999) | **69%** [59, 77] | **1%** [0, 5] | 10 ms |

Failure modes on the source cube (100 validation rollouts of the selected checkpoint):
`act_original` 48 success / 38 never lifted / 11 placed just outside; `act_clean` 99 / 1 / 0;
`act_clean_proprio` 68 / 29 / 1.

![Validation success per epoch (50 rollouts; 100 at epoch 1999) and ACT's validation loss](act_original_curves.png)

## Findings

1. **ACT-like data is solved with images only: 99% on the source cube** (R3 kept, R2's 80% met
   at 300 demos). With random yaw and OCBench's noisy expert the same code reaches 53%: as in
   the ACT paper (scripted vs. human data), demonstration variability matters most.
2. **Changing only the colour breaks every policy (0–7% on target A).** The baseline (a) floor
   is near zero, so the methods that use A's videos have a large gap to close.
3. **Robot state did not help here (69% vs 99%)**, one seed only. Its curve jumps between
   checkpoints (90% at epoch 1400, 54% at 1700), so this needs more seeds before any claim;
   policies that over-rely on proprioception are a known failure (copycat problem).
4. **The lowest-validation-loss checkpoint was always worse** than the last one (40/53, 61/98,
   44/66), and success varies widely between nearby checkpoints while the loss curve is
   smooth: closed-loop selection matters, and 3 training seeds are needed.
5. **R6 holds:** 10 ms per policy call (temporal ensembling queries every step).
6. Operations: Modal preempted `act_clean` once; upstream ACT cannot resume, so the run
   restarted from zero (Modal does not offer non-preemptible GPU containers).

## Decisions (10 November 2026)

- **The ACT-like data (`clean`) is the project data** for every method and object
  (`configs/data.toml`); the stock-data run stays as the reason.
- R2 as in the proposal: 100 demos if they reach 80% on validation, else 300 (running).
- Next: two more training seeds of the chosen size, then the full test.
