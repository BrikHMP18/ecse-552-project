# LAPA labeler on A, D and E (h): protocol

28 November 2026, Team. The previous (h) protocol and implementation (a VQ latent action
model of our own design plus an ACT-like latent policy, 24 November) are superseded: they
were not a faithful implementation of the original LAPA (Ye et al., ICLR 2025,
[arXiv:2410.11758](https://arxiv.org/abs/2410.11758), official code
[github.com/LatentActionPretraining/LAPA](https://github.com/LatentActionPretraining/LAPA)),
and nothing of them ever ran. This document fixes the faithful replacement and the matched
comparison with (b) before any run; every setting below is fixed in advance and is never
chosen from target results (R1).

## Audit: what changed and why

| Stage | Original LAPA | Previous (h) | Now (this protocol) |
|---|---|---|---|
| 1. LAQ | C-ViViT (spatial + causal temporal transformers), NSVQ with codebook replacement, codebook 8 x 4 tokens (8^4 = 4096), cross-attention decoder with stop-grad on x_t, window H per fps | ResNet-18 pair encoder, one 128-entry code, additive decoder, H = 1 | **The official model, vendored unedited** (`third_party/lapa`, submodule at commit `46aca51`), scaled down: dim 512, 4 + 4 layers, 128 px, patch 16 (8x8 grid), 9 channels (3 cameras), 8 x 4 codes, H = 6 (0.6 s at 10 Hz) |
| 2. Latent policy | 7B LWM VLM, (image, instruction) -> z | ACT transformer minus VAE, no instruction | **Omitted by decision** (below) |
| 3. Robot actions | replace the z-head with a discretized action head, fine-tune the VLA on labeled demos | separate decoder g(o, z) | **g kept as our extension** (below) |

What was kept from the previous (h): the pseudo-label interface of (b), the split and
checkpoint conventions, the Modal harness shape, the decoder skeleton, the test helpers.
What was discarded: the LAM architecture, the latent policy and its configs (`lam.toml`,
`lapa_*.toml`), the `LapaPolicy` evaluation path.

## Method

The primary experiment keeps everything of (b) and changes only the labeler of the target
videos.

- **Stage 1, latent action quantization (LAQ, faithful).** The official LAQ pretrains on
  the images of the 300 source demos, the 100 turned demos and the 200 videos of A, D and
  E, read as images only (R1, R3); no action is ever read. Training pairs are
  (x_t, x_{t+H}) with H = 6 (0.6 s at 10 Hz, the paper's fps rule); the official recipe
  (Adam lr 1e-4, no weight decay, gradient clip 0.5, NSVQ codebook replacement on the
  official schedule). One LAQ, shared by the labelers of A, D and E. The scale-down is
  cost: **44.8M parameters** (recorded by every run) against **356.4M** for the official
  configuration (counted untrained); the paper's own Fig. 5a ablates model scale.
  Adaptations vs the official code, all listed: 3 cameras as 9 channels, 128 px/dim 512
  instead of 256 px/dim 1024, and the NSVQ constructs on the actual device (the official
  model hardcodes `device='cuda'`; `_patch_nsvq_device` in `src/action_free/lapa.py`,
  vendored files untouched).
- **The labels.** As in the paper, the trained encoder is the inverse-dynamics labeler:
  every adjacent pair (x_t, x_{t+1}) of a target video gets its 4-token code
  z_t = LAQ(x_t, x_{t+1}) (the paper's own labeling procedure, §3.2; H = 1 even though the
  LAQ trained with H = 6). **g(o_t, z_t)** then maps the code to the absolute target a_t.
  g is our extension: the original maps latent to robot actions by fine-tuning the latent
  policy on (image, instruction) pairs, which produces a *policy*, not a per-transition
  label; the labeler needs the observed transition's code decoded to action space, so g is
  trained on the labeled demos only — the 300 source + 100 turned demos, the role of
  `idm_turned` in (b) (turned demos train the labeler, never a policy). g is a ResNet-18
  (per camera, 128 px) + MLP over the NSVQ quantized embedding (4 x 32), regressing the
  7-dim absolute targets ACT trains on (6 joints in rad, gripper opening in [0, 1],
  `actions.to_absolute`'s layout); it conditions on (o_t, z_t) only, no instruction.
- **Why stage 2 is omitted.** The latent policy predicts z from (x_t, instruction); it is
  a forward model and never contributes to the labels in the original either (the LAQ
  encoder is the labeler). There is no end-to-end LAPA policy arm in this experiment
  (team decision, notes/plan-lapa.md); g replaces stages 2-3's role for labeling only.
- **ACT (unchanged from (b)).** `configs/act_lapa_{A,D,E}.toml` clone `act_idm_*.toml` and
  change only the labels folder: 300 source demos + the target's kept videos, upstream
  ACT, 2000 epochs, batch 8, lr 1e-5, chunk 20, upstream's random 80/20 split. The labels
  are the same folder format as (b) (`episode_XXXXX.npy`, (T, 7) float32 + `labels.json`).
- **Matched samples.** The kept episodes are the ones (b) kept per seed: `exclude_like` in
  `[[targets]]` applies `max_label_step = 2.0` to the published IDM labels of the same
  seed (A 195, D 197, E 179 at seed 0), so both labelers train ACT on the same episodes.
  The LAPA labels' own jumps are a diagnostic only (not expected: z is discrete and g
  smooth); if they appear, the team decides (new config, not a silent change).
- **Target ground-truth actions** stay in their existing roles: oracle (d) and evaluation
  only, never read by the labeler (R1). No external pretraining data is used anywhere: the
  official checkpoints (laq_openx.pt, LWM, VQGAN) are not loaded; the LAQ trains on project
  data only.

## Fixed in advance

| Setting | Value |
|---|---|
| Objects | A, D, E (A first as a pipeline check), and the source for R5 |
| Seeds | 0 only (team decision for cost; the protocol prefers 3 and (b) shows high seed variance: A 63/89/97). LAQ 0 -> g 0 -> labels -> ACT 0 form one chain; seeds 1-2 are an approved-later option after the measured cost |
| LAQ data | the images of the 300 source demos, the 100 turned demos and A, D, E (200 each) |
| LAQ | official code, dim 512, 4 + 4 layers, 8 heads, 128 px, patch 16, codebook 8, 4 tokens, quant dim 32, H = 6; 20 epochs, batch 64, Adam lr 1e-4 (no weight decay), clip 0.5; best.pt = lowest validation reconstruction loss (splits of `data.split_of` per collection) |
| g | ResNet-18 (ImageNet) + MLP, obs 128 px, z as the NSVQ embedding; 20 epochs, batch 64, AdamW lr 3e-4 cosine, weight decay 1e-4 (the IDM's recipe); best.pt = lowest validation standardized MAE |
| z labels | the frozen LAQ's 4-token codes on adjacent frames, computed once per run |
| ACT | as (b), `exclude_like` on the published IDM labels, seed 0 |
| Execution | selected checkpoint, `ensemble:0.01`, 200-step limit, scene `fixed_yaw` |
| Selection | best or last on 100 source validation rollouts (seeds 900,000+), ties to best |
| Test | 100 rollouts per object on the test seeds (1,000,000+), once per selected checkpoint; mean, std and 95% Wilson intervals; per-seed against (b)'s seed-0 rows |
| R4 and R5 | 95% bootstrap intervals from `requirement_intervals.py` (10,000 replicates); the strict rule: R4 holds when the whole interval for G_T lies above 0.5, R5 when the whole interval for the source drop lies below 10 points |

## How R1 holds

- **Never read:** the stored actions and states of the target videos. The LAQ reads them
  as images only; the labels come from the images (z) and from g, which trained on labeled
  source demos only.
- **Turned demos:** they train the LAQ (images, unsupervised) and g (actions), exactly as
  they train the IDM in (b); no policy ever trains on them.
- **Fixed above, before any result.** Nothing is selected with target actions or target
  success; test seeds are used once per selected checkpoint.

## Comparison

- **Main table, as for (b) in [idm.md](idm.md):** (h) on A, D and E and on the source, per
  seed (seed 0), alongside (b)'s seed-0 rows and the 3-seed means of (a), (b), (d), (e).
- **R4:** G_T = (h - a) / (d - a) per target against the baseline (a) and the oracle (d);
  point estimate and 95% bootstrap interval, the strict whole-interval rule.
- **R5:** source drop against (a) (90.7%), same interval and rule.
- **Label error** on the held-out source demos (`scripts/evaluate_decoder.py`): g's
  analogue of the IDM test error, MAE per component, evaluation only. Per-target label
  error against the true targets stays evaluation-only too (as `label_error_by_phase.py`
  for (b)).
- **Expectation, written before the runs:** the LAQ has seen the targets' own frames, so
  the codes cannot suffer (b)'s unseen-wrist-angle failure on E; whether the 4096-code
  discrete space and the trained g keep enough detail for a good label is open, and the
  matched kept-set on E (179 episodes) is the same handicap (b) had. Whatever the result,
  it is reported.

## Results

Success: 100 test rollouts per seed (seeds 1,000,000+), the selected checkpoint (best vs
last on 100 source validation rollouts, seeds 900,000+, ties to best); per seed, then
mean ± std and the 95% Wilson interval over the seeds' rollouts. (b) has 3 seeds; (h) runs
seed 0 first (seeds 1-2 pending the team). Test seeds are used once per selected
checkpoint. Training time: the sum of the per-epoch `seconds` of the A10G training loops
(`history.json` of the IDM/LAQ/g runs); for ACT, the training loop's wall time (`train_seconds`
in `act_run.json`; the A and D runs predate that key and use the loop's printed elapsed). The
per-row `seconds` field of the A and D ACT histories is meaningless (a removed bug), not
container wall-clock. Parameters: the trained models per method (labeler side + the ACT
policy; every method's ACT is identical, 66.5M).

| Tested on | IDM for that target (b) | LAPA labeler (h) | Iterative BCO (g) |
|---|---|---|---|
| A | 83.0 ± 17.8% (63 / 89 / 97) | TBD | TBD |
| D | 83.7 ± 11.0% (75 / 80 / 96) | TBD | TBD |
| E | 95.0 ± 2.6% (98 / 94 / 93) | TBD | TBD |
| source, A config | 92.7% (88 / 96 / 94) | TBD | TBD |
| source, D config | 93.3% (96 / 94 / 90) | TBD | TBD |
| source, E config | 71.0% (43 / 88 / 82) | TBD | TBD |
| Parameters | IDM 12.0M + ACT 66.5M | LAQ 44.8M + g 12.0M + ACT 66.5M | as (b): IDM + ACT |
| Training time (A10G) | IDM 0.16 h/seed; ACT not recorded (~1.3 h/run, estimate) | LAQ 2.45 h + g 0.15 h (seed 0); ACT TBD | TBD |

The (b) numbers come from [idm.md](idm.md) (configs `act_idm_{A,D,E}`); the parameter
counts are computed by the code (ACT 66,549,391; IDM 11,985,863; LAQ 44,837,480, recorded
by every LAQ run against the 356,397,424 official-scale reference; g 12,032,583). The IDM
training time is the sum over the published `idm_turned` histories.

## Limits

- One training seed: no cross-seed variance for (h); compared per seed against (b) seed 0.
- The LAQ is scaled down (44.8M vs 356.4M parameters, both recorded); the paper's
  Fig. 5a shows scale helps, so a larger LAQ could label better.
- g sees z as the only transition information (not x_{t+1}), so the labels cannot exceed
  what the codebook keeps; (b)'s IDM saw the frame pair directly.
- The LAQ trains on the target videos and the IDM does not: that asymmetry *is* the method
  (unsupervised learning from the action-free videos, the paper's point) and is disclosed.
- The kept-episode set is inherited from (b)'s filter, so E's worst poses (wrist below
  about -4.5 rad) train neither labeler's ACT.

## Cost estimate (to update from smoke runs)

Rates as in [bco.md](bco.md): A10G about $1.50/h. The LAQ is 44.8M params over about
100,000 frame pairs: a few hours per run, to be measured by the smoke runs. 9 ACT runs of
(b) cost about 1.3 h each, so seed 0 (3 runs) is about $6. Total estimate for the seed-0
chain, smoke included: **about $15-25**; the budget rule applies (after the smoke runs,
recompute from the measured speeds; if above what the team approved, stop and ask).

## Runs

Each stage runs one seed. Smoke runs add `--smoke` (10 episodes, one epoch) and need no
secrets. Launch only with the team's approval.

1. `modal run scripts/modal_train_lapa.py --config configs/laq.toml --smoke`, then
   `--seeds 0`.
2. `modal run scripts/modal_train_lapa.py --config configs/lapa_g.toml --smoke`, then
   `--seeds 0`.
3. `modal run scripts/modal_train_lapa.py --label laq/seed0/best.pt,lapa_g/seed0/best.pt`;
   `modal volume get ecse552-lapa-outputs labels data/labels/`; the folders then live in the
   Modal volume `ecse552-lapa-outputs` under `labels/<name>/` and the ACT stage reads them
   from there (`lapa:` paths in `act_lapa_{A,D,E}.toml`), not from the dataset repository
   (the runner has no write access to the Autobrik org; a later upload to the dataset repo
   is optional archival, byte-identical, and would not change results).
4. `modal run --detach scripts/modal_train.py --config configs/act_lapa_A.toml --seeds 0
   --model-repo <runner>/ecse552-act-lapa-A`, then D and E.
5. `scripts/select_checkpoint.py outputs/act_lapa_A/n300/seed0` (source validation
   rollouts); `scripts/evaluate_policy.py .../selected.ckpt --object A` and `--object
   source` (test seeds, once); `scripts/evaluate_decoder.py outputs/lapa_g/seed0/best.pt`.
6. Same for D and E; then `scripts/requirement_intervals.py` and the results written here.
