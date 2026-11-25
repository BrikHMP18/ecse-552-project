# LAPA on A, D and E (h): protocol

24 November 2026, Team. Written before any run: no result exists yet. Every setting below
is fixed in advance and is never chosen from target results (R1).

## Method

Latent action pretraining (LAPA; Seo et al., ICLR 2025), adapted to our setting with latent
pseudo-labels. Videos pretrain a latent action space shared by every object; the policy is
then trained in that space on the source demos plus the target videos, whose supervision is
the latent action the pretrained model assigns to each pair of frames, and a small action
decoder maps latent actions back to the absolute targets ACT uses.

- **Stage 1, latent action model (LAM).** A VQ-VAE over frame pairs: the encoder reads
  (o_t, o_{t+1}) and predicts a discrete latent action z from a codebook of 128 codes; the
  decoder reconstructs o_{t+1} from o_t and z. It pretrains on the videos of the source
  demos and of A, D and E, images only; no action is ever read in this stage (R1, R3). One
  LAM, shared by the three policies.
- **Stage 2, policy and decoder.** The policy is ACT's transformer without the VAE (the
  outputs are discrete): the current frame -> a chunk of 20 latent actions. It trains on the
  300 source demos and the 200 target videos with latent pseudo-labels z = LAM(o_t, o_{t+1})
  (cross-entropy), the targets read as images only (R1). The action decoder g(o_t, z) is its
  own ResNet-18 with an MLP head over the absolute targets; it trains on the source demos
  and the 100 turned-grasp demos only, the role of `idm_turned` in (b). The turned demos
  never enter the policy's z-head.
- **Execution.** The policy is queried every step; the chunk's latent actions are decoded
  against the current frame (g(o_t, z_i) for the i-th chunk step) and the chunks are
  combined with ACT's temporal ensembling, as every ACT run.

## Fixed in advance

| Setting | Value |
|---|---|
| Objects | A, D, E (A first as a pipeline check), and the source for R5 |
| Seeds | 0, 1, 2; LAM seed s, policy seed s and its evaluation form one chain |
| LAM data | the videos of the 300 source demos and of A, D, E (200 each), images only |
| LAM | encoder as the IDM's (ImageNet ResNet-18, 9 channels, 128 x 128); codebook 128, latent dim 64, commitment beta 0.25; decoder shared across cameras; 20 epochs, AdamW lr 3e-4 with cosine decay, weight decay 1e-4, batch 64 |
| LAM checkpoint | lowest validation reconstruction loss (splits of `data.split_of` per collection) |
| Policy data | 300 source demos + 200 target videos (latent labels from the frozen LAM of the same seed); the 100 turned demos train the decoder only |
| Policy | ACT's transformer (224 x 224, 4 encoder + 7 decoder layers, 8 heads, hidden 512, feed-forward 3200), chunk 20, no VAE; 2000 epochs, AdamW lr 1e-5 with cosine decay, weight decay 1e-4, batch 8; split: random 80/20 by episode over source + target (seed 1, as ACT's own split), the turned demos unsplit in train |
| Decoder | own ResNet-18 at 128 x 128 + MLP (hidden 512); absolute targets standardized with the labeled training steps (std floored at 1e-2); action loss weight 1 |
| z labels | argmax of the frozen LAM encoder, computed once per run |
| Execution | selected checkpoint, `ensemble:0.01`, 200-step limit, scene `fixed_yaw` |
| Policy checkpoint | best or last on 100 source validation rollouts (seeds 900,000+), ties to best, as every method |
| Test | 100 rollouts per object and seed on the test seeds (1,000,000+), once per selected checkpoint; mean, std and 95% Wilson intervals |
| R4 and R5 | 95% bootstrap intervals from `requirement_intervals.py` (10,000 replicates); the strict rule of (b) and (e): R4 holds when the whole interval for G_T lies above 0.5, R5 when the whole interval for the source drop lies below 10 points |

## How R1 holds

- **Never read:** the stored actions of the target videos. The LAM and the policy read them
  as images only, and the latent labels come from the images; the export keeps no action
  dataset for them. Target states are never read (R3).
- **Turned demos:** they train the action decoder g only, exactly as they train the IDM in
  (b); the policy's z-head never sees them. They are labeled source data.
- **Fixed above, before any result.** Nothing is selected with target actions or target
  success; test seeds are used once per selected checkpoint.

## Comparison

- **Main table, as for (b) in [idm.md](idm.md):** (h) on A, D, E and on the source, per
  seed, mean ± std, 95% Wilson interval over the 300 rollouts.
- **R4:** G_T = (h - a) / (d - a) per target against the baseline (a) and the oracle (d);
  point estimate and 95% bootstrap interval from `requirement_intervals.py`, the strict
  whole-interval rule.
- **R5:** source drop against (a) (90.7%), same interval and rule.
- **Against (b), per seed:** (b) reaches 83.0% on A, 83.7% on D, 95.0% on E, source
  92.7/93.3/71.0%. The two methods train on the same episodes and differ in the label space
  (explicit IDM pseudo-actions vs latent actions with shared pretraining), in the decoder
  (which also sees the turned demos, as `idm_turned`), and in the policy (no VAE, discrete
  outputs).
- **Decoder error** on the 50 held-out source demos (`scripts/evaluate_decoder.py`, the
  decoder's analogue of the IDM test error): MAE per component, evaluation only.
- **Expectation, written before the runs:** the latent labels are discrete and learned on
  the targets' own videos, so they cannot suffer (b)'s main failure, wrist angles it has
  never seen on E: the LAM has seen E's frames. Whether the discrete bottleneck keeps enough
  detail, and whether the decoder decodes unseen (o, z) pairs, is open; whatever the result,
  it is reported.

## Limits

- The policy and decoder are our own reimplementation of ACT's transformer (without the
  VAE) on the same data and settings; differences from ACT itself are not isolated from the
  method's effect.
- The turned demos reach g only through the LAM's codebook, so the LAM's quality bounds what
  the decoder can use; (b)'s IDM saw the turned demos directly.
- A chunk is decoded against the current frame (g(o_t, z_i) for every chunk step), which
  differs from (b)'s step-local IDM labels; temporal ensembling weights the freshest query
  most, but this is a design choice, not a measured necessity.

## Cost estimate (to update from smoke runs)

Rates as in [bco.md](bco.md): A10G about $1.50/h (32 GiB, 2 cores), T4 about $0.85/h.

| Step | Assumption | Hours | Cost |
|---|---:|---:|---:|
| LAM, 3 seeds | `idm_turned` took about 1 h for 30,400 frames x 20 epochs on an A10G ([idm.md](idm.md)). The LAM adds a decoder and 2.9x the data (about 89,000 frames): about 3 h each | 9 | $13.5 |
| Policies, 9 runs | about 1.3 h each: the ACT-sized transformer and the decoder, 2000 epochs x 75 batches, plus the z labels once per run (about 0.1 h) | 12 | $18 |
| Rollouts, 3,600 | as in [bco.md](bco.md): 2 x 100 selection + 2 x 100 test per run, 9 runs, about 95 steps each at 50-100 ms | 4.8-9.5 | $4.0-8.1 |
| Smoke runs | one small run of each stage | 0.7 | $1.0 |
| **Total** | | | **$36.5-40.6** |
| **With a 20% margin** | | | **$43.8-48.7** |

**Budget rule (cost only, never target results):** after the smoke runs, recompute the
total from the measured speeds; if it is still above what the team has approved, stop and
ask before any further run. The estimate above is above the $30 of [bco.md](bco.md): the
team approves the budget before any paid run.

## Data and models

- Source, target and turned collections: the public dataset pinned in `configs/data.toml`.
- LAM and policy checkpoints: the Modal volume `ecse552-lapa-outputs` while the runs go on;
  published to Hugging Face afterwards (revision recorded here when done).

## Runs

Each stage runs the three seeds in parallel. Smoke runs add `--smoke` (10 episodes, one
epoch) and need no secrets. Launch only with the team's approval.

1. `modal run scripts/modal_train_lapa.py --config configs/lam.toml --smoke`, then
   `--seeds 0,1,2`.
2. `modal run scripts/modal_train_lapa.py --config configs/lapa_A.toml --smoke`, then
   `--seeds 0,1,2`; `modal volume get ecse552-lapa-outputs lapa_A outputs/lapa_A/`.
3. `scripts/select_checkpoint.py outputs/lapa_A/n300/seed*` (source validation rollouts).
4. `scripts/evaluate_policy.py outputs/lapa_A/n300/seed*/selected.ckpt --object A` and
   `--object source` (test seeds, once); `scripts/evaluate_decoder.py` on the held-out
   source demos.
5. Same for D and E; then `scripts/requirement_intervals.py` and the results written here.
