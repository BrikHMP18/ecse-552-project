# Baseline selection: adversarial IDM adaptation for action-free object transfer

*Literature review for ECSE 552 · 8 October 2026 · Recommendations for team review; no experiments performed.*

**Question.** A robot has action-labeled visuomotor data with a cube (A, ~50 demos) and only
action-free videos of the same robot with an elongated block (B). We train an inverse dynamics
model (IDM) on A, align it to B with domain-adversarial training (DANN), pseudo-label B, and train
an image-only BC policy on A + pseudo-labeled B. Which published work is closest, is the NeurIPS
2025 paper (GDA) the right anchor, and which five papers define the comparisons we must run?

**Method of this review.** Six parallel searches (GDA and co-training; IDM pseudo-labeling;
adversarial cross-domain imitation and regression DA; latent-action models; flow/track methods;
newest 2025–2026 work plus novelty check), followed by three verification passes (venues/code/
licences against arXiv metadata, proceedings and GitHub; numbers and data contracts against paper
tables; a red-team critique of the shortlist). About 80 papers screened, about 45 read at
method/data-contract level. Claims corrected during verification are listed at the end.

## Answer in one paragraph

GDA is the right **closest related work for the policy-alignment side**, but not the closest work
overall and not runnable as published: its target domain needs action-labeled demos (the smallest
tested is one demo) and proprioception, which enters the policy, the OT cost and the DTW pairing.
Our target has neither. The closest work for the *pipeline* is IDM pseudo-labeling, formalised by
Morin et al. (ICML 2026), whose stated future work is exactly our case: unlabeled data from a
different environment. No paper found applies domain-adversarial alignment to an IDM to
pseudo-label same-robot, action-free video of a new object; the contribution is real but narrow,
and should be framed as a controlled study, not a new method. The literature also contains three
warnings we must design around: DANN is weak for regression, marginal alignment often fails to beat
plain co-training, and a new object likely changes the action distribution (label shift), under
which invariant features can provably hurt.

## Top 5 papers to compare against

All five are implemented as variants of **one shared pipeline** (same encoder, augmentation, action
chunking, BC head, data splits and model-selection rule). This keeps every comparison controlled
and satisfies the course rule that baselines run on our data. Mandatory controls, not counted in
the five: source-only BC (A only), and an oracle BC trained with the true B actions that OCBench
can generate (evaluation reference only, never used for selection).

### 1. Morin et al., ICML 2026 — unadapted IDM pseudo-labeling (the anchor)

Morin, Byeon, Jolicoeur-Martineau, Lachapelle. *On the Sample Efficiency of Inverse Dynamics
Models for Semi-Supervised Imitation Learning.* ICML 2026. [arXiv 2602.02762](https://arxiv.org/abs/2602.02762).
Code: [idm-ssil index](https://github.com/sachaMorin/idm-ssil), MIT, pointing to three MIT repos
(`slachapelle/idm_ssil_maze`, `sachaMorin/lapo-plus`, `sachaMorin/uva`).

- Shows that IDM labeling followed by BC and video-model-plus-IDM converge to the same policy in
  the limit, and that IDM-based learning is more sample efficient than BC because the true IDM is
  simpler and less stochastic than the expert policy.
- Data contract: few labels (10 Push-T demos; 2 per LIBERO-10 task); unlabeled data is the same
  demos with actions removed. **Section 6 names unlabeled data from a different
  environment/expert as future work** — our setting.
- Role: the λ = 0 arm of our own DANN model, with identical augmentation and training. Every
  result is reported as a delta from it. Implement in-house; the repos are not OCBench-ready.
- Cite with BCO ([Torabi et al. 2018](https://arxiv.org/abs/1805.01954)) and VPT
  ([Baker et al. 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c7008aff45b5d8f0973b23e1a22ada0-Abstract-Conference.html))
  as the origin of the recipe.

### 2. DARE-GRAM, CVPR 2023 — is adversarial alignment the right tool for regression?

Nejjar, Wang, Fink. *DARE-GRAM: Unsupervised Domain Adaptation Regression by Aligning Inverse Gram
Matrices.* CVPR 2023. [arXiv 2303.13325](https://arxiv.org/abs/2303.13325).
Code: [ismailnejjar/DARE-GRAM](https://github.com/ismailnejjar/DARE-GRAM), MIT.

- Aligns the inverse Gram matrices of source and target features (angle and scale) in a
  thresholded subspace, designed for continuous outputs.
- Evidence against plain DANN on regression (summed MAE, ResNet-18, Tables 1–2): dSprites
  source-only 0.498, DANN 0.315, RSD 0.237, DARE-GRAM 0.164; MPI3D 0.377 / 0.283 / 0.205 / 0.160.
  RSD's argument ([Chen et al., ICML 2021](https://proceedings.mlr.press/v139/chen21u.html)) is
  that feature alignment distorts feature scale, on which regression depends.
- Role: IDM-level alignment alternative. A reviewer will ask why a classification-era
  discriminator is used to align an action regressor; this comparison answers it.
- Practical: the loss is a few lines on batch features. DANN, RSD and DD reference
  implementations exist in [thuml Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
  `examples/domain_adaptation/image_regression` (MIT; verified by clone). Running RSD as well is
  optional, not required.
- Caveat: the benchmarks regress 1–3 factors; 7-D action increments are untested. Report
  per-dimension error.

### 3. GDA, NeurIPS 2025 — alignment in the policy instead of the IDM

Cheng, Ma, Chen, Mandlekar, Garrett, Xu. *Generalizable Domain Adaptation for Sim-and-Real Policy
Co-Training.* NeurIPS 2025. [arXiv 2509.18631](https://arxiv.org/abs/2509.18631),
[project](https://ot-sim2real.github.io/). Code: [GaTech-RL2/ot-sim2real](https://github.com/GaTech-RL2/ot-sim2real),
MIT (robomimic fork, Python 3.8, diffusers 0.11, POT).

- Co-trains a Diffusion Policy on source and target with an unbalanced-OT loss on the policy's
  observation encoder. The ground cost combines feature distance with a label term; in code the
  label is either `robot0_eef_pos` or the first action of the chunk. Pairs are sampled with
  DTW over proprioceptive trajectories (pair files are precomputed and downloaded; no DTW script
  in the repo).
- Data contract: target demos are action-labeled (10–25; minimum tested 1, App. G) and
  proprioception is required. Simulated shifts are visual (camera, texture); real-world OOD tests
  also vary shape and reset.
- **Verdict on adequacy.** Appropriate as (a) the most recent top-venue work aligning visuomotor
  features across domains, and (b) the policy-side pole of our second research question. Not
  appropriate as an unchanged baseline: with zero target actions it has no published result.
- Role: implement **GDA's UOT loss inside our policy** (POT `sinkhorn_knopp_unbalanced`, about 30
  lines), with the label cost computed from IDM pseudo-actions, and an observation-only variant
  (`cost_scale = 0`). Do not port the robomimic fork: a different backbone and data pipeline
  would confound the alignment-location comparison.
- Run beside it the simplest policy-side arm: gradient reversal on the BC encoder with B frames,
  as in Kato et al. (ICRA 2026, DOI 10.1109/icra57385.2026.11696866; no arXiv or code found). If
  time permits, add CFG-ADDA from Lei et al. (ICML 2026, [arXiv 2604.13645](https://arxiv.org/abs/2604.13645);
  no code): adversarial alignment plus an explicit domain token.

### 4. LAOM, ICML 2025 — using B without explicit pseudo-actions

Nikulin, Zisman, Tarasov, Lyubaykin, Polubarov, Kiselev, Kurenkov. *Latent Action Learning
Requires Supervision in the Presence of Distractors.* ICML 2025.
[arXiv 2502.00379](https://arxiv.org/abs/2502.00379). Code: [dunnolab/laom](https://github.com/dunnolab/laom),
Apache-2.0, PyTorch, single-file `train_lapo.py`, `train_laom.py`, `train_laom_labels.py`,
`train_idm.py`; about 7 h per run on one H100.

- Latent action model (multi-step IDM plus latent forward model, no VQ, large latent) trained on
  all frames; **LAOM+supervision** adds an action-prediction head on the labeled fraction. About
  2.5% labels give roughly 4× downstream improvement.
- Role: the strongest small-scale representative of the latent-action line (LAPO, LAPA, CLAM,
  CoMo). It uses B self-supervisedly through a forward model rather than adversarially, so it
  competes directly with DANN for the same unlabeled data.
- Warning relevant to us: in its cross-embodiment experiment, labels from a different environment
  did no better than BC. Our labels also come from a different object.
- Why not CLAM ([IROS 2026](https://arxiv.org/abs/2505.04999), [clamrobot/clam](https://github.com/clamrobot/clam)):
  close in spirit and ships VPT and LAPA baselines, but has no licence, no custom-data
  documentation and (contrary to earlier notes) **no CALVIN experiment**. Why not UWM
  ([RSS 2025](https://arxiv.org/abs/2504.02792)): a video-action diffusion transformer, too heavy
  for the budget, and no licence file.

### 5. Mean Teacher, NeurIPS 2017 — non-adversarial semi-supervised use of B

Tarvainen, Valpola. *Mean teachers are better role models: Weight-averaged consistency targets
improve semi-supervised deep learning results.* NeurIPS 2017. [arXiv 1703.01780](https://arxiv.org/abs/1703.01780).
Implementation: TLL `examples/semi_supervised_learning` (also FixMatch, Noisy Student, pseudo-label).

- Consistency between a student IDM and an EMA teacher on augmented B frame pairs; the teacher
  provides the pseudo-labels.
- Role: the baseline an expert reviewer names first. It uses exactly the same unlabeled B frames
  as DANN through a different mechanism. If DANN does not beat it, research question 1 is answered
  "no".
- Cheap extension: filter pseudo-labels by teacher-ensemble disagreement, or by replaying them in
  MuJoCo and comparing the resulting motion to the video, as RoboCurate does
  ([arXiv 2602.18742](https://arxiv.org/abs/2602.18742)). Replay uses simulator access, so report
  it as a diagnostic or clearly marked variant.

### Shared controls the five depend on

- **Strong augmentation for every IDM** (random shift and colour jitter, same crop on both frames;
  [RAD](https://arxiv.org/abs/2004.14990), [DrQ](https://arxiv.org/abs/2004.13649)). Without it, an
  adaptation gain may only reflect missing augmentation.
- **Action chunking in the shared BC head.** OCBench's own paper
  ([Park & Levine, arXiv 2610.07056](https://arxiv.org/abs/2610.07056)) reports that fully
  closed-loop BC often fails without chunking. If every method floors near 0%, no question can be
  answered.
- **Conditional alignment as an ablation row.** CDAN ([Long et al., NeurIPS 2018](https://arxiv.org/abs/1705.10667), in TLL)
  conditions the discriminator on predictions. It is the cheap response to label shift
  ([Zhao et al., ICML 2019](https://arxiv.org/abs/1901.09453)).

## Evidence that shapes the experiment

**Marginal alignment often does not beat plain co-training.** Every row is a verified number.

| Paper | Finding | Source |
|---|---|---|
| Wei et al., IROS 2025 | Adversarial and MMD policy-embedding alignment "do not reliably outperform vanilla cotraining"; a probe separates sim from real at 100% at the observation embedding | [arXiv 2503.22634](https://arxiv.org/abs/2503.22634), Sec. VII-B |
| EgoBridge, NeurIPS 2025 | PushT hardest split: MMD 14%, marginal OT 8%, co-training 31%, joint OT with DTW pairing 39%. No DANN baseline was run. Human data carries hand-position labels | [arXiv 2509.19626](https://arxiv.org/abs/2509.19626), Table 7 |
| Lei et al., ICML 2026 | Real world (of 30): CFG-ADDA 21, co-training 15.3, plain ADDA 14.3, real-only 8.6. Alignment degrades under unbalanced mixing (sim-and-sim only) | [arXiv 2604.13645](https://arxiv.org/abs/2604.13645), Table 2, Sec. 5.2 |
| Watahiki et al., CoLLAs 2024 | Replacing MMD by a domain discriminator in multi-domain BC: R2R-Lift 0.11 vs 0.63, V2V-Open 0.10 vs 0.64 (without TCC; gap shrinks with TCC) | [arXiv 2407.16912](https://arxiv.org/abs/2407.16912), Table 2 |
| MotionTrans, 2025 preprint | Human-vs-robot domain classifier "not beneficial" and "always leads to training instability" | [arXiv 2509.17759](https://arxiv.org/abs/2509.17759), Sec. III-D |

Implications:

1. **Measure pseudo-label error on B directly.** OCBench can generate B's true actions. Use them
   only for evaluation, report per-dimension error (wrist rotation and gripper especially), and
   correlate it with closed-loop success. Few published methods can measure this; it separates
   "alignment improved labels" from "the policy got lucky".
2. **Run the shift diagnostic in weeks 1–2.** Train the IDM on A and compare held-out A error with
   B error. Same robot, cameras and background mean the IDM's shift may be small; then no
   adaptation can win and the study becomes a pre-registered negative result. Decide early.
3. **Fix model selection before looking at B.** Choose λ, checkpoints and thresholds by a source
   rule declared in advance; report a small λ sweep as sensitivity, not as selection. Hidden B
   actions must never select anything.
4. **Statistics.** At least 3 seeds and 50 evaluation episodes per seed; expect overlapping
   intervals among adaptation variants; declare the primary metric in advance.

## Related work: cite, do not run

| Work | Why it matters | Why not a baseline |
|---|---|---|
| ATM, RSS 2024 ([2401.00025](https://arxiv.org/abs/2401.00025), MIT) | Point tracks from action-free video; object-agnostic interface | CoTracker preprocessing, language-conditioned track model, proprio by default; answers a different question |
| LDP, ICML 2025 ([2504.16925](https://arxiv.org/abs/2504.16925)) | Planner on action-free data plus IDM; includes a DP-VPT baseline (DP 0.51, DP-VPT 0.59, LDP+action-free 0.66) | JAX, no licence, same-task action-free data |
| CLAM, IROS 2026 ([2505.04999](https://arxiv.org/abs/2505.04999)) | Continuous latent actions with a jointly trained decoder | No licence or data docs; LAOM covers the line |
| UWM, RSS 2025 ([2504.02792](https://arxiv.org/abs/2504.02792)) | Action-free co-training by masking the action diffusion | Heavy; no licence file |
| CoMo, CVPR 2026 ([2505.17006](https://arxiv.org/abs/2505.17006)) | Pseudo-labels video with latent motion, then co-trains DP | Same shape as ours with latent labels; optional ablation |
| FLARE, CoRL 2025 ([2505.15659](https://arxiv.org/abs/2505.15659)) | New objects learned mostly from video plus 1–10 robot demos | No code (linked repo 404); uses some target labels |
| Video Policy, preprint ([2508.00795](https://arxiv.org/abs/2508.00795)) | Action head trained on 12 of 24 RoboCasa tasks, video on all 24 | 8×A100 for two weeks; CC BY-NC code |
| DreamGen, CoRL 2025 ([2505.12705](https://arxiv.org/abs/2505.12705)) | Same-robot IDM labels generated videos of new behaviours, no adaptation | Cosmos-scale video model |
| EgoBridge, NeurIPS 2025 ([2509.19626](https://arxiv.org/abs/2509.19626)) | Joint OT with action-DTW pairing beats marginal alignment | No code; needs action labels on both sides |
| Giving Robots a Hand ([2307.05959](https://arxiv.org/abs/2307.05959)), AMPLIFY ([2506.14198](https://arxiv.org/abs/2506.14198)) | Close robotic precedents for IDM labeling and video-seen/action-unseen tasks | Covered in `review.md` |
| Third-Person IL ([1703.01703](https://arxiv.org/abs/1703.01703)), XDIO ([2105.10037](https://arxiv.org/abs/2105.10037)), DisentanGAIL ([2103.05079](https://arxiv.org/abs/2103.05079)) | Domain confusion for imitation; XDIO's pipeline is BCO after domain mapping | Require online RL or low-dimensional states |
| ADT, IJRR 2019 ([1709.05746](https://arxiv.org/abs/1709.05746)) | Earliest adversarial adaptation of a visuomotor regressor | Lua/Torch7; ADDA-style variant is optional |

## Novelty statement the team can defend

No paper found (about 15 targeted searches across three independent passes, plus related-work
sections of the closest papers) applies domain-adversarial or other distribution alignment to an
IDM so that same-robot, action-free videos of a new object can be pseudo-labeled for BC. Nearest
neighbours: Morin et al. (no shift), DreamGen and RoboCurate (no adaptation; RoboCurate filters),
Kato et al. and Lei et al. (adversarial alignment of the policy, not an IDM, with nuisance or
sim-to-real shifts), XDIO (state-based mapping then BCO). This is absence of evidence from search,
not a proof. Suggested wording: *"a controlled study of where and how domain alignment helps IDM
pseudo-labeling under an object-geometry shift"*, not *"the first method to learn from action-free
video"*.

## Corrections to earlier notes and agent drafts

- OCBench is now on arXiv: Park & Levine, *Behavioral Cloning Mystery*, [arXiv 2610.07056](https://arxiv.org/abs/2610.07056). Cite this instead of the unversioned PDF.
- CLAM has **no CALVIN experiment** (DMControl, MetaWorld, real WidowX only); its labeled set is a held-out "random-medium" dataset, not play data. Canonical repo is `clamrobot/clam`, no licence.
- LDP's code is JAX, not PyTorch, with no licence. UWM has no licence file.
- FLARE is CoRL 2025 (PMLR v305); its linked code returns 404.
- Wei et al. is IROS 2025; its alignment result is in the main text (Sec. VII-B), and 100% probe accuracy holds only at the observation embedding.
- Lei et al. is an ICML 2026 poster (listed there as *A Mechanistic Understanding of Sim-and-Real Co-Training in Generative Policies*).
- GDA's real-world OOD tests include shape and reset changes, so it is not purely an appearance method.
- TLL includes `rsd.py`, `dann.py`, `dd.py` and `erm.py` for image regression (a red-team claim that RSD is absent was wrong).
- Video Policy reports only 24-task averages (half-task action head 0.41 vs DP 0.21), not separate numbers for video-only tasks.

## Gaps

- No experiment, runtime check or compute estimate was performed.
- Unverified: the Vidar masked-IDM accuracy figures (secondary summary only); whether CFG-ADDA needs target actions in every variant; compute for Morin et al. and LDP.
- The elongated block and its scripted grasp do not exist in OCBench yet; the oracle and B depend on them.
