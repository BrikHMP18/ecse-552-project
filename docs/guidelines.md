# Guidelines for the report and the slides

The report (`docs/report/main.tex`) and the slides (`docs/presentation/main.tex`) tell the
same story, in the same order, with the same names, colours and numbers. Rubrics:
[course requirements](course-requirements.md).

## Model: the instructor's sample talk

The instructor posted a sample talk on myCourses (`Emad_40to50mins.pdf`; keep a copy at
`docs/Emad_40to50mins.pdf`, not in Git: the repo is public). It is 44 slides for 40–50
minutes; take its structure, not its length.

- Title, then an **outline** slide.
- **Background** defines the problem in general terms with a schematic before any detail.
- **Existing methods**, each with its **shortcoming**, then our **idea** in one sentence.
- The **method** is built step by step on one pipeline figure (the same figure, one more
  step highlighted per slide).
- **Datasets**: one slide, each source with its size.
- **Validation design** as a figure before the results (how a number is produced).
- **Results** against every baseline in one table, with a measure of significance (ours:
  mean ± std over seeds and the Wilson interval).
- **Shortcomings**, each with an arrow to what would fix it; then **next steps**.
- **Where to find** the code, data and models; a **Questions** slide; slide numbers.

## Storyline

| # | Part | Question it answers | Rubric |
|---|---|---|---|
| 1 | Motivation | Why care: new objects need new demonstrations, videos are cheap | B1; C1, C3 |
| 2 | Problem | What is given (source demos, target videos), what is sought, R1–R6 | B2, B3; C2, C4 |
| 3 | Data | Which collections, how they are distributed, which method uses which | B6; C5 |
| 4 | Method | ACT once; then, per method, what is trained and what changes | B4, B5; C6, C8 |
| 5 | Evaluation | Splits, checkpoint selection, test, metrics | C7 |
| 6 | Results | Each method against the alternatives and against R1–R6 | B7; C9, C10 |
| 7 | Critique | Constraints met, what went wrong, limitations, advantages; then conclusions | B8; C11 |

## Method: ACT is fixed, the methods change its training data

Explain ACT once, with a figure, then every method as a change to its inputs. No method
changes ACT's architecture or hyperparameters.

- **ACT** (original code, unmodified): conditional VAE; ResNet-18 per camera (front, side,
  wrist; 224 px); transformer encoder 4 layers, decoder 7 layers, 8 heads, hidden 512,
  feed-forward 3,200; latent z of 32 from an action encoder used in training only (z = 0 at
  test); chunk of 20 absolute joint targets; L1 + 10 KL; batch 8, lr 1e-5, 2,000 epochs;
  robot-state input zeroed (images only, R3); temporal ensembling (k = 0.01) at test.
- **IDM** (methods b, c): (o_t, o_t+1) -> a_t. ResNet-18 (ImageNet) per camera at 128 px,
  9 channels (both frames and their difference); camera features concatenated (3 x 512);
  MLP 512 -> 7 standardized absolute targets; MSE; AdamW, 20 epochs, batch 64.
- **DANN** (method c): discriminator MLP 1536 -> 256 -> 1 on the IDM features through a
  gradient reversal layer; each step 64 source + 64 target pairs; lambda ramps to 0.5;
  last epoch kept ([dann.md](experiments/dann.md)).

Use this table in both documents (report Table `tab:methods`, slide "Policy,
alternatives"):

| | Method | What is trained | ACT trains on | What changes against (a) |
|---|---|---|---|---|
| (a) | Source only | ACT | 300 source demos | Nothing (reference) |
| (e) | Augmentation | ACT | 300 source demos, colour jitter and shift | The training images |
| (d) | Oracle | ACT | 300 source + 200 target demos, true actions | Target data with true actions (breaks R1 on purpose: upper bound) |
| (b) | Unadapted IDM | IDM, then ACT | 300 source + 200 target videos, IDM labels | A stage before ACT labels the target videos |
| (c) | DANN-IDM | IDM with discriminator, then ACT | As (b), labels of the adapted IDM | Only how the IDM is trained |

The IDM of (b) and (c) trains on the 300 source demos plus 100 source demos with a turned
grasp; ACT never sees those 100.

## Data

- **Distribution:** object start uniform in a 20 x 20 cm area, yaw 0; fixed arm start;
  3 RGB cameras, 224 x 224, 10 Hz; 7 actions; demos of 70–76 steps (~7 s); success = whole
  object inside the 12 cm square, released and still, within 200 steps.
- **Collections:** one table with object, change, episodes, seeds and *used by*:
  source 300 (all methods), source held out 50 (IDM test), source turned 100 (IDM only),
  A–E 200 each (videos; true actions only for (d) and evaluation), F 200 (IDM label-error
  check only).
- **Splits:** ACT 80/20 by episode (240/60 source only; 400/100 with a target); IDM 8 of
  every 10 source episodes for training; DANN adapts on 160 of the 200 target episodes;
  validation rollouts from seed 900,000, test rollouts from 1,000,000.
- Say in one sentence why the demos are deterministic and have no grasp turns.

## Critique (the ending)

1. R1–R6 per method: met or not, with the number.
2. What went wrong: the first ACT reimplementation (49%), stochastic data (53%), the IDM's
   full-turn wrist errors and the filter, E seed 0 losing the source, non-bit-exact GPU
   rendering.
3. Limitations: simulation only, one task, one scripted deterministic expert; 3 seeds and
   wide intervals; the IDM saw turned grasps in labeled source data, so E tests transferring
   a known motion, not discovering a new one; augmentation settings chosen once.
4. Advantages: no target actions or state; the deployed policy is the unchanged ACT.
5. Conclusions answer the research question; then what we would do next.

## Rules

- Numbers only from `docs/experiments/`; mean ± std over 3 seeds and the Wilson interval;
  no result is dropped.
- Same method letters (a)–(e), object letters and colours (`make_report_figures.py`)
  everywhere.
- Whenever target actions appear, say what for: the oracle or evaluation (R1).
- Slides: at most 15 main slides for 15 minutes, figures over text, slide numbers, every
  member presents.
- Modest claims: a controlled study, not a new method.

## Instructor's feedback on the proposal

| Suggestion | Status (23 November) |
|---|---|
| Confirm first that the IDM infers actions at all | Done: source sMAE 0.023; label error per target ([idm.md](experiments/idm.md)) |
| At 50 Hz frames barely differ: frames further apart or action chunks | Control at 10 Hz (0.1 s between frames) and the IDM is accurate; say so in the report |
| Grasp width (B, C): DANN may erase the cues for the gripper | B and C give no oracle gap; on D, DANN breaks the wrist labels more than the gripper ([dann.md](experiments/dann.md)); discuss |
| Errors per action component | Done (report `tab:idm`) |
| Errors per phase (approach, grasp, lift) | **Missing**: local analysis, no training |
| Run one established action-free method (latent action pretraining, iterative BCO) | **Missing**: (b) is non-iterative BCO; iterative BCO on top of (b) is the cheapest addition (needs a team decision and Modal runs) |
| Confidence intervals for R4 and R5 | **Missing**: Wilson intervals exist for success rates, not for G_T or the source drop; bootstrap, no training |

## Gaps in the drafts (23 November)

- Report: `tab:methods` lacks "what is trained" and "what changes"; ACT has no figure;
  `tab:objects` lacks "used by" and the splits per method; DANN details are a `\todo`.
- Slides: no outline, validation-design or "where to find" slide; no ACT slide or figure;
  methods are bullets, not the table, and no shortcoming of each alternative; the data
  slide has no distribution or splits; 16 main slides; DANN details are a `TODO`.
