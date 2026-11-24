# Iterative BCO on D (g): protocol

23 November 2026, Myriam. Written before any run: no result exists yet. Every setting below
is fixed in advance and is never chosen from target results (R1).

## Method

BCO(alpha) (Torabi et al., 2018) on top of (b), which is non-iterative BCO: the IDM learns
from the agent's own interaction, labels the target videos, and the policy learns from the
labels; then the policy acts, its own transitions go back into the IDM, and the loop
repeats. The 400 labelled source demos play the part of BCO's pre-demonstration
interaction.

- **Round 0 = (b)**, already done: IDM `idm_turned` and policy `act_idm_D`, seeds 0, 1, 2.
- **Round r = 1, 2**, for each seed s separately:
  1. **Rollouts.** Policy pi_{r-1}, seed s, acts on target D in simulation for 200 episodes.
     Recorded: the three camera images, the action actually sent to the robot and the
     robot's joint positions before each step.
  2. **IDM.** IDM_{r-1}, seed s, is fine-tuned on the 400 source demos plus every rollout
     of seed s so far (round 2: rounds 1 and 2). A rollout step is labelled
     `to_absolute(action, qpos)`, as the source demos are.
  3. **Labels.** IDM_r labels the 200 D videos (`export_labels.py`).
  4. **Policy.** pi_r, seed s, is trained as `act_idm_D` with these labels.
  5. **Checkpoint.** `select_checkpoint.py` picks pi_r's checkpoint on source validation
     rollouts.

Method (g) is pi_2. Each round adds about 0.6 of the source demos' transitions (alpha
about 0.6; the measured value is reported).

## Fixed in advance

| Setting | Value |
|---|---|
| Target | D only |
| Rounds | 2 |
| Seeds | 0, 1, 2; IDM seed s, its rollouts and policy seed s form one chain |
| Rollouts per round and seed (N) | 200, every one kept: no filter on success or anything else |
| Rollout seeds | round r: 800,000 + 200 (r - 1) + i, i = 0 to 199; the same for the three seeds |
| Rollout execution | as the evaluation: selected checkpoint, temporal ensembling `ensemble:0.01`, 200-step limit, ends at success, scene `fixed_yaw` |
| IDM | fine-tuned from the previous round's IDM of the same seed (round 1: `idm_turned`); 5 epochs, AdamW lr 1e-4 with cosine decay, weight decay 1e-4, batch 64, image size 128; target standardization kept from `idm_turned` |
| IDM checkpoint | lowest sMAE on the validation episodes (`data.split_of`) of the source demos and rollouts, as for `idm_turned` |
| Policy | as `act_idm_D`: 300 source demos + 200 D videos with the round's labels, label filter 2 rad, 2000 epochs, trained from scratch |
| Policy checkpoint | best or last on 100 source validation rollouts (seeds 900,000+) |
| Test | pi_1 and pi_2: 100 rollouts on D and 100 on the source (seeds 1,000,000+), once each |

Fine-tuning instead of retraining from scratch is fixed by the budget (see Cost). It is
also BCO(alpha)'s own update. Its 5 epochs and lr 1e-4 are not tuned.

## How R1 holds

- **Departure from the proposal, agreed by the team on 23 November:** the policy acts on
  the target object in simulation during training, as BCO assumes. The report says so.
- **Never read:** the stored actions of the 200 D videos. The IDM reads them as images
  only. Their true actions are used only to measure label error, which is reported and
  never used to decide.
- **IDM labels on rollouts:** the action our own policy sent to the robot, made absolute
  with our robot's joint positions, exactly as for the source demos. The policy and the
  IDM still see images only (R3).
- **Fixed above, before any result:** the rounds, N, every setting and both checkpoint
  rules. The rollouts' success is recorded and reported only.
- **Test seeds** are used once per selected checkpoint (pi_1 and pi_2).

## Comparison

- **Main table, as for (b) in [idm.md](idm.md):** pi_2 on D and on the source. Per seed, mean
  ± std, 95% Wilson interval over the 300 rollouts.
- **R4:** share of the oracle's gain on D recovered, (g - a) / (d - a), with a = 0.0% and
  d = 92.7%.
- **R5:** source drop against (a) (90.7%).
- Intervals for R4 and R5 come from `requirement_intervals.py`.
- **Against (b), per seed**, the same chain: D 75 / 80 / 96% (83.7%), source 96 / 94 / 90%.
  Augmentation (e), 66.7% on D, is shown for reference.
- **Per round**, round 0 = (b): test success of pi_1 and pi_2.
- **Label error on the 200 D videos** (evaluation only), each round's IDM: sMAE, wrist3 and
  gripper, per phase (`label_error_by_phase.py`), and the episodes the filter drops. Also
  the IDM's error on the 50 held-out source demos.
- **Rollouts:** success, mean length, number of transitions, alpha.
- **Expectation, written before the runs:** (b)'s main error on D is the gripper in grasp and
  carry (0.08 and 0.12; [idm.md](idm.md)), states the source demos do not show. Rollouts on D
  show them, so that error should fall, and D success should lie between (b) and the oracle.
  Whatever the result, it is reported.

## Limits

- The fine-tuned IDM gets extra training that (b)'s IDM did not, so part of any gain over (b)
  could come from longer training.

## Cost estimate (to update from smoke runs)

**Assumed rates (check modal.com/pricing before launching):** A10G $1.10/h, T4 $0.59/h,
CPU about $0.05 per core-hour, memory about $0.008 per GiB-hour. Training containers use an
A10G, 32 GiB and 2 cores: about **$1.50/h**. Rollout containers use a T4, 16 GiB and 2 cores:
about **$0.85/h**. The T4 only makes rollouts cheaper; the policy is the same.

| Step | Assumption | Hours | Cost |
|---|---|---:|---:|
| IDM, fine-tuned, 6 jobs | `idm_turned` took about 1 h for 30,400 frames x 20 epochs ([idm.md](idm.md)). Time scales with frames x epochs. Rollouts add about 19,000 frames per round (200 x 95 steps), so 5 epochs take 0.4 h (round 1) and 0.56 h (round 2), plus 0.1 h for loading and labelling | 3.5 | $5.2 |
| Policies, 6 runs | 1.15 h each: 2000 epochs x 50 batches = 100,000 steps, as the baseline (66-69 min at 26 it/s, [act_baseline_a.md](act_baseline_a.md)); data export on CPU, negligible | 6.9 | $10.4 |
| Rollouts, 3,600 | Per seed: 2 x 200 BCO, 2 x 200 checkpoint selection, 2 x 200 test. About 95 steps each (84% succeed in about 70 steps, the rest run 200), so 342,000 steps at 50-100 ms each (policy 10 ms, R6; simulator and three cameras assumed 40-90 ms) | 4.8-9.5 | $4.0-8.1 |
| Smoke runs | One small run each of IDM, policy and rollouts | 0.7 | $0.9 |
| **Total** | | | **$20.5-24.6** |
| **With a 20% margin** | | | **$24.6-29.5** |

**Retraining the IDM from scratch instead** (20 epochs on all the data: 1.7 h in round 1,
2.35 h in round 2) would cost 12.2 h and $18.2 for the IDM alone. The total would be
$33.5-37.6, above $30 even without a margin. Hence the fine-tuning above.

**Budget rule (cost only, never target results):**
- After the smoke runs, recompute the total from the measured speeds.
- If the total with a 20% margin is above $30, drop the test of pi_1 first (saves
  $0.7-1.4). If it is still above, stop and ask the team before any run.
- During the runs, check the Modal spend after every step. Stop after the current step if
  spend plus the remaining estimate exceeds $30.

## Data

- **Source collections:** the public dataset pinned in `configs/data.toml`.
- **Rollouts, labels and IDM checkpoints:** a Modal volume while the runs go on, published
  to the public dataset `md373/dataBCO` after each round. The revision is recorded here.
- **Policies:** pushed to the user's own Hugging Face account.

## Runs

Each step runs the three seeds in parallel. `bco:` paths are in the volume `ecse552-bco`
(`action_free/bco.py`). Smoke runs add `--smoke`, use 2 rollouts or 10 episodes, and need
no secret: `ECSE552_MODAL_SECRETS= modal run ... --smoke`.

1. Round 0, copied into the volume:
   - the (b) policies: `modal_rollouts.py --step fetch --repo Autobrik/ecse552-act-idm-D
     --prefix n300/seed{seed} --files selected.ckpt,act_run.json,dataset_stats.pkl
     --out bco:policies/act_idm_D/seed{seed}`
   - the `idm_turned` checkpoints, into `bco:idm/idm_turned/seed{seed}/best.pt`.
     **Open:** they are only in Brik's Modal volume, not on Hugging Face.
2. Round r:
   1. `modal_rollouts.py --step collect --round r --policy <pi_{r-1}> --out bco:rollouts/D-r<r>-seed{seed}`.
      pi_0 is `bco:policies/act_idm_D/seed{seed}/selected.ckpt`; pi_1 is
      `out:act_bco_D_r1/n300/seed{seed}/selected.ckpt`.
   2. `modal_train_idm.py --config configs/idm_bco_D_r<r>.toml --seeds 0,1,2`: fine-tunes,
      then labels D into `bco:labels/`.
   3. `modal_train.py --config configs/act_bco_D_r<r>.toml --seeds 0,1,2 --model-repo <own repo>`.
   4. `modal_rollouts.py --step select --policy out:act_bco_D_r<r>/n300/seed{seed}`, then
      `--step test --object D` and `--step test --object source` on `.../selected.ckpt`.
   5. Publish the round's rollouts and labels to `md373/dataBCO`.
