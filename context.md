# Project context and handoff

Status on **8 November 2026**. Read this first when you continue the project, with or
without Claude. Details and evidence for every decision are in `README.md` ("Setup
findings for team review"). Plan and requirements come from `docs/proposal/main.tex`.

## One-paragraph summary

A UR5e arm with a Robotiq gripper (OCBench simulation, MuJoCo) learns to move a cube into
a goal square from three RGB cameras (front, side, wrist; 224×224). Labeled
demonstrations exist only for the **red source cube**. For each target object we have
**videos without actions**. An inverse dynamics model (IDM), trained on cube transitions
`(o_t, o_{t+1}) -> a_t` and adapted to the target with DANN, labels those videos, and an
image-only ACT policy trains on cube demos plus pseudo-labeled target videos. Methods
(a)–(f) and requirements R1–R6 are defined in the proposal.

## Done so far

All work is on branch `ccr-43569388-x2usrz` of this repository and on branch `ecse552` of
the OCBench fork (`third_party/ocbench`, pinned to `f0b9f49`).

| Commit | What |
|---|---|
| `614591a` | Rendered camera views (`docs/ocbench_cameras.png`). |
| `86155ec` | Python 3.12 pinned, submodule URL switched to HTTPS, reproducible collector (`src/action_free/collect.py`, `scripts/collect_demos.py`), camera check (`scripts/render_cameras.py`), first tests. |
| `4a6adbd` | Proposal colours: red source cube, blue target A (`src/action_free/objects.py`). |
| `cb99d90` | PyTorch 2.15 and torchvision 0.29 added. |
| `343723f` | 12 cm gray goal square and strict success in every scene. |
| fork `3d22c71` | `BlockEnv(cube_colors=...)`; stock default unchanged. |
| fork `f0b9f49` | `BlockEnv(goal_marker_size=..., success_hold_time=...)`; stock default unchanged. |

Verified:

- `uv sync --locked` installs everything; `uv run pytest` passes 8 tests; ruff is clean.
- Headless rendering gives observations of shape `(3, 224, 224, 3)`.
- Collection is byte-for-byte reproducible per seed. OCBench's own generator is not.
- The same seed gives the **same initial cube and arm state for every object**
  (source and A checked), so evaluation can share initial states across objects and methods.
- Expert with the goal square: 94.5% success over 200 state-only episodes; 5/5 visual
  episodes, 68–94 steps, about 4.4 MB and 28 s per episode on CPU rendering. Estimate for
  300 demos: about 1.3 GB and 2.4 h on CPU (EGL on a GPU should be faster; not measured).
- Stock OCBench behaviour is unchanged by the fork edits (30 seeds compared).

Not done: no dataset has been collected, and `data.py`, `idm.py`, `policy.py`,
`train.py` and `evaluate.py` are still empty stubs. No GPU run has happened yet.

## Decisions already taken

| Topic | Decision | Change from proposal? |
|---|---|---|
| Environment | `visual-block-cpu-lite-single-task1-v0`, CPU MuJoCo | No |
| Control rate | **10 Hz** (`lite`), at most 200 steps (20 s) | Yes: proposal says 50 Hz |
| ACT chunk | **10 actions (1 s)** | Yes: proposal says 25 |
| R6 | Policy inference **< 100 ms per call** | Yes: proposal says < 0.5 s per chunk |
| Colours | Source red cube; A blue cube; B, C red blocks; D blue block | No (OCBench default is blue, so the fork sets it) |
| Goal | Fixed goal at (0.425, 0) marked by a **dark-gray square, 12 cm inside** | New |
| Success | Whole cube inside the square, resting, touching only the table, still, for **1 s** | Yes: stock success fires within 4 cm, often with the cube still in the gripper |
| Data storage | One `episode_XXXXX.npz` per episode, one image per control step | No |
| Python | 3.12 (3.13 cannot install `labmaze`) | — |

The final report must explain every "Yes" above.

## Set up a local checkout

```bash
git checkout main && git pull                 # after this branch is merged
git submodule sync --recursive                # the submodule URL changed to HTTPS
git submodule update --init --recursive       # OCBench at f0b9f49 (fork branch ecse552)
git submodule status                          # no "+" in front of the hash
uv sync --locked
uv run pytest                                 # 8 passed
```

If `git status` shows `third_party/ocbench` as modified, the submodule is on the wrong
commit: run the two `git submodule` commands above. Do not stash it. `uv sync` has only
been tested on Linux; JAX's CUDA 12 extra may not install on macOS or Windows.

## Next steps, in order

Owners follow the proposal's roles table: Brik (data and evaluation), Zheng Ye (IDM),
Myriam (adversarial adaptation), Gonzalo (policy and baselines).

1. **GPU machine check** (Brik). Run `nvidia-smi`: PyTorch 2.15 from PyPI needs driver
   ≥ 580 (CUDA 13.2). If it is older, switch to the CUDA 12.8 wheels from
   `download.pytorch.org`. Then:
   ```bash
   MUJOCO_GL=egl uv run python scripts/render_cameras.py      # expect (3, 224, 224, 3)
   MUJOCO_GL=egl uv run python scripts/collect_demos.py --num-episodes 5 --out-dir data/pilot
   ```
   Record time per episode and disk use.
2. **Fix seed ranges before collecting** (team). Seeds are used in order, failures
   included, so ranges must not overlap. Proposed:
   - source demos: `--base-seed 0`
   - target A videos: `--base-seed 100000`
   - evaluation initial states: seeds from 1 000 000 on, the same for every method and
     object.

   Write the final choice in `configs/`.
3. **Collect data** (Brik). 100 source demos (300 if cube-only ACT scores below 80% on
   the cube) and 200 target-A videos:
   ```bash
   MUJOCO_GL=egl uv run python scripts/collect_demos.py --object source --num-episodes 100 --base-seed 0
   MUJOCO_GL=egl uv run python scripts/collect_demos.py --object A --num-episodes 200 --base-seed 100000
   ```
   Target episodes also store true actions. Only the oracle (d) and evaluation may read
   them (R1).
4. **`src/action_free/data.py`** (Brik, Zheng Ye). Load episode files, split **by
   episode** 80/10/10 with a fixed seed, build `(o_t, o_{t+1}, a_t)` transitions and
   policy chunks of 10, and never return `qpos`/`qvel` or target actions to a model.
   Add tests for split isolation and for those exclusions (`tests/README.md`).
5. **IDM, unadapted** (Zheng Ye): `idm.py` and `train.py`. A CNN shared across the 3
   cameras on `o_t`, `o_{t+1}` and their difference, then an MLP, trained with action MSE.
   Select checkpoints by source validation error only. This gives the labels for method (b).
6. **ACT policy and evaluation** (Gonzalo, Brik): `policy.py` and `evaluate.py`. An
   image-only ACT (CVAE, L1 + KL), chunk 10, no joint inputs. Rollouts use the strict
   success, shared evaluation seeds, 3 seeds × 100 episodes, and also record latency (R6).
   First result: **baseline (a)**, cube-only ACT, on the cube and on target A.
7. **DANN adaptation** (Myriam): method (c). Domain discriminator, gradient reversal and a
   prescheduled λ ramp; λ = 0 must reproduce (b). Then oracle (d), which uses true target
   actions, and augmentation (e).
8. **Targets B, C (and optional D)**. New block geometry and adapted expert grasps go in
   the fork (`BlockEnv`, `ocbench/oracles/`). In `objects.py` they currently raise
   `NotImplementedError`. Validate expert success before collecting.
9. **Report** (`docs/report/main.tex`) last. Sections 1–3, 6, 7 and the AI disclosure
   can be drafted from the proposal at any time.

## Open questions and known gaps

- Pushing the cube into the square also counts as success. Decide whether to require a
  lift during the episode (this matters for policy evaluation, not for expert demos).
- The fork's `cube_colors` and goal square apply to CPU environments only, not MJWarp.
- JAX is installed through OCBench's `train` extra but the project does not use it.
  Dropping that extra would shrink the install.
- Fork housekeeping: `ecse552` is the project branch. `ecse552-cube-colors` is obsolete,
  and the fork's `master` is untouched.
- `README.md` still describes the proposal as a draft in places. Refresh it when the
  report starts.

## Continuing with Claude Code

From the repository root, start `claude` and ask it to read `context.md` and `README.md`
first. Update this file whenever a step above is finished or a decision changes.
