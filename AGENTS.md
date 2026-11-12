# Instructions for coding agents

Read [README.md](README.md) first: state of the project, results and next steps.

## Rules

- Work on `main`. Other people and agents may work in the same checkout: touch only the
  files of your task, `git add` file by file (never `git add -A`), `git pull --rebase`
  before every push.
- Before every push: `uv run ruff check src scripts tests`,
  `uv run ruff format --check src scripts tests`, `MUJOCO_GL=egl uv run pytest -q`.
- Short commit messages, no AI attribution trailers.
- Minimal: no unused code, short docs. Code and docs in English.
- `third_party/act` is never edited. Simulator changes go to the OCBench fork
  (`third_party/ocbench`, `master`): commit and push there first, then update the submodule.
- Ask before a design decision. Launch training (Modal) only when a team member says so;
  no local training. Local rendering and rollouts are fine (`MUJOCO_GL=egl`).
- After a task, add one row to [`docs/ai-use-log.md`](docs/ai-use-log.md): what was done
  and how it was verified.

## Protocol (do not change without the team)

- 300 source demos for every policy. `source-turned-n100-seed600000` is for the IDM only.
- Target actions are used only by the oracle (d) and for evaluation (R1).
- Seeds: validation rollouts 900,000+, test rollouts 1,000,000+. Test seeds are used once
  per selected checkpoint (`scripts/evaluate_policy.py --seeds test`); every decision uses
  validation seeds.
- Checkpoint: `scripts/select_checkpoint.py` (best vs last on 100 source validation
  rollouts). 3 training seeds (0, 1, 2); report every seed, mean, std and the Wilson
  interval.
- A result is never dropped or replaced after seeing it; explain it.

## Where things are

- Results and decisions: `docs/experiments/` (`baselines.md`, `augmentation.md`,
  `target_E.md`). Report: `docs/report/main.tex`; parts the team fills are red `\todo{}`.
- Seeds, collections and dataset revision: `configs/data.toml`. One config per run in
  `configs/`.
- Train: `modal run --detach scripts/modal_train.py --config <config> --seeds 0,1,2`
  (`--smoke` first).
- Figures: `uv run --with matplotlib python scripts/make_report_figures.py`. `make report`
  and `make docx` build the documents; `make docx` also rewrites
  `docs/proposal/main.docx`, restore it with `git checkout`.
