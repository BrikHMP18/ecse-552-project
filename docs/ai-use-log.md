# Generative-AI use log

The course allows generative AI in project deliverables only if it follows
McGill's research guidance and is **explicitly disclosed and justified**
(<https://www.mcgill.ca/gps/files/gps/using_generative_ai_in_research_gps_guidelines_november_2024.pdf>).
Add a row whenever AI materially affects the work; the final disclosure in each
deliverable must match this log.

| Date | Member | Tool | Purpose | Material affected | Human verification / limits |
|---|---|---|---|---|---|
| 2026-10-07 to 2026-11-09 | Brik | Codex, Claude Code | Literature review and baseline selection; proposal template, formatting and draft text revised by the team; repository and documentation setup; simulation changes in the OCBench fork (cube colours, goal square with strict success, start area, fixed arm pose); seeded data collection, exact replay check, viewer, Hugging Face publishing (npz and LeRobot copies); evaluation protocol; training and evaluation infrastructure (Modal, W&B); a first ACT reimplementation and its diagnostics, then a thin adapter that runs the original ACT unmodified; tests | `docs/research/`, `docs/references.bib`, `docs/proposal/`, `docs/report/`, `docs/experiments/`, `README.md`, `src/`, `scripts/`, `configs/`, `tests/`, `third_party/ocbench` (fork) | Brik directed the work, made the design decisions (scene, data sizes, protocol, ACT settings) and approved every paid run. Checks: automated tests, exact replay of stored episodes, the re-run expert reproducing all stored actions, end-to-end training and evaluation tests; reported numbers come from logged runs. The team must verify scientific claims and citations before submission. |

The proposal's disclosure also names ChatGPT (literature search, prose). Earlier per-session
entries were condensed into the row above when the repository history was reset (9 November
2026); add a new row for each later material use.
| 2026-11-09 | Brik | Claude Code | Overnight run Brik approved: ACT-like data variant (fixed cube yaw, deterministic expert) and a robot-state reference option; three original-ACT runs on Modal; validation selection, tests, failure diagnosis, learning curves and write-up | `third_party/ocbench` (fork `54783cb`), `src/action_free/`, `configs/`, `scripts/`, `docs/experiments/act_original.md`, HF `Autobrik/ecse552-act-original`, `-act-clean`, `-act-clean-proprio` | Numbers come from the logged rollouts (selection.json, results.json); one training seed per run, so the conclusions are provisional. |
