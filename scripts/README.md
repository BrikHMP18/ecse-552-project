# Experiment entry points

Collection, training, and evaluation commands will live here. Keep scripts thin:
reusable logic belongs in `src/action_free/`, and parameters belong in `configs/`.
Run commands from the project root with `uv run python scripts/<script>.py`.

Write collected datasets to `data/` and checkpoints, logs, and evaluation outputs
to `outputs/`. Neither directory is tracked by Git. No experiment commands are
implemented yet.
