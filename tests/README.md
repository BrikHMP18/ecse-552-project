# Project tests

Add tests alongside implemented behavior, especially episode boundaries,
image/action alignment, split isolation, and exclusion of target actions and
privileged state from learned-model inputs.

Run `uv run pytest`. `test_collect.py` checks that collection is reproducible per seed
and that episode arrays are aligned; it uses the state-based environment, so it needs
no rendering.
