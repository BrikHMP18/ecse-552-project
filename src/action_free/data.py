"""Episode files, episode-level splits and detour annotations of a collection.

Splits are made per episode, never per transition. Episodes are assigned in consecutive
blocks of ten (files are numbered in seed order): within every block a fixed permutation
sends eight episodes to train and two to validation, so the first 100 episodes split exactly
80/20 and keep their split in the first 200 and 300. This split is recorded in
`initial_states.csv` and is the one the IDM uses; the ACT baselines use upstream ACT's own
random 80/20 split (`act_original.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BLOCK = 10
SPLIT_SEED = 552
# Positions within a block of ten: 8 train, 2 validation.
_BLOCK_LAYOUT = np.array(["train"] * 8 + ["val"] * 2)


def split_of(index: int) -> str:
    """Split of the episode with this file index (`episode_XXXXX.npz`)."""
    block, pos = divmod(index, BLOCK)
    perm = np.random.default_rng([SPLIT_SEED, block]).permutation(BLOCK)
    return str(_BLOCK_LAYOUT[perm[pos]])


def episode_files(data_dir: Path, num_episodes: int | None = None) -> list[Path]:
    """The first `num_episodes` kept episodes of a collection, in file order."""
    files = sorted(Path(data_dir).glob("episode_*.npz"))
    if num_episodes is not None:
        if len(files) < num_episodes:
            raise ValueError(f"{data_dir} has {len(files)} episodes, {num_episodes} requested.")
        files = files[:num_episodes]
    if num_episodes is not None and num_episodes % BLOCK:
        raise ValueError(f"Use a multiple of {BLOCK} episodes so splits stay 80/20.")
    return files


def detour_files(data_dir: Path) -> set[str]:
    """Episode files in which the expert made a deliberate detour (initial_states.csv)."""
    from action_free.initial_states import read_csv

    rows = read_csv(data_dir) or []
    return {r["file"] for r in rows if str(r.get("num_detours", "")) not in ("", "0")}
