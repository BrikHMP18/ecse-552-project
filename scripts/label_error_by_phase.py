"""Pseudo-label error per phase of the episode: approach, grasp, carry, release.

    uv run python scripts/label_error_by_phase.py data/E-deterministic-n200-seed500000 \
        data/labels/E-deterministic-idm_turned-seed0 data/labels/E-deterministic-idm_dann_E-seed0

Phases come from the episode's true gripper targets and object height, read for evaluation
only (R1): approach until the gripper is commanded to close, grasp until the object is 1 cm
above its start, carry until the gripper is commanded to open, then release. Prints, per
labels folder and phase, the share of steps and the MAE per component (radians; gripper in
[0, 1]).
"""

import argparse
from pathlib import Path

import numpy as np

from action_free.actions import gripper_opening
from action_free.idm import ACTION_NAMES, absolute_targets, split_files

PHASES = ("approach", "grasp", "carry", "release")
OBJECT_Z = 16  # qpos index of the object's height (free joint from index 14)
MARGIN = 0.05  # gripper target beyond the current opening that counts as a command
LIFT = 0.01  # m above the start height that counts as lifted


def first(mask: np.ndarray) -> int:
    """Index of the first True, or the length if there is none."""
    return int(np.argmax(mask)) if mask.any() else len(mask)


def phases(f: Path) -> np.ndarray:
    """Phase index (0 to 3) of each step of an episode."""
    with np.load(f) as ep:
        qpos = ep["qpos"]
    grip, opening, z = absolute_targets(f)[:, 6], gripper_opening(qpos), qpos[:, OBJECT_Z]
    t = np.arange(len(z))
    close = first(grip > opening + MARGIN)
    lift = first((z > z[0] + LIFT) & (t >= close))
    release = first((grip < opening - MARGIN) & (t >= lift))
    return (t >= close).astype(int) + (t >= lift) + (t >= release)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("collection", type=Path)
    p.add_argument("labels", type=Path, nargs="+")
    a = p.parse_args()
    files = split_files(a.collection)
    phase = np.concatenate([phases(f) for f in files])
    true = np.concatenate([absolute_targets(f) for f in files])
    for labels in a.labels:
        error = np.abs(np.concatenate([np.load(labels / f"{f.stem}.npy") for f in files]) - true)
        print(
            f"{labels.name}\n{'phase':<9} {'steps':>6} " + " ".join(f"{n:>8}" for n in ACTION_NAMES)
        )
        for i, name in enumerate(PHASES):
            e = error[phase == i]
            mae = " ".join(f"{v:8.3f}" for v in e.mean(0))
            print(f"{name:<9} {len(e) / len(error):6.1%} {mae}")


if __name__ == "__main__":
    main()
