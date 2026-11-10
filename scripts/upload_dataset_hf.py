"""Publish collections to a public Hugging Face dataset repository, with a dataset card.

    uv run python scripts/upload_dataset_hf.py data/source-deterministic-n300-seed0 data/A-deterministic-n200-seed100000 \
        --repo Autobrik/ecse552-ocbench-block

Each collection directory is uploaded as a folder of the same name (episodes, manifest,
provenance, packages). The card is regenerated from every collection's manifest and
provenance. Prints the resulting commit, which training pins as the dataset revision.
"""

# ruff: noqa: E501  (markdown table rows in the dataset card)
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

PROJECT = "https://github.com/BrikHMP18/ecse-552-project"
FORK = "https://github.com/BrikHMP18/ocbench"

CARD = """---
license: mit
pretty_name: ECSE 552 OCBench single-block demonstrations
task_categories:
- robotics
tags:
- robotics
- imitation-learning
- inverse-dynamics
- mujoco
- ocbench
size_categories:
- 10K<n<100K
---

# ECSE 552: OCBench single-block demonstrations

Scripted-expert demonstrations for the McGill ECSE 552 project *Learning to Manipulate a New
Object from Action-Free Videos* ([code]({project})). A UR5e arm with a Robotiq gripper
(OCBench, MuJoCo) moves a cube or block into a goal square, seen by three RGB cameras.

Every episode stores images **and** actions. In the project, actions of target objects
(everything except `source`) are hidden from every method except the true-action oracle
and evaluation: those collections stand in for action-free videos.

## Collections

| Folder | Object | Scene / expert | Episodes | Seeds | Expert success | Transitions | Mean length | Code |
|---|---|---|---:|---|---:|---:|---:|---|
{rows}

`source` is the labeled red cube. Targets: A changes only the colour (blue cube); B the
geometry (9 x 6 x 6 cm block, same 6 cm grasp width); C the geometry and the grasp width
(9 x 7 x 6 cm); D is C in blue. The same seed gives the same initial object and arm state
for every object.

## Scene

- Environment `visual-block-cpu-lite-single-task1-v0` from the
  [OCBench fork]({fork}), 10 Hz control, at most 200 steps (20 s).
- Cameras: front, side, wrist, 224 x 224 RGB, one image per control step.
- Actions: 7 normalized values in [-1, 1], six joint increments and one gripper increment.
- Goal: fixed at (0.425, 0) m, marked by a dark-gray square of 12 cm inner side.
  Success requires the whole cube inside the square, resting on the table, released and
  still for 1 s; episodes stop at success.
- Initial state: the cube in a 20 x 20 cm area beside the goal (x 0.32-0.52 m,
  y 0.15-0.35 m); the arm always starts from the same pose (effector at (0.35, 0, 0.30) m,
  yaw 0). Scene `fixed_yaw`: cube yaw 0, like ACT's simulated tasks; `random_yaw`: uniform.
- Expert `deterministic`: OCBench's scripted expert without deliberate mistakes, at a
  fixed speed and without 90-degree grasp turns (since 10 November 2026; earlier revisions
  turned the cube grasp in 30% of episodes); `stochastic`: stock OCBench, with mistakes (`p_mistake` = 0.1 per subgoal:
  some episodes first carry the cube to a random spot, `num_detours`) and a random speed.
- `*-deterministic-*` folders (`fixed_yaw` + `deterministic`) are the project data for every
  method; `*-stochastic-*` (`random_yaw` + `stochastic`) are kept for reference.
- Blocks start with the 9 cm side along x. The expert grasps them across the second side
  (6 or 7 cm), never turned by 90 degrees, and the 7 cm ones 2 cm above the centre (lower,
  the gripper linkage hits the block). Success uses the block's real footprint.
- Only successful expert episodes are kept; failed seeds are skipped (see `manifest.json`).

## Files

Each folder holds `episode_XXXXX.npz` files (numbered in seed order), `manifest.json`
(every attempt with its seed, outcome and detours, plus the scene parameters),
`initial_states.csv` (cube position and yaw, arm joints, split and detours of every
episode), `provenance.txt` (command and commits) and `packages.txt`. One episode with T
steps:

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `observations` | (T+1, 3, 224, 224, 3) | uint8 | front, side, wrist images before each step, plus the final frame |
| `actions` | (T, 7) | float32 | action applied at each step (exactly the values stepped) |
| `rewards`, `masks`, `terminals` | (T,) | | OCBench reward, 1 - terminated, episode end |
| `qpos`, `qvel` | (T, ...) | float32 | simulator state before each step: privileged, never a model input |
| `seed`, `success`, `length` | scalars | | |

`observations[t + 1]` is the result of `actions[t]` applied in state `observations[t]`:
replaying the stored actions from the seed reproduces `qpos` exactly.

```python
import numpy as np
ep = np.load("source-deterministic-n300-seed0/episode_00000.npz")
front, side, wrist = ep["observations"][0]
```

Training collections split per episode, in blocks of ten (8 train, 2 validation), so the
first 100 episodes split 80/20 and keep their split in the first 200 and 300 (`split_of` in
`src/action_free/data.py`). `source-test-*` is a separate held-out collection (own seed
range) for test errors only; policies are tested with simulator rollouts.

## Reproduce

```bash
git clone --recurse-submodules {project}.git && cd ecse-552-project
git checkout <commit from the table> && git submodule update --init --recursive
uv sync --locked
MUJOCO_GL=egl uv run python scripts/collect_demos.py --object source --num-episodes 300 --base-seed 0
```

Each folder's `provenance.txt` has the exact command and commits. Folders were renamed on
10 November 2026 (`clean` -> `deterministic`, unlabeled -> `stochastic`; see `provenance.txt`);
earlier commands ran with the collector defaults of their commit.

Simulator state is bit-exact per seed; GPU (EGL) images can differ from the stored ones in
about 0.03% of pixels (shadow edges).

## License and credit

MIT, like [OCBench](https://github.com/seohongpark/ocbench) (arXiv 2610.07056), the
benchmark this data is generated with.
"""


def collection_row(d: Path) -> tuple[str, str]:
    m = json.loads((d / "manifest.json").read_text())
    seeds = [e["seed"] for e in m["episodes"]]
    prov = dict(
        line.split("=", 1)
        for line in (d / "provenance.txt").read_text().splitlines()
        if "=" in line
    )
    row = (
        f"| `{d.name}/` | {m['config']['object_name']}: {m['object']['description']}"
        f"{' (held-out test)' if m['config'].get('role') == 'test' else ''} "
        f"| {m['config'].get('scene', 'random_yaw')} / {m['config'].get('expert', 'stochastic')} "
        f"| {m['num_kept']} | {min(seeds)}-{max(seeds)} | {m['expert_success_rate']:.1%} "
        f"| {m['total_transitions']} | {m['mean_length']:.1f} "
        f"| `{prov['project_commit'][:7]}` |"
    )
    return row, prov["project_commit"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("collections", type=Path, nargs="+")
    p.add_argument("--repo", default="Autobrik/ecse552-ocbench-block")
    a = p.parse_args()

    rows, commits = zip(*(collection_row(d) for d in a.collections))
    if any(c.endswith("-dirty") for c in commits):
        raise SystemExit(f"Collections made from uncommitted code: {commits}")
    card = CARD.format(rows="\n".join(rows), project=PROJECT, fork=FORK)

    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=False, exist_ok=True)
    for d in a.collections:
        print(f"uploading {d} ...", flush=True)
        api.upload_folder(
            repo_id=a.repo,
            repo_type="dataset",
            folder_path=str(d),
            path_in_repo=d.name,
            commit_message=f"Add {d.name}",
        )
    info = api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=a.repo,
        repo_type="dataset",
        commit_message="Dataset card",
    )
    print(f"https://huggingface.co/datasets/{a.repo}  revision {info.oid}")


if __name__ == "__main__":
    main()
