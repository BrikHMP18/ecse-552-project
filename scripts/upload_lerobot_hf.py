"""Publish each project collection as its own public LeRobot dataset (Hugging Face visualizer).

    uv run --with 'lerobot[dataset]' python scripts/upload_lerobot_hf.py \
        data/source-deterministic-n300-seed0 data/A-deterministic-n200-seed100000

One repository per collection: Autobrik/ecse552-ocbench-<object>[-test]. The .npz copy
(scripts/upload_dataset_hf.py) stays the training data; videos here are lossy.
"""

# ruff: noqa: E501
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from action_free.act_original import CAMERAS
from action_free.data import episode_files

NPZ_REPO = "Autobrik/ecse552-ocbench-block"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
VECTORS = {
    "observation.state": [*JOINTS, "gripper_driver"],  # qpos[0:7]
    "observation.environment_state": [
        f"object_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")
    ],
    "action": [f"d_{j}" for j in (*JOINTS, "gripper")],
}
FEATURES = {
    **{
        f"observation.images.{c}": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
        for c in CAMERAS
    },
    **{k: {"dtype": "float32", "shape": (7,), "names": n} for k, n in VECTORS.items()},
}

DESCRIPTION = """LeRobot copy of `{folder}` from [{npz}](https://huggingface.co/datasets/{npz})
(exact .npz data and full documentation), for the McGill ECSE 552 project *Learning to
Manipulate a New Object from Action-Free Videos*
([code](https://github.com/BrikHMP18/ecse-552-project), commit `{commit}`).

Scripted-expert episodes in OCBench (UR5e + Robotiq, MuJoCo): object `{obj}` ({desc}),
{n} episodes, 10 Hz, three 224 x 224 cameras (front, side, wrist), 7 normalized actions (six
joint increments, one gripper increment). Scene `{scene}`, expert `{expert}`.{note}

`observation.state` (joints) and `observation.environment_state` (object pose) are privileged
simulator state: the project's models use images only. Episode `i` here is
`episode_{{i:05d}}.npz` in the source collection (seed in its `manifest.json`). Videos are
lossy; train from the .npz copy for exact pixels."""
TARGET_NOTE = """

**Target object: in the project these episodes are action-free videos.** `action` is used
only by the true-action oracle and for evaluation."""
TEST_NOTE = """

**Held-out source episodes** (own seed range): only for the inverse dynamics model's test error."""


def publish(d: Path, user: str, work: Path) -> None:
    m = json.loads((d / "manifest.json").read_text())
    cfg = m["config"]
    if (cfg.get("scene"), cfg.get("expert")) != ("fixed_yaw", "deterministic"):
        raise SystemExit(f"{d}: not a project (deterministic) collection")
    prov = dict(
        x.split("=", 1) for x in (d / "provenance.txt").read_text().splitlines() if "=" in x
    )
    if prov["project_commit"].endswith("-dirty"):
        raise SystemExit(f"{d}: made from uncommitted code")
    obj = cfg["object_name"]
    test = cfg.get("role") == "test"
    repo = f"{user}/ecse552-ocbench-{obj}{'-test' if test else ''}"
    root = work / repo.split("/")[1]
    shutil.rmtree(root, ignore_errors=True)
    ds = LeRobotDataset.create(
        repo, fps=10, features=FEATURES, root=root, robot_type="ur5e_robotiq"
    )
    task = f"Pick up the {m['object']['description'].split(' (')[0]} and place it inside the gray square."
    for f in episode_files(d):
        ep = np.load(f)
        for t in range(int(ep["length"])):
            ds.add_frame(
                {
                    **{
                        f"observation.images.{c}": ep["observations"][t, i]
                        for i, c in enumerate(CAMERAS)
                    },
                    "observation.state": ep["qpos"][t, :7],
                    "observation.environment_state": ep["qpos"][t, -7:],
                    "action": ep["actions"][t],
                    "task": task,
                }
            )
        ds.save_episode()
    ds.finalize()
    assert ds.meta.total_frames == m["total_transitions"], (ds.meta.total_frames, m)
    ds.push_to_hub(
        tags=["ecse552", "ocbench", "ur5e", "mujoco", "simulation"],
        license="mit",
        private=False,
        dataset_description=DESCRIPTION.format(
            folder=d.name,
            npz=NPZ_REPO,
            commit=prov["project_commit"][:7],
            obj=obj,
            desc=m["object"]["description"],
            n=m["num_kept"],
            scene=cfg["scene"],
            expert=cfg["expert"],
            note=TEST_NOTE if test else "" if obj == "source" else TARGET_NOTE,
        ),
    )
    print(f"https://huggingface.co/datasets/{repo}  {ds.meta.total_episodes} episodes", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("collections", type=Path, nargs="+")
    p.add_argument("--user", default="Autobrik")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/lerobot"))
    a = p.parse_args()
    for d in a.collections:
        publish(d, a.user, a.work_dir)


if __name__ == "__main__":
    main()
