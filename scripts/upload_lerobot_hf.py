"""Publish each project collection as its own public LeRobot dataset (Hugging Face visualizer).

    uv run --with 'lerobot[dataset]' python scripts/upload_lerobot_hf.py \
        data/source-deterministic-n300-seed0 data/A-deterministic-n200-seed100000

One repository per collection: Autobrik/ecse552-ocbench-<object>[-test][-turned]. The .npz copy
(scripts/upload_dataset_hf.py) stays the training data; videos here are lossy.
`--augment-sample N` publishes instead the first N episodes with the image augmentation of
method (e), one random draw per episode, as an illustration (<repository>-augmented-sample).
`--labels NAME` publishes a target with its IDM pseudo-labels `data/labels/<target>-NAME/`
as `action` and the true absolute targets as `true_action`, for viewing only
(<repository>-NAME). `--no-push` builds the dataset in `--work-dir` without publishing it.
"""

# ruff: noqa: E501
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from action_free.act_original import AUGMENT, CAMERAS, augment_images
from action_free.actions import to_absolute
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
TARGETS = [*JOINTS, "gripper_opening"]  # absolute targets, as ACT and the IDM use them
LABEL_FEATURES = {
    **FEATURES,
    **{k: {"dtype": "float32", "shape": (7,), "names": TARGETS} for k in ("action", "true_action")},
}
ACTIONS = "7 normalized actions (six joint increments, one gripper increment)"
LABEL_ACTIONS = "7 absolute targets (six joint positions in radians, the gripper opening in [0, 1])"

DESCRIPTION = """LeRobot copy of `{folder}` from [{npz}](https://huggingface.co/datasets/{npz})
(exact .npz data and full documentation), for the McGill ECSE 552 project *Learning to
Manipulate a New Object from Action-Free Videos*
([code](https://github.com/BrikHMP18/ecse-552-project), commit `{commit}`).

Scripted-expert episodes in OCBench (UR5e + Robotiq, MuJoCo): object `{obj}` ({desc}),
{n} episodes, 10 Hz, three 224 x 224 cameras (front, side, wrist), {actions}.
Scene `{scene}`, expert `{expert}`.{note}

`observation.state` (joints) and `observation.environment_state` (object pose) are privileged
simulator state: the project's models use images only. Episode `i` here is
`episode_{{i:05d}}.npz` in the source collection (seed in its `manifest.json`). Videos are
lossy; train from the .npz copy for exact pixels."""
TARGET_NOTE = """

**Target object: in the project these episodes are action-free videos.** `action` is used
only by the true-action oracle and for evaluation."""
TEST_NOTE = """

**Held-out source episodes** (own seed range): only for the inverse dynamics model's test error."""
AUGMENT_NOTE = """

**Illustration only, not training data.** The first {n} episodes with the image augmentation
of the project's method (e), settings `{settings}`. Here one random draw per episode (seed =
episode index) is applied to its three cameras; training draws a new one for every sample, so
no fixed dataset reproduces it. Actions and states are unchanged."""
LABEL_NOTE = """

**Pseudo-labels, for viewing only: never train on this copy.** `action` is the pseudo-label of
each step from the inverse dynamics model (`{labels}`, IDM checkpoint `{ckpt}`, commit
`{commit}`), what method (b) trains on. `true_action` is the expert's true absolute target of
the same step, here only to compare (in the project, target actions are used only by the
oracle and for evaluation)."""


def publish(
    d: Path, user: str, work: Path, augment: int = 0, labels: str = "", push: bool = True
) -> None:
    m = json.loads((d / "manifest.json").read_text())
    cfg = m["config"]
    if cfg.get("scene") != "fixed_yaw" or cfg.get("expert") not in ("deterministic", "turned"):
        raise SystemExit(f"{d}: not a project (deterministic or turned) collection")
    prov = dict(
        x.split("=", 1) for x in (d / "provenance.txt").read_text().splitlines() if "=" in x
    )
    if prov["project_commit"].endswith("-dirty"):
        raise SystemExit(f"{d}: made from uncommitted code")
    obj = cfg["object_name"]
    test = cfg.get("role") == "test"
    repo = f"{user}/ecse552-ocbench-{obj}{'-test' if test else ''}"
    repo += "-turned" if cfg["expert"] == "turned" else ""
    repo += "-augmented-sample" if augment else ""
    label_dir, info = None, {}
    if labels:
        if obj == "source":
            raise SystemExit(f"{d}: --labels is for target collections")
        label_dir = d.parent / "labels" / f"{d.name.split('-n')[0]}-{labels}"
        info = json.loads((label_dir / "labels.json").read_text())
        if info["collection"] != d.name:
            raise SystemExit(f"{label_dir} labels {info['collection']}, not {d.name}")
        repo += f"-{labels}"
    root = work / repo.split("/")[1]
    shutil.rmtree(root, ignore_errors=True)
    ds = LeRobotDataset.create(
        repo,
        fps=10,
        features=LABEL_FEATURES if labels else FEATURES,
        root=root,
        robot_type="ur5e_robotiq",
    )
    task = f"Pick up the {m['object']['description'].split(' (')[0]} and place it inside the gray square."
    files = episode_files(d)[:augment] if augment else episode_files(d)
    for n, f in enumerate(files):
        ep = np.load(f)
        images = ep["observations"][: int(ep["length"])]  # (T, cameras, H, W, 3) uint8
        if augment:
            torch.manual_seed(n)
            x = torch.from_numpy(images).flatten(0, 1).permute(0, 3, 1, 2) / 255
            x = (augment_images(x) * 255).round().byte().permute(0, 2, 3, 1)
            images = x.reshape(images.shape).numpy()
        actions = ep["actions"]
        if label_dir:
            actions = np.load(label_dir / f"{f.stem}.npy")
            true = to_absolute(ep["actions"], ep["qpos"])
            assert actions.shape == true.shape == (len(images), 7), (f, actions.shape)
        for t in range(len(images)):
            ds.add_frame(
                {
                    **{f"observation.images.{c}": images[t, i] for i, c in enumerate(CAMERAS)},
                    "observation.state": ep["qpos"][t, :7],
                    "observation.environment_state": ep["qpos"][t, -7:],
                    "action": actions[t],
                    **({"true_action": true[t]} if label_dir else {}),
                    "task": task,
                }
            )
        ds.save_episode(parallel_encoding=sys.platform != "darwin")  # its workers crash on macOS
    ds.finalize()
    assert augment or ds.meta.total_frames == m["total_transitions"], ds.meta.total_frames
    if not push:
        print(f"built {root}  {ds.meta.total_episodes} episodes (not published)", flush=True)
        return
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
            n=augment or m["num_kept"],
            actions=LABEL_ACTIONS if labels else ACTIONS,
            scene=cfg["scene"],
            expert=cfg["expert"],
            note=(TEST_NOTE if test else "" if obj == "source" else TARGET_NOTE)
            + (AUGMENT_NOTE.format(n=augment, settings=AUGMENT) if augment else "")
            + (
                LABEL_NOTE.format(
                    labels=f"labels/{label_dir.name}",
                    ckpt=info["idm_checkpoint"],
                    commit=info["commit"][:7],
                )
                if labels
                else ""
            ),
        ),
    )
    print(f"https://huggingface.co/datasets/{repo}  {ds.meta.total_episodes} episodes", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("collections", type=Path, nargs="+")
    p.add_argument("--user", default="Autobrik")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/lerobot"))
    p.add_argument("--augment-sample", type=int, default=0, metavar="N")
    p.add_argument("--labels", default="", metavar="NAME", help="e.g. idm_turned-seed0")
    p.add_argument("--no-push", action="store_true")
    a = p.parse_args()
    for d in a.collections:
        publish(d, a.user, a.work_dir, a.augment_sample, a.labels, not a.no_push)


if __name__ == "__main__":
    main()
