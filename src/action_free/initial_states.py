"""Where each episode starts: cube position and yaw, and arm joints.

Every collection has `initial_states.csv` (one row per kept episode), read from the stored
simulator state `qpos[0]`. Training runs keep the initial states of their training episodes
in the checkpoint, and evaluation compares each rollout's start with them, so results show
whether a start lies inside the training distribution (see `evaluate.py`).

The cube is symmetric under 90-degree turns, so `cube_yaw_mod90_deg` is the yaw that
matters for a cube; elongated blocks need the full `cube_yaw_deg`. `num_detours` counts the
expert's deliberate mistakes in the episode (see `collect.count_detours`).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

FIELDS = [
    "file",
    "seed",
    "split",
    "cube_x",
    "cube_y",
    "cube_yaw_deg",
    "cube_yaw_mod90_deg",
    *[f"arm_q{i}" for i in range(6)],
    "success",
    "length",
    "num_detours",
]


def qpos_indices(env) -> tuple[int, np.ndarray]:
    """Start of the cube's free joint in qpos, and the arm's qpos indices."""
    u = env.unwrapped
    return int(u._model.jnt_qposadr[u._model.joint("object_joint_0").id]), np.asarray(
        u._arm_joint_ids
    )


def from_qpos(qpos: np.ndarray, cube_adr: int, arm_idx: np.ndarray) -> dict:
    x, y, _, w, qx, qy, qz = (float(v) for v in qpos[cube_adr : cube_adr + 7])
    yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return {
        "cube_x": x,
        "cube_y": y,
        "cube_yaw_deg": yaw,
        "cube_yaw_mod90_deg": yaw % 90.0,
        **{f"arm_q{i}": float(q) for i, q in enumerate(np.asarray(qpos)[arm_idx])},
    }


def current(env) -> dict:
    """Initial state of a live environment (call right after `reset`)."""
    adr, arm = qpos_indices(env)
    return from_qpos(env.unwrapped._data.qpos, adr, arm)


def write_csv(data_dir: Path, env_name: str, object_name: str) -> Path:
    """Write `initial_states.csv` for every episode file of a collection."""
    from action_free.collect import make_env
    from action_free.data import split_of

    env = make_env(env_name.removeprefix("visual-"), object_name)  # same qpos layout
    adr, arm = qpos_indices(env)
    env.close()
    manifest = Path(data_dir) / "manifest.json"
    detours, role = {}, "train"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        role = m["config"].get("role", "train")
        for e in m["episodes"]:
            if e.get("kept") and "num_detours" in e:
                detours[e["file"]] = e["num_detours"]
    path = Path(data_dir) / "initial_states.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for f in sorted(Path(data_dir).glob("episode_*.npz")):
            with np.load(f) as ep:
                row = {
                    "file": f.name,
                    "seed": int(ep["seed"]),
                    "split": "test" if role == "test" else split_of(int(f.stem.split("_")[1])),
                    **from_qpos(ep["qpos"][0], adr, arm),
                    "success": bool(ep["success"]),
                    "length": int(ep["length"]),
                    "num_detours": detours.get(f.name, ""),
                }
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
    return path


def read_csv(data_dir: Path) -> list[dict] | None:
    path = Path(data_dir) / "initial_states.csv"
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in r:
            if k.startswith(("cube_", "arm_")):
                r[k] = float(r[k])
    return rows
