"""Train and run the original ACT (third_party/act, Zhao et al. 2023) on our episodes.

The upstream code is a pinned, unmodified submodule. This module adds only what is needed to
run it on our data, from the outside:

1. `export`: our episodes -> ACT's HDF5 layout (`/observations/images/<camera>`,
   `/observations/qpos`, `/observations/qvel`, `/action`, attribute `sim`). Actions are
   absolute joint targets (`actions.to_absolute`), as in ACT. `qpos` and `qvel` are all zeros,
   so the policy cannot use robot state (R3: images only); this is the one deliberate change
   from ACT.
2. Upstream assumptions of its bimanual, fixed-length simulation, adapted without editing
   upstream files:
   - state/action dimension is hardcoded to 14: our 7 action dimensions are zero-padded to
     14 (the extra outputs are trivially zero and ignored);
   - all episodes have one length: `get_norm_stats` is replaced by the same statistics over
     concatenated steps, and each sampled action sequence is padded to `MAX_EPISODE_LEN` with
     `is_pad = True`, the mask upstream already applies.
3. `train`: registers our task in upstream `constants`, records the train/validation episodes
   that upstream's own random 80/20 split picks, mirrors loss histories to JSON (and W&B),
   and calls upstream `imitate_episodes.main` with ACT's arguments.
4. `ActPolicy`: loads a trained checkpoint for our OCBench rollouts (`evaluate.py`) with
   upstream's normalization. It predicts absolute targets, which the low-level controller
   turns into OCBench deltas using the robot state.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from action_free.actions import to_absolute

ACT_ROOT = Path(
    os.environ.get("ACT_ROOT", Path(__file__).resolve().parents[2] / "third_party" / "act")
)
TASK_NAME = "sim_ocbench_cube"  # "sim_" prefix: upstream reads SIM_TASK_CONFIGS
CAMERAS = ("front", "side", "wrist")
STATE_DIM = 14  # hardcoded upstream (two 7-DoF arms)
ACTION_DIM = 7  # ours: six arm joints and the gripper
MAX_EPISODE_LEN = 200  # environment time limit, so the longest possible episode
# Fixed upstream values (imitate_episodes.main) that a loaded policy must match.
UPSTREAM_POLICY = {
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
}


def _import_upstream():
    for p in (ACT_ROOT, ACT_ROOT / "detr"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import constants  # noqa: F401  (upstream module names)
    import utils

    return utils


@contextmanager
def _argv(args: list[str]):
    """Upstream re-parses sys.argv when it builds the model (detr/main.py)."""
    saved = sys.argv
    sys.argv = ["imitate_episodes.py", *args]
    try:
        yield
    finally:
        sys.argv = saved


def _cli(a: dict) -> list[str]:
    out = []
    for k, v in a.items():
        if isinstance(v, bool):
            out += [f"--{k}"] if v else []
        else:
            out += [f"--{k}", str(v)]
    return out


def export(files: list[Path], out_dir: Path) -> list[dict]:
    """Write ACT HDF5 episodes `episode_<i>.hdf5` (i in file order) and `episodes.json`."""
    import h5py

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for i, f in enumerate(files):
        manifest = Path(f).parent / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["config"].get("role") == "test":
            raise ValueError(f"{f} belongs to a held-out test collection; never train on it.")
        with np.load(f) as ep:
            steps = int(ep["length"])
            obs = ep["observations"][:steps]  # o_t for every action a_t
            action = np.zeros((steps, STATE_DIM), np.float32)
            action[:, :ACTION_DIM] = to_absolute(ep["actions"], ep["qpos"])
            seed = int(ep["seed"])
        with h5py.File(out_dir / f"episode_{i}.hdf5", "w") as h:
            h.attrs["sim"] = True
            g = h.create_group("observations")
            images = g.create_group("images")
            for k, cam in enumerate(CAMERAS):
                images.create_dataset(
                    cam, data=obs[:, k], chunks=(1, *obs.shape[2:]), compression="lzf"
                )
            g.create_dataset("qpos", data=np.zeros((steps, STATE_DIM), np.float32))
            g.create_dataset("qvel", data=np.zeros((steps, STATE_DIM), np.float32))
            h.create_dataset("action", data=action)
        index.append({"episode": i, "file": Path(f).name, "seed": seed, "length": steps})
    (out_dir / "episodes.json").write_text(json.dumps(index, indent=2))
    return index


def _patch_variable_length(utils) -> None:
    """Upstream stacks whole episodes; ours end at success, so lengths differ."""
    import h5py

    def get_norm_stats(dataset_dir, num_episodes):
        qpos, action = [], []
        for i in range(num_episodes):
            with h5py.File(Path(dataset_dir) / f"episode_{i}.hdf5", "r") as root:
                qpos.append(root["/observations/qpos"][()])
                action.append(root["/action"][()])
        q, a = np.concatenate(qpos), np.concatenate(action)
        return {  # same statistics as upstream (unbiased std, clipped at 1e-2)
            "action_mean": a.mean(0),
            "action_std": np.clip(a.std(0, ddof=1), 1e-2, np.inf),
            "qpos_mean": q.mean(0),
            "qpos_std": np.clip(q.std(0, ddof=1), 1e-2, np.inf),
            "example_qpos": qpos[-1],
        }

    base = utils.EpisodicDataset
    if getattr(base, "_padded", False):
        utils.get_norm_stats = get_norm_stats
        return

    class PaddedEpisodicDataset(base):
        _padded = True

        def __getitem__(self, index):
            image, qpos, action, is_pad = super().__getitem__(index)
            pad = MAX_EPISODE_LEN - action.shape[0]
            if pad > 0:
                action = torch.cat([action, torch.zeros(pad, action.shape[1])])
                is_pad = torch.cat([is_pad, torch.ones(pad, dtype=torch.bool)])
            return image, qpos, action, is_pad

    utils.get_norm_stats = get_norm_stats
    utils.EpisodicDataset = PaddedEpisodicDataset


def act_args(ckpt_dir: Path, seed: int, cfg: dict) -> dict:
    """The arguments of upstream's CLI (imitate_episodes.py), as its main() receives them."""
    return {
        "eval": False,
        "onscreen_render": False,
        "ckpt_dir": str(ckpt_dir),
        "policy_class": "ACT",
        "task_name": TASK_NAME,
        "batch_size": cfg["batch_size"],
        "seed": seed,
        "num_epochs": cfg["num_epochs"],
        "lr": cfg["lr"],
        "kl_weight": cfg["kl_weight"],
        "chunk_size": cfg["chunk_size"],
        "hidden_dim": cfg["hidden_dim"],
        "dim_feedforward": cfg["dim_feedforward"],
        "temporal_agg": False,
    }


def describe_split(dataset_dir: Path, source_dir: Path, split: dict) -> dict:
    """Object, scene and training initial states of a run, for out-of-distribution checks."""
    from action_free.initial_states import read_csv

    index = json.loads((Path(dataset_dir) / "episodes.json").read_text())
    manifest = json.loads((Path(source_dir) / "manifest.json").read_text())
    starts = {r["file"]: r for r in read_csv(source_dir) or []}
    train_files = [index[i]["file"] for i in split.get("train", [])]
    return {
        "source_dir": str(source_dir),
        "object": manifest["config"]["object_name"],
        "scene": manifest["object"]["env_kwargs"],
        "train_files": train_files,
        "val_files": [index[i]["file"] for i in split.get("val", [])],
        "train_initial_states": [
            [starts[f]["cube_x"], starts[f]["cube_y"], starts[f]["cube_yaw_mod90_deg"]]
            for f in train_files
            if f in starts
        ],
    }


def train(
    dataset_dir: Path,
    ckpt_dir: Path,
    num_episodes: int,
    seed: int,
    cfg: dict,
    source_dir: Path | None = None,
    run_info: dict | None = None,
    wandb_run=None,
) -> dict:
    """Run upstream training on exported episodes; returns what `act_run.json` records."""
    utils = _import_upstream()
    import constants

    constants.SIM_TASK_CONFIGS[TASK_NAME] = {
        "dataset_dir": str(dataset_dir),
        "num_episodes": num_episodes,
        "episode_len": MAX_EPISODE_LEN,
        "camera_names": list(CAMERAS),
    }
    _patch_variable_length(utils)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split = {}
    original_load_data = utils.load_data

    def load_data(*args, **kwargs):
        out = original_load_data(*args, **kwargs)
        split["train"] = sorted(int(i) for i in out[0].dataset.episode_ids)
        split["val"] = sorted(int(i) for i in out[1].dataset.episode_ids)
        return out

    utils.load_data = load_data
    import imitate_episodes

    imitate_episodes.load_data = load_data
    original_plot = imitate_episodes.plot_history
    logged = {"epochs": 0}

    def plot_history(train_history, validation_history, num_epochs, out_dir, seed_):
        original_plot(train_history, validation_history, num_epochs, out_dir, seed_)
        per_epoch = len(train_history) // max(len(validation_history), 1)
        rows = []
        for e, val in enumerate(validation_history):
            tr = train_history[e * per_epoch : (e + 1) * per_epoch]
            row = {"epoch": e, **{f"val/{k}": float(v) for k, v in val.items()}}
            if tr:
                row |= {f"train/{k}": float(np.mean([float(d[k]) for d in tr])) for k in tr[0]}
            rows.append(row)
        (Path(out_dir) / "history.json").write_text(json.dumps(rows))
        if wandb_run is not None:
            for row in rows[logged["epochs"] :]:
                wandb_run.log({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
            logged["epochs"] = len(rows)

    imitate_episodes.plot_history = plot_history
    args = act_args(ckpt_dir, seed, cfg)
    with _argv(_cli(args)):
        imitate_episodes.main(args)

    history = json.loads((ckpt_dir / "history.json").read_text())
    best = min(history, key=lambda r: r["val/loss"])
    record = {
        "act_args": args,
        "chunk_size": cfg["chunk_size"],
        "num_episodes": num_episodes,
        "split": split,  # upstream's own random 80/20 split of episode indices
        "best_epoch": best["epoch"],
        "best_val_loss": best["val/loss"],
        "last_epoch": cfg["num_epochs"] - 1,
        "data": describe_split(dataset_dir, source_dir, split) if source_dir else None,
        **(run_info or {}),
    }
    (ckpt_dir / "act_run.json").write_text(json.dumps(record, indent=2))
    return record


class ActPolicy:
    """A trained upstream ACT policy with the interface `evaluate.rollout` uses."""

    action_mode = "absolute"

    def __init__(self, ckpt_path: Path, device: torch.device):
        _import_upstream()
        from policy import ACTPolicy

        ckpt_path = Path(ckpt_path)
        run = json.loads((ckpt_path.parent / "act_run.json").read_text())
        with open(ckpt_path.parent / "dataset_stats.pkl", "rb") as fh:
            stats = pickle.load(fh)
        a = run["act_args"]
        policy_config = {
            "lr": a["lr"],
            "num_queries": a["chunk_size"],
            "kl_weight": a["kl_weight"],
            "hidden_dim": a["hidden_dim"],
            "dim_feedforward": a["dim_feedforward"],
            "camera_names": list(CAMERAS),
            **UPSTREAM_POLICY,
        }
        with _argv(_cli(a)):
            policy = ACTPolicy(policy_config)
        policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.policy = policy.to(device).eval()
        self.device = device
        self.run = run
        self.cfg = SimpleNamespace(chunk=a["chunk_size"])
        self.mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=device)
        self.std = torch.as_tensor(stats["action_std"], dtype=torch.float32, device=device)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 (B, cameras, H, W, 3) -> absolute targets (B, chunk, 7)."""
        x = images.permute(0, 1, 4, 2, 3).float() / 255.0  # upstream: (B, cam, C, H, W) in [0, 1]
        qpos = torch.zeros(x.shape[0], STATE_DIM, device=self.device)  # R3: no robot state
        a_hat = self.policy(qpos, x) * self.std + self.mean
        return a_hat[..., :ACTION_DIM]
