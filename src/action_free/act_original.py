"""Train and run the original ACT (third_party/act, Zhao et al. 2023) on our episodes.

The upstream code is a pinned, unmodified submodule. This module adds only what is needed to
run it on our data, from the outside:

1. `export_dataset`: our episodes -> ACT's HDF5 layout (`/observations/images/<camera>`,
   `/observations/qpos`, `/observations/qvel`, `/action`, attribute `sim`). Actions are
   absolute joint targets (`actions.to_absolute`), as in ACT. `qpos` is all zeros, so the
   policy cannot use robot state (R3: images only), the one deliberate change from ACT;
   `use_robot_state=True` exports the arm joints and gripper opening instead (a reference
   run that measures what R3 costs, never part of the method comparison). A run can train
   on several collections: source demos with their stored actions, plus a target collection
   with either its stored actions (oracle (d)) or pseudo-labels (methods (b), (c)): a
   directory of `episode_XXXXX.npy`, (T, 7) float32 absolute targets predicted by the IDM,
   used instead of the stored actions and states, which are then never read (R1, R3).
2. Upstream assumptions of its bimanual, fixed-length simulation, adapted without editing
   upstream files:
   - state/action dimension is hardcoded to 14: our 7 action dimensions are zero-padded to
     14 (the extra outputs are trivially zero and ignored);
   - all episodes have one length: `get_norm_stats` is replaced by the same statistics over
     concatenated steps, and each sampled action sequence is padded to `MAX_EPISODE_LEN` with
     `is_pad = True`, the mask upstream already applies.
   Method (e) adds image augmentation to the training samples there (`augment_images`).
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

from action_free.actions import ARM_QPOS, gripper_opening, to_absolute

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


# Method (e): image augmentation of the training samples (configs/act_augment.toml).
AUGMENT = {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.5, "shift_px": 4}


def augment_images(images: torch.Tensor) -> torch.Tensor:
    """Colour jitter (hue over the full circle) and a shift of up to `shift_px` pixels, the same
    for the cameras of one sample. `images`: (cameras, 3, H, W) in [0, 1]."""
    from torchvision.transforms import ColorJitter

    jitter = ColorJitter(*(AUGMENT[k] for k in ("brightness", "contrast", "saturation", "hue")))
    s, (h, w) = AUGMENT["shift_px"], images.shape[-2:]
    padded = torch.nn.functional.pad(jitter(images), (s, s, s, s), mode="replicate")
    dy, dx = torch.randint(0, 2 * s + 1, (2,)).tolist()
    return padded[..., dy : dy + h, dx : dx + w]


def robot_state(qpos: np.ndarray, use_robot_state: bool) -> np.ndarray:
    """Upstream's 14-dim `qpos` input: arm joints and gripper opening, or zeros (R3)."""
    qpos = np.atleast_2d(qpos)
    state = np.zeros((qpos.shape[0], STATE_DIM), np.float32)
    if use_robot_state:
        state[:, :6] = qpos[:, ARM_QPOS]
        state[:, 6] = gripper_opening(qpos)
    return state


def _pseudo_labels(labels: Path, f: Path, steps: int) -> np.ndarray:
    """The IDM's absolute targets for one action-free episode: `<labels>/<episode>.npy`."""
    a = np.load(Path(labels) / f"{f.stem}.npy")
    if a.shape != (steps, ACTION_DIM):
        raise ValueError(
            f"{labels}/{f.stem}.npy has shape {a.shape}, expected {(steps, ACTION_DIM)}."
        )
    return a.astype(np.float32)


def export_dataset(
    collections: list[tuple[list[Path], Path | None]], out_dir: Path, use_robot_state: bool = False
) -> list[dict]:
    """Write ACT HDF5 episodes `episode_<i>.hdf5` (i consecutive over the collections, in file
    order) and `episodes.json`. Each collection is (episode files, labels): `labels` is None
    to use the stored actions (source demos; a target only for the oracle (d)), or a directory
    of pseudo-labels (`episode_XXXXX.npy`, (T, 7) float32 absolute targets) used instead."""
    import h5py

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for files, labels in collections:
        if labels is not None and use_robot_state:
            raise ValueError("The robot state of action-free videos is privileged (R3).")
        for f in map(Path, files):
            manifest = f.parent / "manifest.json"
            if (
                manifest.exists()
                and json.loads(manifest.read_text())["config"].get("role") == "test"
            ):
                raise ValueError(f"{f} belongs to a held-out test collection; never train on it.")
            with np.load(f) as ep:
                steps = int(ep["length"])
                obs = ep["observations"][:steps]  # o_t for every action a_t
                seed = int(ep["seed"])
                action = np.zeros((steps, STATE_DIM), np.float32)
                if labels is None:
                    action[:, :ACTION_DIM] = to_absolute(ep["actions"], ep["qpos"])
                    state = robot_state(ep["qpos"], use_robot_state)
                else:  # action-free video: its stored actions and states are never read
                    action[:, :ACTION_DIM] = _pseudo_labels(labels, f, steps)
                    state = np.zeros((steps, STATE_DIM), np.float32)
            i = len(index)
            with h5py.File(out_dir / f"episode_{i}.hdf5", "w") as h:
                h.attrs["sim"] = True
                g = h.create_group("observations")
                images = g.create_group("images")
                for k, cam in enumerate(CAMERAS):
                    images.create_dataset(
                        cam, data=obs[:, k], chunks=(1, *obs.shape[2:]), compression="lzf"
                    )
                g.create_dataset("qpos", data=state)
                g.create_dataset("qvel", data=np.zeros((steps, STATE_DIM), np.float32))
                h.create_dataset("action", data=action)
            index.append(
                {
                    "episode": i,
                    "dir": str(f.parent),
                    "file": f.name,
                    "seed": seed,
                    "length": steps,
                    "labels": None if labels is None else str(labels),
                }
            )
    (out_dir / "episodes.json").write_text(json.dumps(index, indent=2))
    return index


def export(
    files: list[Path], out_dir: Path, use_robot_state: bool = False, labels: Path | None = None
) -> list[dict]:
    """`export_dataset` for one collection."""
    return export_dataset([(files, labels)], out_dir, use_robot_state)


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
            if getattr(self, "augment", False):  # set on the training set only (`train`)
                image = augment_images(image)
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


def describe_split(dataset_dir: Path, split: dict) -> dict:
    """Collections (the first is the source), objects, scenes and training initial states of
    a run, for out-of-distribution checks."""
    from action_free.initial_states import read_csv

    index = json.loads((Path(dataset_dir) / "episodes.json").read_text())
    collections, starts = [], {}
    for d in dict.fromkeys(e["dir"] for e in index):  # unique, in order
        manifest = json.loads((Path(d) / "manifest.json").read_text())
        labels = next(e["labels"] for e in index if e["dir"] == d)
        info = Path(labels) / "labels.json" if labels else None
        collections.append(
            {
                "dir": d,
                "object": manifest["config"]["object_name"],
                "scene_name": manifest["config"].get("scene", "random_yaw"),
                "expert": manifest["config"].get("expert", "stochastic"),
                "scene": manifest.get("object", {}).get("env_kwargs"),
                "num_episodes": sum(e["dir"] == d for e in index),
                "actions": "stored" if labels is None else "pseudo-labels",
                "labels": labels,
                "labels_info": json.loads(info.read_text()) if info and info.exists() else None,
            }
        )
        starts.update({(d, r["file"]): r for r in read_csv(d) or []})
    name = lambda e: f"{Path(e['dir']).name}/{e['file']}"  # noqa: E731
    train = [index[i] for i in split.get("train", [])]
    return {
        **{k: collections[0][k] for k in ("object", "scene_name", "expert", "scene")},
        "source_dir": collections[0]["dir"],
        "collections": collections,
        "train_files": [name(e) for e in train],
        "val_files": [name(index[i]) for i in split.get("val", [])],
        "train_initial_states": [
            [s["cube_x"], s["cube_y"], s["cube_yaw_mod90_deg"]]
            for e in train
            if (s := starts.get((e["dir"], e["file"])))
        ],
    }


def train(
    dataset_dir: Path,
    ckpt_dir: Path,
    num_episodes: int,
    seed: int,
    cfg: dict,
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
        out[0].dataset.augment = bool(cfg.get("augment", False))  # before workers start
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
        "use_robot_state": bool(cfg.get("use_robot_state", False)),
        "augment": AUGMENT if cfg.get("augment", False) else None,
        "num_episodes": num_episodes,
        "split": split,  # upstream's own random 80/20 split of episode indices
        "best_epoch": best["epoch"],
        "best_val_loss": best["val/loss"],
        "last_epoch": cfg["num_epochs"] - 1,
        "data": describe_split(dataset_dir, split),
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
        self.uses_robot_state = bool(run.get("use_robot_state", False))
        t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        self.mean, self.std = t(stats["action_mean"]), t(stats["action_std"])
        self.qpos_mean, self.qpos_std = t(stats["qpos_mean"]), t(stats["qpos_std"])

    @torch.no_grad()
    def predict(self, images: torch.Tensor, qpos: np.ndarray | None = None) -> torch.Tensor:
        """uint8 (B, cameras, H, W, 3) [+ simulator qpos for the reference run] -> absolute
        targets (B, chunk, 7)."""
        x = images.permute(0, 1, 4, 2, 3).float() / 255.0  # upstream: (B, cam, C, H, W) in [0, 1]
        if self.uses_robot_state:
            if qpos is None:
                raise ValueError("This policy was trained with robot state: pass qpos.")
            state = torch.as_tensor(robot_state(qpos, True), device=self.device)
        else:
            state = torch.zeros(x.shape[0], STATE_DIM, device=self.device)  # R3
        state = (state - self.qpos_mean) / self.qpos_std  # upstream normalization
        a_hat = self.policy(state, x) * self.std + self.mean
        return a_hat[..., :ACTION_DIM]
