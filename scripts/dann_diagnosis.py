"""Where the DANN labels go wrong: the wrist offset and the role of batch norm.

    uv run python scripts/dann_diagnosis.py offset data/D-deterministic-n200-seed400000 \
        data/labels/D-deterministic-idm_turned-seed0 data/labels/D-deterministic-idm_dann_D-seed0
    uv run python scripts/dann_diagnosis.py batchnorm data/D-deterministic-n200-seed400000 \
        outputs/idm_turned/seed0/best.pt outputs/idm_dann_D/seed0/best.pt

offset: per labels folder, the wrist3 error averaged over each episode (mean and std across
episodes), the share of steps off by more than 0.5 rad that are within 0.3 rad of a full turn,
and the correlation between labels and true wrist3.

batchnorm: re-estimates the batch-norm running statistics (no gradient, weights unchanged) on
source pairs, target pairs, or mixed 64 + 64 batches as in DANN training, then labels the
target. If one of them brings the error back to that of (b), the statistics were the cause;
if none does, the weights. The source pairs are the train split of the checkpoint's
collections (every 8th episode), the target pairs the target's train split (every 4th) as
images only; the error is on all target episodes and on the held-out source demos.

True target actions are read for evaluation only (R1).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from action_free.idm import (
    TransitionDataset,
    absolute_targets,
    errors,
    load_idm,
    load_transitions,
    predict_transitions,
    select_device,
    split_files,
)

WRIST3 = 5
SOURCE_TEST = "data/source-deterministic-test-n50-seed50000"


def offset(target: Path, label_dirs: list[Path]) -> None:
    files = split_files(target)
    truth = [absolute_targets(f)[:, WRIST3] for f in files]
    for d in label_dirs:
        labels = [np.load(d / f"{f.stem}.npy")[:, WRIST3] for f in files]
        per_episode = np.array([np.mean(p - y) for p, y in zip(labels, truth)])
        e = np.concatenate(labels) - np.concatenate(truth)
        far = np.abs(e) > 0.5
        turns = np.round(e / (2 * np.pi))
        full_turn = far & (turns != 0) & (np.abs(e - 2 * np.pi * turns) < 0.3)
        corr = np.corrcoef(np.concatenate(labels), np.concatenate(truth))[0, 1]
        print(
            f"{d.name}: MAE {np.abs(e).mean():.3f}; per-episode offset "
            f"{per_episode.mean():+.3f} ± {per_episode.std():.3f}; steps > 0.5 rad "
            f"{100 * far.mean():.1f}%, full turns among them "
            f"{100 * full_turn.sum() / max(far.sum(), 1):.0f}%; correlation {corr:.2f}"
        )


@torch.no_grad()
def reestimate(model: torch.nn.Module, sets: list, device: torch.device) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all batches
    model.train()
    loaders = [
        DataLoader(TransitionDataset(s), batch_size=64, shuffle=True, drop_last=True) for s in sets
    ]
    for batches in zip(*loaders):
        obs = torch.cat([b["obs"] for b in batches]).to(device)
        next_obs = torch.cat([b["next_obs"] for b in batches]).to(device)
        model.features(obs, next_obs)
    model.eval()


def batchnorm(target: Path, checkpoints: list[Path]) -> None:
    torch.manual_seed(0)
    device = select_device()
    test = {
        "target": load_transitions(split_files(target)),
        "source": load_transitions(split_files(Path(SOURCE_TEST))),
    }
    target_pairs = load_transitions(split_files(target, "train")[::4], labeled=False)
    for c in checkpoints:
        collections = torch.load(c, map_location="cpu")["config"]["collections"]
        files = [f for col in collections for f in split_files(Path(col), "train")[::8]]
        source_pairs = load_transitions(files, labeled=False)
        modes = {
            "as trained": None,
            "source": [source_pairs],
            "mixed": [source_pairs, target_pairs],
            "target": [target_pairs],
        }
        for mode, sets in modes.items():
            model = load_idm(c, device)
            if sets:
                reestimate(model, sets, device)
            std = model.target_std.cpu()
            e = {
                k: errors(predict_transitions(model, t, device), t.targets, std)
                for k, t in test.items()
            }
            print(
                f"{c.parent.parent.name}/{c.parent.name}, batch norm {mode}: target sMAE "
                f"{e['target']['smae']:.3f}, wrist3 {e['target']['mae']['wrist3']:.3f}, gripper "
                f"{e['target']['mae']['gripper']:.3f}; source sMAE {e['source']['smae']:.3f}",
                flush=True,
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("analysis", choices=("offset", "batchnorm"))
    p.add_argument("target", type=Path, help="target collection")
    p.add_argument("inputs", type=Path, nargs="+", help="labels folders or IDM checkpoints")
    a = p.parse_args()
    (offset if a.analysis == "offset" else batchnorm)(a.target, a.inputs)


if __name__ == "__main__":
    main()
