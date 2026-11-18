"""Label error of an IDM checkpoint on held-out source demos and target videos.

    uv run python scripts/evaluate_idm.py outputs/idm/seed0/best.pt
    uv run python scripts/evaluate_idm.py outputs/idm/seed0/best.pt \
        --collections data/E-deterministic-n200-seed500000

The model sees images only (`labeled=False`); each episode's true absolute targets are read
separately, only to measure the error (R1). Writes label_error.json next to the checkpoint:
per collection, sMAE (standardized MAE over the moving components), MSE of standardized
targets, and MAE per component (radians; gripper in [0, 1]).
"""

import argparse
import json
from pathlib import Path

import torch

from action_free.idm import (
    ACTION_NAMES,
    absolute_targets,
    errors,
    load_idm,
    load_transitions,
    predict_transitions,
    select_device,
    split_files,
)

COLLECTIONS = [
    "data/source-deterministic-test-n50-seed50000",
    "data/A-deterministic-n200-seed100000",
    "data/D-deterministic-n200-seed400000",
    "data/E-deterministic-n200-seed500000",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--collections", nargs="+", default=COLLECTIONS)
    a = p.parse_args()
    device = select_device()
    model = load_idm(a.checkpoint, device)
    size = torch.load(a.checkpoint, map_location="cpu")["config"]["image_size"]
    std = model.target_std.cpu()

    results = {}
    for c in a.collections:
        files = split_files(Path(c))
        # One episode at a time, so only one episode's frames are in memory.
        pred = torch.cat(
            [
                predict_transitions(model, load_transitions([f], size, labeled=False), device)
                for f in files
            ]
        )
        true = torch.cat([torch.from_numpy(absolute_targets(f)) for f in files])
        results[Path(c).name] = {"episodes": len(files), **errors(pred, true, std)}

    print(f"{'collection':<44} {'sMAE':>7} " + " ".join(f"{n:>8}" for n in ACTION_NAMES))
    for name, r in results.items():
        mae = " ".join(f"{r['mae'][n]:8.4f}" for n in ACTION_NAMES)
        print(f"{name:<44} {r['smae']:7.4f} {mae}")
    out = a.checkpoint.parent / "label_error.json"
    out.write_text(json.dumps({"checkpoint": str(a.checkpoint), **results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
