"""Pseudo-label a target collection with an IDM checkpoint (README: Pseudo-labels).

    uv run python scripts/export_labels.py outputs/idm_turned/seed0/best.pt \
        data/E-deterministic-n200-seed500000

Writes `data/labels/<target>-<idm>-seed<seed>/` (e.g. `E-deterministic-idm_turned-seed0`):
one `episode_XXXXX.npy` per episode, (T, 7) float32 absolute targets, and `labels.json`
(collection, checkpoint, commit). The IDM sees images only; stored actions and states are
never read (R1, R3). Upload the folder to the dataset repository as `labels/<name>/`.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from action_free.idm import (
    load_idm,
    load_transitions,
    predict_transitions,
    select_device,
    split_files,
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("collection", type=Path)
    p.add_argument("--out-root", type=Path, default=Path("data/labels"))
    a = p.parse_args()
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    idm = a.checkpoint.parent.parent.name  # outputs/<idm>/seed<seed>/best.pt
    target = a.collection.name.split("-n")[0]  # e.g. E-deterministic
    out = a.out_root / f"{target}-{idm}-seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)

    device = select_device()
    model = load_idm(a.checkpoint, device)
    files = split_files(a.collection)
    for f in files:
        t = load_transitions([f], cfg["image_size"], labeled=False)
        np.save(out / f"{f.stem}.npy", predict_transitions(model, t, device).numpy())
    info = {
        "collection": a.collection.name,
        "episodes": len(files),
        "idm_checkpoint": str(a.checkpoint),
        "idm_epoch": ckpt["epoch"],
        "idm_config": cfg,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "src/action_free/idm.py", "scripts")),
    }
    (out / "labels.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {len(files)} episodes to {out}")


if __name__ == "__main__":
    main()
