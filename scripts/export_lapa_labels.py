"""Pseudo-label a target collection with the LAPA labeler (README: Pseudo-labels).

    uv run python scripts/export_lapa_labels.py outputs/laq/seed0/best.pt \
        outputs/lapa_g/seed0/best.pt data/A-deterministic-n200-seed100000

Writes `data/labels/<target>-lapa-seed<seed>/` (e.g. `A-deterministic-lapa-seed0`): one
`episode_XXXXX.npy` per episode, (T, 7) float32 absolute targets, and `labels.json`
(collection, LAQ and g checkpoints, commit). The labeler sees images only; stored actions
and states are never read (R1, R3). Upload the folder to the dataset repository as
`labels/<name>/`.
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from action_free.idm import select_device, split_files
from action_free.lapa import label_collection, load_g, load_laq


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("laq_checkpoint", type=Path)
    p.add_argument("g_checkpoint", type=Path)
    p.add_argument("collection", type=Path)
    p.add_argument("--out-root", type=Path, default=Path("data/labels"))
    a = p.parse_args()
    laq_ckpt = torch.load(a.laq_checkpoint, map_location="cpu")
    g_ckpt = torch.load(a.g_checkpoint, map_location="cpu")
    target = a.collection.name.split("-n")[0]  # e.g. A-deterministic
    out = a.out_root / f"{target}-lapa-seed{laq_ckpt['config']['seed']}"
    out.mkdir(parents=True, exist_ok=True)

    device = select_device()
    laq = load_laq(a.laq_checkpoint, device)
    decoder = load_g(a.g_checkpoint, device)
    files = split_files(a.collection)
    label_collection(laq, decoder, files, laq.config["image_size"], out, device)
    info = {
        "collection": a.collection.name,
        "episodes": len(files),
        "laq_checkpoint": str(a.laq_checkpoint),
        "laq_epoch": laq_ckpt["epoch"],
        "laq_config": laq_ckpt["config"],
        "g_checkpoint": str(a.g_checkpoint),
        "g_epoch": g_ckpt["epoch"],
        "g_config": g_ckpt["config"],
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "src/action_free/lapa.py", "scripts")),
    }
    (out / "labels.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {len(files)} episodes to {out}")


if __name__ == "__main__":
    main()
