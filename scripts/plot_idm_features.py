"""PCA of IDM features on source and target validation pairs, before and after DANN.

    uv run --with matplotlib python scripts/plot_idm_features.py \
        outputs/idm_turned/seed0/best.pt outputs/idm_dann_A/seed0/last.pt

The source collections and the target are those of the DANN checkpoint's config. Both are
read as images only (labeled=False, R1). Each panel is a PCA (torch.pca_lowrank) fitted on
that model's source and target features together. Writes features_pca.png next to the DANN
checkpoint.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from action_free.idm import (
    TransitionDataset,
    load_idm,
    load_transitions,
    select_device,
    split_files,
)

SOURCE_COLOR = "#2a78d6"
TARGET_COLOR = "#eb6834"


@torch.no_grad()
def features(checkpoint: Path, files: list[Path], device: torch.device) -> torch.Tensor:
    """IDM features (N, 1536) of every validation pair, one episode at a time."""
    model = load_idm(checkpoint, device)
    size = torch.load(checkpoint, map_location="cpu")["config"]["image_size"]
    out = []
    for f in files:
        loader = DataLoader(TransitionDataset(load_transitions([f], size, labeled=False)), 256)
        for b in loader:
            out.append(model.features(b["obs"].to(device), b["next_obs"].to(device)).cpu())
    return torch.cat(out)


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    """Projection of the rows of x on their first two principal components."""
    mean = x.mean(0)
    _, _, v = torch.pca_lowrank(x - mean, q=2, center=False)
    return (x - mean) @ v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("before", type=Path, help="checkpoint without DANN (idm_turned)")
    p.add_argument("after", type=Path, help="DANN checkpoint")
    a = p.parse_args()
    cfg = torch.load(a.after, map_location="cpu")["config"]
    source_files = [f for c in cfg["collections"] for f in split_files(Path(c), "val")]
    target_files = split_files(Path(cfg["target"]), "val")
    device = select_device()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = ["Before (no DANN)", "After DANN"]
    for ax, checkpoint, title in zip(axes, [a.before, a.after], titles):
        source = features(checkpoint, source_files, device)
        target = features(checkpoint, target_files, device)
        z = pca_2d(torch.cat([source, target]))
        n = len(source)
        ax.scatter(z[:n, 0], z[:n, 1], s=4, alpha=0.3, color=SOURCE_COLOR, label="source")
        ax.scatter(z[n:, 0], z[n:, 1], s=4, alpha=0.3, color=TARGET_COLOR, label="target")
        ax.set_title(f"{title}: {checkpoint.parent.parent.name}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(color="0.9")
        ax.legend(markerscale=3)
    fig.suptitle(f"IDM features, validation pairs, target {Path(cfg['target']).name}")
    fig.tight_layout()
    out = a.after.parent / "features_pca.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
