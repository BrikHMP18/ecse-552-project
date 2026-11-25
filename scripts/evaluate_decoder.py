"""Decoder error of a LAPA checkpoint on held-out source demos (diagnostic only).

    uv run python scripts/evaluate_decoder.py outputs/lapa_A/n300/seed0/selected.ckpt

The action decoder g(o, z) maps a latent action to an absolute target. The latent labels
come from the LAM of the same run (its path is in the run's config; --lam overrides it).
The 50 held-out source demos (seeds 50,000+) are read images-only; their stored actions are
read separately, only to measure the error, as for the IDM (evaluate_idm.py). Writes
decoder_error.json next to the checkpoint: per collection, the MAE per component (radians;
gripper in [0, 1]).
"""

import argparse
import json
from pathlib import Path

import torch

from action_free.idm import (
    ACTION_NAMES,
    absolute_targets,
    load_transitions,
    select_device,
    split_files,
)
from action_free.lapa import ActionDecoder, load_lam, z_labels

COLLECTIONS = ["data/source-deterministic-test-n50-seed50000"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--collections", nargs="+", default=COLLECTIONS)
    p.add_argument("--lam", type=Path, default=None, help="LAM checkpoint; default: the run's")
    a = p.parse_args()
    device = select_device()
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    decoder = ActionDecoder(cfg["codebook"], hidden=cfg["decoder_hidden"], pretrained=False)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.to(device).eval()
    lam = load_lam(a.lam or Path(cfg["lam"]), device)
    size = lam.config["image_size"]

    results = {}
    for c in a.collections:
        files = split_files(Path(c))
        pred, true = [], []
        for f in files:
            t = load_transitions([f], size, labeled=False)  # the LAM's resolution
            z = z_labels(lam, t, device)
            obs = load_transitions([f], cfg["decoder_image_size"], labeled=False).frames[:-1]
            pred.append(decoder.predict(obs.to(device), z.to(device)).cpu())
            true.append(torch.from_numpy(absolute_targets(f)))
        mae = (torch.cat(pred) - torch.cat(true)).abs().mean(0)
        results[Path(c).name] = {
            "episodes": len(files),
            "mae": dict(zip(ACTION_NAMES, mae.tolist())),
        }

    print(f"{'collection':<44} " + " ".join(f"{n:>8}" for n in ACTION_NAMES))
    for name, r in results.items():
        mae = " ".join(f"{r['mae'][n]:8.4f}" for n in ACTION_NAMES)
        print(f"{name:<44} {mae}")
    out = a.checkpoint.parent / "decoder_error.json"
    out.write_text(json.dumps({"checkpoint": str(a.checkpoint), **results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
