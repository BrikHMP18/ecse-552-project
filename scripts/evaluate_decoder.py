"""Error of the action decoder g on held-out source demos (diagnostic only).

    uv run python scripts/evaluate_decoder.py outputs/lapa_g/seed0/best.pt

The action decoder g(o_t, z_t) maps a transition's latent action to an absolute target;
the latent labels come from the frozen LAQ recorded in the run's config (`laq`; --laq
overrides it). The 50 held-out source demos (seeds 50,000+) are read images-only; their
stored actions are read separately, only to measure the error, as for the IDM
(evaluate_idm.py). Writes decoder_error.json next to the checkpoint: per collection, the
MAE per component (radians; gripper in [0, 1]), the standardized MAE and the MSE.
"""

import argparse
import json
from pathlib import Path

import torch

from action_free.idm import (
    ACTION_NAMES,
    absolute_targets,
    errors,
    select_device,
    split_files,
)
from action_free.lapa import load_g, load_laq, load_pairs, predict_g, z_codes

COLLECTIONS = ["data/source-deterministic-test-n50-seed50000"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--collections", nargs="+", default=COLLECTIONS)
    p.add_argument("--laq", type=Path, default=None, help="LAQ checkpoint; default: the run's")
    a = p.parse_args()
    device = select_device()
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    decoder = load_g(a.checkpoint, device)
    laq = load_laq(a.laq or Path(ckpt["laq"]), device)
    size = laq.config["image_size"]

    results = {}
    for c in a.collections:
        files = split_files(Path(c))
        pred, true = [], []
        for f in files:
            pairs = load_pairs([f], size, 1, labeled=False)  # images only
            z = z_codes(laq, pairs, device)
            pred.append(predict_g(decoder, pairs, z, device))
            true.append(torch.from_numpy(absolute_targets(f)))
        v = errors(torch.cat(pred), torch.cat(true), decoder.target_std.cpu())
        results[Path(c).name] = {
            "episodes": len(files),
            "smae": v["smae"],
            "mse": v["mse"],
            "mae": v["mae"],
        }

    print(f"{'collection':<44} " + " ".join(f"{n:>8}" for n in ACTION_NAMES) + "     sMAE")
    for name, r in results.items():
        mae = " ".join(f"{r['mae'][n]:8.4f}" for n in ACTION_NAMES)
        print(f"{name:<44} {mae} {r['smae']:8.4f}")
    out = a.checkpoint.parent / "decoder_error.json"
    out.write_text(json.dumps({"checkpoint": str(a.checkpoint), **results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
