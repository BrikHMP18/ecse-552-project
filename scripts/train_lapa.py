"""Train the LAM or the LAPA policy locally (verification and smoke runs only; the results
run on Modal, scripts/modal_train_lapa.py).

    uv run python scripts/train_lapa.py --stage lam --config configs/lam.toml \
        --out outputs/lam/seed0
    uv run python scripts/train_lapa.py --stage lam --config configs/lam.toml \
        --out outputs/lam/smoke --smoke
    uv run python scripts/train_lapa.py --stage policy --config configs/lapa_A.toml \
        --lam outputs/lam/smoke/best.pt --out outputs/lapa_A/smoke --smoke

The LAM stage writes best.pt, last.pt, history.json and config.json. The policy stage
exports the config's collections (source, targets, turned) to HDF5 next to --out and writes
policy_best.ckpt, policy_last.ckpt, lapa_run.json, history.json and config.json. --smoke
trains one epoch on 10 episodes per collection, mainly to check the pipeline before
spending credits on Modal.
"""

import argparse
import tomllib
from pathlib import Path

from action_free.data import episode_files
from action_free.lapa import export_lapa, train_lam, train_lapa


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("lam", "policy"), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None, help="Overrides the config's seed")
    p.add_argument("--lam", type=Path, default=None, help="Policy stage: LAM checkpoint")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    if a.seed is not None:
        cfg["seed"] = a.seed
    n = 10 if a.smoke else None
    if a.stage == "lam":
        if a.smoke:
            cfg["epochs"] = 1
        best = train_lam(cfg, a.out, max_episodes=n)
        print(f"best epoch {best['epoch']}: validation reconstruction {best['val_recon']:.4f}")
        return
    if a.lam is not None:
        cfg["lam"] = str(a.lam)
    if a.smoke:
        cfg["num_epochs"] = 1
    collections = []
    for c in cfg["collections"]:
        collections.append((episode_files(Path(c), n), "source", True))
    for c in cfg.get("targets", []):
        collections.append((episode_files(Path(c), n), "target", False))
    for c in cfg.get("turned", []):
        collections.append((episode_files(Path(c), n), "turned", True))
    dataset_dir = a.out / "dataset"
    export_lapa(collections, dataset_dir)
    record = train_lapa(cfg, dataset_dir, a.out)
    print(f"best epoch {record['best_epoch']}: validation loss {record['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
