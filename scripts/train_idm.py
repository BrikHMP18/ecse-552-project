"""Train the IDM on source demos and keep the checkpoint with the lowest validation error.

    uv run python scripts/train_idm.py --config configs/idm.toml --out outputs/idm/source/seed0
    uv run python scripts/train_idm.py --config configs/idm.toml --out outputs/idm/smoke --smoke

Writes best.pt (lowest standardized validation MAE), last.pt, history.json (per-epoch train
loss, validation loss and per-component error) and config.json in the output directory.
--smoke trains one epoch on 10 episodes per split and collection, only to check that the
pipeline runs. Mainly to avoid overspending credits on modal with faulty scripts. 
"""

import argparse
import tomllib
from pathlib import Path

from action_free.idm import train_idm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None, help="Overrides the config's seed")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.smoke:
        cfg["epochs"] = 1
    best = train_idm(cfg, a.out, max_episodes=10 if a.smoke else None)
    mae = ", ".join(f"{k} {v:.4f}" for k, v in best["val"]["mae"].items())
    print(f"best epoch {best['epoch']}: validation sMAE {best['val']['smae']:.4f}; MAE {mae}")


if __name__ == "__main__":
    main()
