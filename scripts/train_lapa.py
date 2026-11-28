"""Train the LAQ or the action decoder g locally (verification and smoke runs only; the
results run on Modal, scripts/modal_train_lapa.py).

    uv run python scripts/train_lapa.py --stage laq --config configs/laq.toml \
        --out outputs/laq/seed0
    uv run python scripts/train_lapa.py --stage laq --config configs/laq.toml \
        --out outputs/laq/smoke --smoke
    uv run python scripts/train_lapa.py --stage g --config configs/lapa_g.toml \
        --laq outputs/laq/smoke/best.pt --out outputs/lapa_g/smoke --smoke

The LAQ stage writes best.pt, last.pt, history.json and config.json (with the trained
parameter count). The g stage trains on the config's collections with z labels from the
frozen LAQ and writes best.pt, last.pt, history.json and config.json. --smoke trains one
epoch on 10 episodes per collection, mainly to check the pipeline before spending credits
on Modal.
"""

import argparse
import tomllib
from pathlib import Path

from action_free.lapa import train_g, train_laq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("laq", "g"), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None, help="Overrides the config's seed")
    p.add_argument("--laq", type=Path, default=None, help="g stage: LAQ checkpoint")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    if a.seed is not None:
        cfg["seed"] = a.seed
    n = 10 if a.smoke else None
    if a.stage == "laq":
        if a.smoke:
            cfg["epochs"] = 1
            cfg["record_reference"] = False  # the 300M official-scale count is slow on CPU
        best = train_laq(cfg, a.out, max_episodes=n)
        print(f"best epoch {best['epoch']}: validation reconstruction {best['val_recon']:.4f}")
        return
    if a.laq is not None:
        cfg["laq"] = str(a.laq)
    if a.smoke:
        cfg["epochs"] = 1
    best = train_g(cfg, a.out, max_episodes=n)
    print(f"best epoch {best['epoch']}: validation sMAE {best['val']['smae']:.4f}")


if __name__ == "__main__":
    main()
