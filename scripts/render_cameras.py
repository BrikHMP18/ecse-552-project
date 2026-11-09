"""Render the three policy cameras (front, side, wrist) after one reset.

Smoke test for headless rendering before any data collection. Run from the project root:

    MUJOCO_GL=egl uv run python scripts/render_cameras.py
    MUJOCO_GL=glfw xvfb-run -a uv run python scripts/render_cameras.py   # no EGL

Writes a side-by-side PNG and prints the observation shape, which should be (3, 224, 224, 3).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from action_free.collect import make_env
from action_free.objects import OBJECTS

CAMERAS = ("front", "side", "wrist")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-name", default="visual-block-cpu-lite-single-task1-v0")
    parser.add_argument("--object", default="source", choices=sorted(OBJECTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene", default="default")
    parser.add_argument("--out", type=Path, default=Path("outputs/cameras.png"))
    parser.add_argument("--gap", type=int, default=10, help="Pixels between views.")
    args = parser.parse_args()

    env = make_env(args.env_name, args.object, scene=args.scene)
    ob, _ = env.reset(seed=args.seed)
    env.close()
    print(f"observation: shape={ob.shape} dtype={ob.dtype}")
    if ob.ndim != 4 or ob.shape[0] != len(CAMERAS):
        raise SystemExit(f"Expected {len(CAMERAS)} stacked camera views, got {ob.shape}.")

    n, h, w, _ = ob.shape
    canvas = np.full((h, n * w + (n - 1) * args.gap, 3), 255, dtype=np.uint8)
    for i, view in enumerate(ob):
        canvas[:, i * (w + args.gap) : i * (w + args.gap) + w] = view
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(args.out)
    print(f"saved {', '.join(CAMERAS)} to {args.out}")


if __name__ == "__main__":
    main()
