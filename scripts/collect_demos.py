"""Collect scripted-expert demonstrations with one image per control step.

Run from the project root (headless rendering, see the README):

    MUJOCO_GL=egl uv run python scripts/collect_demos.py --num-episodes 300
    MUJOCO_GL=glfw xvfb-run -a uv run python scripts/collect_demos.py --num-episodes 5 \
        --out-dir data/pilot

Each episode is seeded (base seed + attempt index), so the same command reproduces the
same initial states. Provenance (commits, packages) is written next to the episodes.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from action_free.collect import DEFAULT_ENV, CollectConfig, collect
from action_free.objects import OBJECTS


def git_rev(path: str = ".") -> str:
    rev = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(["git", "-C", path, "diff", "--quiet"]).returncode != 0
    return rev + ("-dirty" if dirty else "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default=DEFAULT_ENV)
    p.add_argument("--object", default="source", choices=sorted(OBJECTS))
    p.add_argument("--num-episodes", type=int, default=300)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument(
        "--out-dir", type=Path, default=None, help="Default: data/<object>-n<num>-seed<base-seed>"
    )
    p.add_argument(
        "--keep-failures",
        action="store_true",
        help="Keep healthy but unsuccessful episodes (default: retry next seed).",
    )
    p.add_argument(
        "--role",
        choices=("train", "test"),
        default="train",
        help="test: held-out collection (split 'test', refused by training)",
    )
    p.add_argument(
        "--no-terminate-at-success",
        action="store_true",
        help="Run to the time limit instead of stopping at first success.",
    )
    a = p.parse_args()

    out_dir = a.out_dir or Path("data") / f"{a.object}-n{a.num_episodes}-seed{a.base_seed}"
    cfg = CollectConfig(
        out_dir=out_dir,
        env_name=a.env_name,
        object_name=a.object,
        num_episodes=a.num_episodes,
        base_seed=a.base_seed,
        keep_failures=a.keep_failures,
        terminate_at_success=not a.no_terminate_at_success,
        role=a.role,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(["uv", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "packages.txt").write_text(freeze)
    (out_dir / "provenance.txt").write_text(
        f"command={' '.join(sys.argv)}\n"
        f"project_commit={git_rev()}\n"
        f"ocbench_commit={git_rev('third_party/ocbench')}\n"
        f"mujoco_gl={os.environ.get('MUJOCO_GL', '')}\n"
    )

    s = collect(cfg)
    print(
        f"\nkept {s['num_kept']}/{s['num_attempts']} attempts, "
        f"expert success {s['expert_success_rate']:.1%}, mean length {s['mean_length']:.1f}, "
        f"{s['total_transitions']} transitions, {s['wall_time_s']} s -> {out_dir}"
    )


if __name__ == "__main__":
    main()
