"""Classify why closed-loop rollouts fail: grasp, transport, placement or release.

    MUJOCO_GL=egl uv run python scripts/diagnose_failures.py <run>/selected.ckpt
    MUJOCO_GL=egl uv run python scripts/diagnose_failures.py <ckpt> --execution chunk

Rolls the policy out from validation seeds (default; same states as checkpoint selection)
and sorts each episode by the cube's trajectory: never lifted (grasp failed), lifted but
ended far from the square, placed within 10 cm but not fully inside, inside but not
released/still in time, or success. Uses simulator state for the analysis only.
Writes failures_<seeds>_<object>[_<execution>].json next to the checkpoint.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from action_free.collect import DEFAULT_ENV, make_env
from action_free.evaluate import (
    DEFAULT_EXECUTION,
    TEST_BASE_SEED,
    VALIDATION_BASE_SEED,
    load_policy,
    rollout,
)

GOAL_XY = np.array([0.425, 0.0])
LIFT_Z = 0.06  # resting cube centre is at 0.03 m


def classify(r: dict) -> str:
    end = r["end"]
    if r["success"]:
        return "success"
    if end["max_cube_z"] < LIFT_Z:
        return "never lifted (grasp failed)"
    if end["conditions"]["inside"]:
        return "inside the square but not released and still in time"
    if np.linalg.norm(np.asarray(end["cube_xy"]) - GOAL_XY) < 0.10:
        return "placed within 10 cm but not fully inside"
    return "lifted but ended far from the square"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--object", default="source")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--seeds", choices=("validation", "test"), default="validation")
    p.add_argument("--execution", default=DEFAULT_EXECUTION, help="chunk, replan:N or ensemble:M")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(str(a.checkpoint), device)
    scene = (policy.run.get("data") or {}).get("scene_name") or "default"
    env = make_env(DEFAULT_ENV, a.object, scene=scene)
    base = VALIDATION_BASE_SEED if a.seeds == "validation" else TEST_BASE_SEED
    episodes = []
    for i in range(a.num_episodes):
        r = rollout(env, policy, base + i, device, execution=a.execution)
        episodes.append(
            {"seed": r["seed"], "category": classify(r), "start": r["start"], **r["end"]}
        )
    env.close()

    counts = Counter(e["category"] for e in episodes)
    out = {
        "checkpoint": str(a.checkpoint),
        "object": a.object,
        "seeds": a.seeds,
        "execution": a.execution,
        "action_mode": policy.action_mode,
        "counts": dict(counts.most_common()),
        "episodes": episodes,
    }
    suffix = "_" + a.execution.replace(":", "")
    path = a.checkpoint.parent / f"failures_{a.seeds}_{a.object}{suffix}.json"
    path.write_text(json.dumps(out, indent=2))
    for k, v in counts.most_common():
        print(f"{v:4d}/{a.num_episodes}  {k}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
