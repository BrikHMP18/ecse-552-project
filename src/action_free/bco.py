"""Iterative BCO, method (g) (docs/experiments/bco.md): the policy's own rollouts on the
target object, stored as episodes the IDM trains on.

A rollout episode has the layout of `collect.py` (observations, actions, qpos, qvel, seed,
success, length) plus `healthy`; its actions are those the policy sent to the robot, so
`idm.absolute_targets` labels it as it labels a source demo. Every rollout is kept.

Paths in BCO configs name the Modal volumes: `bco:<path>` is `<path>` in the volume
`ecse552-bco`, under `runs/` (or `smoke/` for smoke runs), and `out:<path>` is in the outputs
volume of the app that reads it (IDM or ACT). The rollout app also mounts the IDM outputs as
`idm:<path>`. `{seed}` is the training seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROLLOUT_BASE_SEED = 800_000  # configs/data.toml [seeds] bco_rollouts
ROLLOUTS_PER_ROUND = 200
VOLUME_ROOT = Path("/bco")


def rollout_seeds(round_: int, n: int | None = None) -> list[int]:
    """Initial-state seeds of a round's rollouts (the first n; all 200 by default), the same
    for every training seed."""
    first = ROLLOUT_BASE_SEED + ROLLOUTS_PER_ROUND * (round_ - 1)
    return [first + i for i in range(n or ROLLOUTS_PER_ROUND)]


def resolve(path: str, seed: int, smoke: bool = False) -> str:
    """A config path with `{seed}` filled in and `bco:`, `out:` or `idm:` mapped to a mount
    point."""
    path = path.format(seed=seed)
    if path.startswith("bco:"):
        return str(VOLUME_ROOT / ("smoke" if smoke else "runs") / path.removeprefix("bco:"))
    if path.startswith("out:"):
        return "/outputs/" + path.removeprefix("out:")
    if path.startswith("idm:"):
        return "/idm/" + path.removeprefix("idm:")
    return path


def write_rollout(out_dir: Path, index: int, r: dict) -> str:
    """One recorded rollout (`evaluate.rollout(..., record=True)`) as `episode_XXXXX.npz`."""
    name = f"episode_{index:05d}.npz"
    np.savez_compressed(
        Path(out_dir) / name,
        observations=np.asarray(r["frames"], dtype=np.uint8),
        actions=r["actions"],
        qpos=r["qpos"],
        qvel=r["qvel"],
        seed=r["seed"],
        success=r["success"],
        length=r["length"],
        healthy=r["healthy"],
    )
    return name


def collect_rollouts(
    checkpoint: str, object_name: str, seeds: list[int], out_dir: Path, execution: str, log=print
) -> dict:
    """Roll out a policy from each seed's initial state; write every episode and a manifest."""
    import torch

    from action_free.collect import DEFAULT_ENV, make_env
    from action_free.evaluate import load_policy, rollout

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("episode_*.npz")):
        raise FileExistsError(f"{out_dir} already contains episodes; use a new directory.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(checkpoint, device)
    scene = (policy.run.get("data") or {}).get("scene_name") or "fixed_yaw"
    env = make_env(DEFAULT_ENV, object_name, scene=scene)
    episodes = []
    try:
        for i, seed in enumerate(seeds):
            r = rollout(env, policy, seed, device, execution=execution, record=True)
            write_rollout(out_dir, i, r)
            episodes.append({k: r[k] for k in ("seed", "success", "length", "healthy")})
            log(f"[{i + 1}/{len(seeds)}] seed={seed} success={r['success']} len={r['length']}")
    finally:
        env.close()
    manifest = {
        "checkpoint": checkpoint,
        "object": object_name,
        "scene": scene,
        "execution": execution,
        "num_episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "total_transitions": int(sum(e["length"] for e in episodes)),
        "episodes": episodes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
