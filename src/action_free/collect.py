"""Reproducible scripted-expert collection with one image per control step.

OCBench's own generator (`data_gen_scripts/generate_ocbench.py`) resets environments
without a seed, so two runs with the same `--seed` produce different initial states, and
it holds a whole dataset in memory before saving. Here every episode has its own seed,
which fixes both the environment's initial state and the expert's random choices, and
each episode is written to its own file as soon as it finishes.

Episode file `episode_XXXXX.npz` (T control steps, C cameras):
    observations  (T+1, C, H, W, 3) uint8   frames before each step plus the final frame
    actions       (T, 7) float32            normalized joint and gripper increments
    rewards, masks, terminals (T,)
    qpos, qvel    (T, ...) float32          PRIVILEGED simulator state: evaluation and
                                            debugging only, never a learned-model input
    seed, success, length                   scalars

Episodes that end unhealthy, or fail when `keep_failures` is false, are discarded and
the next seed is tried; `manifest.json` records every attempt, the object and the scene
parameters, and `initial_states.csv` the start of every kept episode.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from action_free.objects import env_kwargs, get_object

# Scripted-expert variants: OCBench's default (deliberate detours with p = 0.1, random speed)
# and a deterministic one (no detours, fixed speed), like ACT's scripted demonstrations.
EXPERTS: dict[str, dict] = {
    "ocbench": {},
    "deterministic": {"p_mistake": 0.0, "speed_profile": (1.0, 1.0, 1.0)},
}

DEFAULT_ENV = "visual-block-cpu-lite-single-task1-v0"


@dataclass
class CollectConfig:
    out_dir: Path
    env_name: str = DEFAULT_ENV
    object_name: str = "source"
    num_episodes: int = 300
    base_seed: int = 0
    keep_failures: bool = False
    terminate_at_success: bool = True
    max_attempts_factor: int = 3
    role: str = "train"  # "test": held-out collection, never trained on or selected with
    scene: str = "default"  # objects.SCENES
    expert: str = "ocbench"  # EXPERTS


def make_env(
    env_name: str,
    object_name: str = "source",
    terminate_at_success: bool = True,
    scene: str = "default",
):
    """Make a CPU OCBench environment showing the named object in a named scene."""
    import ocbench

    if "-cpu-" not in env_name:
        raise ValueError(f"Use a CPU environment (with '-cpu-'), got {env_name}.")
    return ocbench.make(
        env_name, terminate_at_success=terminate_at_success, **env_kwargs(object_name, scene)
    )


def make_expert(env, env_name: str, expert: str = "ocbench"):
    name = env_name.removeprefix("visual-")
    if not name.startswith("block-"):
        raise ValueError(f"Only block environments are supported, got {env_name}.")
    from ocbench.oracles.controllers.block import BlockController

    return BlockController(env=env, **EXPERTS[expert])


def run_episode(env, env_name: str, seed: int, expert: str = "ocbench") -> tuple[dict, dict]:
    """Roll out the scripted expert from the initial state fixed by `seed`."""
    np.random.seed(seed)  # The OCBench expert samples from NumPy's global RNG.
    ob, info = env.reset(seed=seed)
    expert = make_expert(env, env_name, expert)
    expert.reset(ob, info, transition_idx=0)

    steps: dict[str, list] = {
        k: [] for k in ("observations", "actions", "rewards", "masks", "terminals", "qpos", "qvel")
    }
    success, healthy = False, True
    done = False
    while not done:
        t = len(steps["actions"])
        action = np.clip(np.asarray(expert.select_action(ob, info, transition_idx=t)), -1, 1)
        # Step with exactly the float32 values that are stored: contact dynamics amplify the
        # float64 -> float32 rounding (1e-8) enough to change replayed grasps.
        action = action.astype(np.float32)
        next_ob, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        success = success or bool(info["success"])
        healthy = healthy and bool(info["healthy"])
        steps["observations"].append(ob)
        steps["actions"].append(action)
        steps["rewards"].append(float(reward))
        steps["masks"].append(1.0 - float(terminated))
        steps["terminals"].append(done)
        steps["qpos"].append(info["prev_qpos"])
        steps["qvel"].append(info["prev_qvel"])
        ob = next_ob
    steps["observations"].append(ob)
    expert.finish_episode(len(steps["actions"]) - 1, info)
    num_detours = count_detours(expert)

    episode = {
        "observations": np.asarray(steps["observations"], dtype=np.uint8),
        "actions": np.asarray(steps["actions"], dtype=np.float32),
        "rewards": np.asarray(steps["rewards"], dtype=np.float32),
        "masks": np.asarray(steps["masks"], dtype=np.float32),
        "terminals": np.asarray(steps["terminals"], dtype=bool),
        "qpos": np.asarray(steps["qpos"], dtype=np.float32),
        "qvel": np.asarray(steps["qvel"], dtype=np.float32),
    }
    meta = {
        "seed": seed,
        "success": success,
        "healthy": healthy,
        "length": len(steps["actions"]),
        "speed": float(expert.speed),
        "p_mistake": float(expert.p_mistake),
        "num_detours": num_detours,
    }
    return episode, meta


def count_detours(expert) -> int:
    """Deliberate expert mistakes (OCBench `p_mistake`): the cube is first carried to a
    random spot, then picked up again and taken to the goal."""
    return sum(bool(s.get("is_potential_high_level_mistake")) for s in expert.segments)


def collect(cfg: CollectConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("episode_*.npz")):
        raise FileExistsError(f"{out_dir} already contains episodes; use a new directory.")

    env = make_env(cfg.env_name, cfg.object_name, cfg.terminate_at_success, cfg.scene)
    attempts, kept = [], []
    seed = cfg.base_seed
    max_attempts = cfg.num_episodes * cfg.max_attempts_factor
    start = time.time()
    try:
        while len(kept) < cfg.num_episodes:
            if len(attempts) >= max_attempts:
                raise RuntimeError(
                    f"Only {len(kept)} usable episodes after {max_attempts} "
                    "attempts; check the expert."
                )
            episode, meta = run_episode(env, cfg.env_name, seed, cfg.expert)
            keep = meta["healthy"] and (meta["success"] or cfg.keep_failures)
            meta["kept"] = keep
            if keep:
                meta["file"] = f"episode_{len(kept):05d}.npz"
                np.savez_compressed(
                    out_dir / meta["file"],
                    **episode,
                    seed=seed,
                    success=meta["success"],
                    length=meta["length"],
                )
                kept.append(meta)
            attempts.append(meta)
            print(
                f"[{len(kept)}/{cfg.num_episodes}] seed={seed} success={meta['success']} "
                f"len={meta['length']} kept={keep}",
                flush=True,
            )
            seed += 1
    finally:
        env.close()

    lengths = [m["length"] for m in kept]
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "object": {
            "description": get_object(cfg.object_name).description,
            "env_kwargs": env_kwargs(cfg.object_name, cfg.scene),
        },
        "num_kept": len(kept),
        "num_attempts": len(attempts),
        "expert_success_rate": float(np.mean([m["success"] for m in attempts])),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "total_transitions": int(sum(lengths)),
        "wall_time_s": round(time.time() - start, 1),
        "episodes": attempts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    from action_free.initial_states import write_csv

    write_csv(out_dir, cfg.env_name, cfg.object_name)
    return summary
