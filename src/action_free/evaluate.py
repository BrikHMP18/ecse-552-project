"""Closed-loop policy rollouts: success, episode length and inference latency.

Every method and object is evaluated from the same initial states: test episode i uses
seed `TEST_BASE_SEED + i` (`configs/data.toml`), which fixes the cube and arm state for every
object. Validation rollouts (`VALIDATION_BASE_SEED`, source object only) choose between an
ACT run's `policy_best.ckpt` (lowest validation loss, upstream's choice) and
`policy_last.ckpt` (`select_checkpoint`); test seeds are used once, on the selected one.
Policies are the original ACT (`act_original.ActPolicy`); `DEFAULT_EXECUTION` is ACT's
temporal ensembling. The policy sees only the three camera images and predicts a chunk of
actions. `execution` sets how chunks are used: "chunk" runs each chunk open loop before the
next query (the proposal's default); "replan:N" runs the first N actions of each chunk;
"ensemble:M" queries every step and averages all chunks' predictions for the current step
with weights exp(-M * i), i = 0 for the oldest (ACT's temporal ensembling). Policies trained
on absolute joint targets are executed through `actions.to_delta`, which uses the robot's
current joint positions (low-level control). Episodes end at strict success or at the
200-step (20 s) limit.

Latency (R6) is the wall time of one policy call (images to action chunk, including the
host-to-device copy), measured after warm-up with CUDA synchronization.
Target-object actions are never read here: evaluation needs only the simulator.

Every rollout records its initial state and how it relates to the training distribution
stored in the checkpoint: whether the cube starts inside the training start area, and the
distance to the nearest training start. The summary lists what differs from training
(object, scene parameters), so out-of-distribution evaluations are explicit.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from action_free import initial_states
from action_free.actions import to_delta
from action_free.collect import DEFAULT_ENV, make_env
from action_free.objects import env_kwargs

TEST_BASE_SEED = 1_000_000  # final evaluation, every method and object
VALIDATION_BASE_SEED = 900_000  # checkpoint selection and the R2 decision, source only
EVAL_BASE_SEED = TEST_BASE_SEED
DEFAULT_EXECUTION = "ensemble:0.01"  # ACT's temporal ensembling (imitate_episodes.py, k = 0.01)


@dataclass
class EvalConfig:
    checkpoint: str
    object_name: str = "source"
    num_episodes: int = 100
    base_seed: int = EVAL_BASE_SEED
    env_name: str = DEFAULT_ENV
    video_episodes: int = 0  # write MP4s of the first N episodes
    out_dir: str | None = None
    execution: str = DEFAULT_EXECUTION  # "chunk", "replan:N" or "ensemble:M"
    scene: str | None = None  # default: the scene the policy was trained in
    seeds: list[int] | None = None  # explicit seeds instead of base_seed + i


def load_policy(path: str, device: torch.device):
    """A trained original-ACT checkpoint (`*.ckpt` next to act_run.json, dataset_stats.pkl)."""
    from action_free.act_original import ActPolicy

    return ActPolicy(Path(path), device)


def checkpoint_epoch(path: str, run: dict) -> int:
    """Training epoch of an upstream checkpoint file."""
    name = Path(path).stem
    selection = Path(path).parent / "selection.json"
    if name == "selected" and selection.exists():
        return int(json.loads(selection.read_text())["selected_epoch"])
    if name == "policy_best":
        return int(run.get("best_epoch", -1))
    if name == "policy_last":
        return int(run.get("last_epoch", -1))
    if name.startswith("policy_epoch_"):
        return int(name.split("_")[2])
    return -1


def parse_execution(execution: str, chunk: int) -> tuple[str, float]:
    """("replan", N) or ("ensemble", M) from "chunk", "replan:N" or "ensemble:M"."""
    if execution == "chunk":
        return "replan", chunk
    kind, _, value = execution.partition(":")
    if kind == "replan" and 1 <= int(value) <= chunk:
        return "replan", int(value)
    if kind == "ensemble":
        return "ensemble", float(value)
    raise ValueError(f"execution must be chunk, replan:N (1..{chunk}) or ensemble:M: {execution!r}")


def rollout(
    env,
    policy,
    seed: int,
    device: torch.device,
    keep_frames: bool = False,
    execution: str = "chunk",
) -> dict:
    np.random.seed(seed)
    ob, info = env.reset(seed=seed)
    u = env.unwrapped
    start = initial_states.current(env)
    k = policy.cfg.chunk
    kind, param = parse_execution(execution, k)
    absolute = getattr(policy, "action_mode", "delta") == "absolute"
    frames = [ob] if keep_frames else None
    latencies, success, steps, done, max_cube_z = [], False, 0, False, 0.0
    queue: list[np.ndarray] = []
    history: list[tuple[int, np.ndarray]] = []  # (step the chunk starts at, chunk) for ensembling
    while not done:
        if kind == "ensemble" or not queue:
            t0 = time.perf_counter()
            images = torch.from_numpy(np.ascontiguousarray(ob)[None]).to(device)
            if getattr(policy, "uses_robot_state", False):  # reference run only (not R3)
                pred = policy.predict(images, u._data.qpos.copy())[0].float().cpu().numpy()
            else:
                pred = policy.predict(images)[0].float().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            if kind == "ensemble":
                history = [(s, c) for s, c in history if steps - s < k] + [(steps, pred)]
            else:
                queue = list(pred[: int(param)])
        if kind == "ensemble":
            preds = np.stack([c[steps - s] for s, c in history])  # oldest first
            w = np.exp(-param * np.arange(len(preds)))
            action = (w[:, None] * preds).sum(0) / w.sum()
        else:
            action = queue.pop(0)
        if absolute:
            action = to_delta(action, u._data.qpos)  # low-level control uses the robot state
        ob, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        steps += 1
        success = success or bool(info["success"])
        done = bool(terminated or truncated)
        max_cube_z = max(max_cube_z, float(u._data.joint("object_joint_0").qpos[2]))
        if keep_frames:
            frames.append(ob)
    return {
        "seed": seed,
        "success": success,
        "length": steps,
        "start": {k: round(v, 4) for k, v in start.items() if k.startswith("cube_")},
        "end": {
            "max_cube_z": round(max_cube_z, 4),
            "cube_xy": [round(float(v), 4) for v in u._data.joint("object_joint_0").qpos[:2]],
            "conditions": u.goal_marker_conditions(0),
        },
        "latencies_s": latencies,
        "frames": np.asarray(frames) if keep_frames else None,
    }


def _json(x):
    return json.loads(json.dumps(x))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def compare_to_training(
    train_data: dict | None, object_name: str, episodes: list[dict], scene_name: str = "default"
) -> dict:
    """Mark each episode's start against the training distribution; summarize differences."""
    if not train_data:
        return {"note": "checkpoint has no training-data description"}
    scene = _json(env_kwargs(object_name, scene_name))
    train_scene = train_data.get("scene") or {}
    diffs = {
        k: {"train": train_scene.get(k), "eval": scene.get(k)}
        for k in sorted(set(scene) | set(train_scene))
        if train_scene.get(k) != scene.get(k)
    }
    starts = np.asarray(train_data.get("train_initial_states") or [], dtype=float)
    bounds = train_scene.get("object_sampling_bounds")
    for e in episodes:
        xy = np.array([e["start"]["cube_x"], e["start"]["cube_y"]])
        if bounds is not None:
            lo, hi = np.asarray(bounds, dtype=float)
            e["start"]["inside_train_area"] = bool(np.all((xy >= lo) & (xy <= hi)))
        if len(starts):
            e["start"]["nearest_train_xy_cm"] = round(
                float(np.min(np.linalg.norm(starts[:, :2] - xy, axis=1)) * 100), 2
            )
    nearest = [
        e["start"]["nearest_train_xy_cm"] for e in episodes if "nearest_train_xy_cm" in e["start"]
    ]
    inside = [
        e["start"]["inside_train_area"] for e in episodes if "inside_train_area" in e["start"]
    ]
    return {
        "train_object": train_data.get("object"),
        "eval_object": object_name,
        "same_object": train_data.get("object") == object_name,
        "scene_differences": diffs,  # e.g. cube_colors for a target object
        "starts_inside_train_area": float(np.mean(inside)) if inside else None,
        "nearest_train_xy_cm": {"mean": float(np.mean(nearest)), "max": float(np.max(nearest))}
        if nearest
        else None,
        "num_train_starts": int(len(starts)),
    }


def evaluate(cfg: EvalConfig, log=print) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(cfg.checkpoint, device)
    scene = cfg.scene or (policy.run.get("data") or {}).get("scene_name") or "default"
    env = make_env(cfg.env_name, cfg.object_name, scene=scene)
    out = Path(cfg.out_dir) if cfg.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    # Warm-up (cuDNN autotuning, allocator) so latencies reflect steady state.
    ob, _ = env.reset(seed=cfg.base_seed)
    state = env.unwrapped._data.qpos.copy() if getattr(policy, "uses_robot_state", False) else None
    for _ in range(5):
        if state is None:
            policy.predict(torch.from_numpy(ob[None]).to(device))
        else:
            policy.predict(torch.from_numpy(ob[None]).to(device), state)

    seeds = cfg.seeds or [cfg.base_seed + i for i in range(cfg.num_episodes)]
    episodes = []
    t_start = time.time()
    try:
        for i, seed in enumerate(seeds):
            keep = i < cfg.video_episodes
            r = rollout(env, policy, seed, device, keep_frames=keep, execution=cfg.execution)
            if r["frames"] is not None and out:
                from action_free.replay import tile, write_video

                tag = "success" if r["success"] else "fail"
                write_video(np.stack([tile(o) for o in r["frames"]]), out / f"ep{i:03d}_{tag}.mp4")
            r.pop("frames")
            episodes.append(r)
            rate = np.mean([e["success"] for e in episodes])
            log(f"[{i + 1}/{len(seeds)}] seed={r['seed']} success={r['success']} "
                f"len={r['length']} running success={rate:.1%}")  # fmt: skip
    finally:
        env.close()

    lat = np.concatenate([e["latencies_s"] for e in episodes]) * 1000
    run = policy.run
    distribution = compare_to_training(run.get("data"), cfg.object_name, episodes, scene)
    summary = {
        "config": asdict(cfg),
        "checkpoint_epoch": checkpoint_epoch(cfg.checkpoint, run),
        "action_mode": policy.action_mode,
        "execution": cfg.execution,
        "scene": scene,
        "uses_robot_state": bool(getattr(policy, "uses_robot_state", False)),
        "provenance": {
            "project_commit": _git("rev-parse", "HEAD"),
            "ocbench_commit": _git("-C", "third_party/ocbench", "rev-parse", "HEAD"),
            "env_kwargs": _json(env_kwargs(cfg.object_name, scene)),
            "train_provenance": run.get("provenance"),
        },
        "distribution": distribution,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "num_episodes": len(episodes),
        "mean_length_success": float(
            np.mean([e["length"] for e in episodes if e["success"]] or [np.nan])
        ),
        "latency_ms": {
            "mean": float(lat.mean()),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "max": float(lat.max()),
        },
        "wall_time_s": round(time.time() - t_start, 1),
        "episodes": [{k: v for k, v in e.items() if k != "latencies_s"} for e in episodes],
    }
    if out:
        (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def choose_checkpoint(best_success: float, last_success: float) -> str:
    """The pre-registered rule: higher validation success wins, ties go to the best-loss one."""
    return "last" if last_success > best_success else "best"


def select_checkpoint(
    run_dir: Path, num_episodes: int = 100, execution: str = DEFAULT_EXECUTION, log=print
) -> dict:
    """Choose between `policy_best.ckpt` (lowest validation loss) and `policy_last.ckpt` of an
    ACT run by success on source validation rollouts; ties go to `policy_best.ckpt`. Writes
    `selection.json` and copies the winner to `selected.ckpt`. Uses only the source object and
    validation seeds (R1)."""
    import shutil

    run_dir = Path(run_dir)
    results = {}
    for name in ("best", "last"):
        cfg = EvalConfig(
            checkpoint=str(run_dir / f"policy_{name}.ckpt"),
            object_name="source",
            num_episodes=num_episodes,
            base_seed=VALIDATION_BASE_SEED,
            out_dir=str(run_dir / f"validation_{name}"),
            execution=execution,
        )
        results[name] = evaluate(cfg, log=lambda *a, **k: None)
        log(f"{run_dir} policy_{name}.ckpt: validation success {results[name]['success_rate']:.1%}")
    chosen = choose_checkpoint(results["best"]["success_rate"], results["last"]["success_rate"])
    selection = {
        "rule": "higher source validation success (seeds 900000+), ties to policy_best.ckpt",
        "num_episodes": num_episodes,
        "execution": execution,
        "selected": f"policy_{chosen}.ckpt",
        "selected_epoch": results[chosen]["checkpoint_epoch"],
        "validation_success": {k: v["success_rate"] for k, v in results.items()},
        "epochs": {k: v["checkpoint_epoch"] for k, v in results.items()},
    }
    shutil.copyfile(run_dir / f"policy_{chosen}.ckpt", run_dir / "selected.ckpt")
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    return selection
