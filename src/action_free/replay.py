"""Open-loop replay of recorded episodes and episode videos.

Replaying the stored actions from the episode's seed must reproduce the stored frames and
simulator state step by step. This checks the alignment the IDM and the policy rely on:
`observations[t + 1]` is the result of applying `actions[t]` to `observations[t]`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from action_free.collect import make_env


def episode_object(episode_path: Path) -> tuple[str, str, str]:
    """Environment name, object and scene of an episode, from its directory's manifest."""
    cfg = json.loads((Path(episode_path).parent / "manifest.json").read_text())["config"]
    return cfg["env_name"], cfg["object_name"], cfg.get("scene", "default")


def replay(episode: dict, env_name: str, object_name: str, scene: str = "default") -> dict:
    """Step the recorded actions from the recorded seed; return replayed arrays and checks."""
    seed = int(episode["seed"])
    actions = np.asarray(episode["actions"])
    env = make_env(env_name, object_name, scene=scene)
    np.random.seed(seed)  # Same global RNG state as during collection.
    ob, info = env.reset(seed=seed)
    frames, qpos, success, terminated_at = [ob], [], False, None
    try:
        for t, action in enumerate(actions):
            ob, _, terminated, truncated, info = env.step(action)
            frames.append(ob)
            qpos.append(info["prev_qpos"])
            success = success or bool(info["success"])
            if (terminated or truncated) and terminated_at is None:
                terminated_at = t
    finally:
        env.close()

    frames = np.asarray(frames, dtype=np.uint8)
    qpos = np.asarray(qpos, dtype=np.float32)
    recorded = np.asarray(episode["observations"])
    diff = np.abs(frames.astype(np.int16) - recorded.astype(np.int16)).max(-1)  # (T+1, C, H, W)
    # GPU (EGL) rendering is not bit-exact: a few shadow-edge pixels change between two
    # renders of the same state. Compare against the gap between consecutive recorded
    # frames, which an off-by-one misalignment would produce.
    changed = (diff > 10).mean(axis=(1, 2, 3))
    motion = (np.abs(np.diff(recorded.astype(np.int16), axis=0)).max(-1) > 10).mean(axis=(1, 2, 3))
    max_qpos_diff = float(np.abs(qpos - np.asarray(episode["qpos"])).max())
    return {
        "frames": frames,
        "checks": {
            "seed": seed,
            "steps": len(actions),
            "recorded_success": bool(episode["success"]),
            "replayed_success": success,
            "terminated_at_last_step": terminated_at == len(actions) - 1,
            "frames_identical": bool(diff.max() == 0),
            "max_changed_pixel_fraction": float(changed.max()),
            "median_motion_pixel_fraction": float(np.median(motion)),
            "frames_match": bool(changed.max() < 0.1 * max(np.median(motion), 1e-3)),
            "max_qpos_diff": max_qpos_diff,
            "state_matches": max_qpos_diff < 1e-4,
        },
    }


def tile(observation: np.ndarray, gap: int = 4) -> np.ndarray:
    """Place the camera views of one observation (C, H, W, 3) side by side."""
    c, h, w, _ = observation.shape
    canvas = np.full((h, c * w + (c - 1) * gap, 3), 255, dtype=np.uint8)
    for i, view in enumerate(observation):
        canvas[:, i * (w + gap) : i * (w + gap) + w] = view
    return canvas


def write_video(frames: np.ndarray, path: Path, fps: float = 10.0) -> Path:
    """Write (T, H, W, 3) uint8 frames as MP4 with ffmpeg, or as GIF without it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    if shutil.which("ffmpeg") is None:
        from PIL import Image

        path = path.with_suffix(".gif")
        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=1000 / fps, loop=0)
        return path
    # Even dimensions for yuv420p.
    frames = np.pad(frames, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)), constant_values=255)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frames.shape[2]}x{frames.shape[1]}",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path),
    ]  # fmt: skip
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    return path
