"""Replay a recorded episode open loop and write a video of it.

Steps the stored actions from the episode's seed in a fresh environment and checks that
frames, simulator state and success match the recording. The video shows the recorded
frames (top) and the replayed frames (bottom), front, side and wrist cameras left to right.

    MUJOCO_GL=egl uv run python scripts/replay_episode.py data/pilot/episode_00000.npz
    MUJOCO_GL=egl uv run python scripts/replay_episode.py data/pilot/episode_00000.npz --no-replay

Exits with status 1 if the replay does not reproduce the recording.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from action_free.replay import episode_object, replay, tile, write_video


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("episode", type=Path, help="episode_XXXXX.npz written by collect_demos.py")
    p.add_argument("--out", type=Path, default=None, help="Default: outputs/replay/<dir>_<ep>.mp4")
    p.add_argument("--no-replay", action="store_true", help="Only write the recorded video.")
    p.add_argument("--fps", type=float, default=10.0, help="10 = real time at 10 Hz control.")
    a = p.parse_args()

    episode = dict(np.load(a.episode))
    env_name, object_name, scene = episode_object(a.episode)
    recorded = np.stack([tile(o) for o in episode["observations"]])
    out = a.out or Path("outputs/replay") / f"{a.episode.parent.name}_{a.episode.stem}.mp4"
    print(
        f"{a.episode}: object={object_name} seed={int(episode['seed'])} "
        f"steps={int(episode['length'])} success={bool(episode['success'])}"
    )

    if a.no_replay:
        print(f"video -> {write_video(recorded, out, a.fps)}")
        return

    result = replay(episode, env_name, object_name, scene)
    replayed = np.stack([tile(o) for o in result["frames"]])
    gap = np.full((len(recorded), 6, recorded.shape[2], 3), 255, dtype=np.uint8)
    print(f"video -> {write_video(np.concatenate([recorded, gap, replayed], axis=1), out, a.fps)}")

    checks = result["checks"]
    print(json.dumps(checks, indent=2))
    ok = (
        checks["frames_match"]
        and checks["state_matches"]
        and checks["replayed_success"] == checks["recorded_success"]
        and checks["terminated_at_last_step"]
    )
    print("REPLAY OK" if ok else "REPLAY MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
