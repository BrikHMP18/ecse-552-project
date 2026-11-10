"""Watch recorded episodes in the interactive MuJoCo viewer, with the recorded cameras on top.

Replays the stored actions from each episode's seed in the same scene (state variant of the
task, so no offscreen rendering), in real time. Along the top of the window it shows the
front, side and wrist images stored in the episode file for the current step: exactly what
the policy and the IDM see. The text box shows step, action and success.

Rotate/zoom with the mouse, pause with space, Tab / Shift+Tab for the side panels. Do not
Ctrl+drag during a replay: it applies forces to the scene and changes the outcome. The window stays
on the last frame until you close it (or pass --close-when-done).

    uv run python scripts/view_episode.py data/source-deterministic-n300-seed0/episode_00000.npz
    uv run python scripts/view_episode.py data/A-deterministic-n200-seed100000 --episodes 0 1

Needs a display (not MUJOCO_GL=egl).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from action_free.collect import make_env
from action_free.replay import episode_object, tile

CAMERAS = "front | side | wrist (recorded)"


def overlay(viewer, observation: np.ndarray, max_width_fraction: float = 0.9) -> None:
    """Draw the recorded camera views centred along the top of the viewer window."""
    vp = viewer.viewport
    if vp is None or vp.width <= 0 or vp.height <= 0:
        return
    strip = tile(observation)
    scale = min(
        1.0, max_width_fraction * vp.width / strip.shape[1], 0.4 * vp.height / strip.shape[0]
    )
    if scale < 1.0:
        size = (max(1, int(strip.shape[1] * scale)), max(1, int(strip.shape[0] * scale)))
        strip = np.asarray(Image.fromarray(strip).resize(size, Image.BILINEAR))
    h, w = strip.shape[:2]
    # The 3D viewport is offset inside the window when the side panels are shown.
    rect = mujoco.MjrRect(vp.left + (vp.width - w) // 2, vp.bottom + vp.height - h - 8, w, h)
    viewer.set_images((rect, strip))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", type=Path, help="An episode .npz or a collection directory.")
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="Indices in a directory.")
    p.add_argument("--speed", type=float, default=1.0, help="1 = real time (10 Hz).")
    p.add_argument("--pause", type=float, default=1.5, help="Seconds held after each episode.")
    p.add_argument(
        "--close-when-done",
        action="store_true",
        help="Close the window after the last episode (default: keep it open).",
    )
    a = p.parse_args()

    if a.path.is_dir():
        files = sorted(a.path.glob("episode_*.npz"))
        files = [files[i] for i in a.episodes] if a.episodes else files
    else:
        files = [a.path]
    env_name, object_name, scene = episode_object(files[0])
    # Same scene as the cameras see: no target overlay, no success recolouring.
    env = make_env(env_name.removeprefix("visual-"), object_name, scene=scene)
    env.unwrapped._visualize_info = False
    u = env.unwrapped
    dt = u._control_timestep / a.speed

    def show(viewer, ep, t, success, name):
        overlay(viewer, ep["observations"][t])
        steps = len(ep["actions"])
        action = ep["actions"][min(t, steps - 1)]
        viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "episode\nobject\nstep\naction\nsuccess",
                f"{name} (seed {int(ep['seed'])})\n{object_name}\n{t}/{steps}\n"
                + " ".join(f"{x:+.2f}" for x in action)
                + f"\n{success}",
            )
        )
        viewer.sync()

    # Side panels hidden for a larger scene; Tab / Shift+Tab show them.
    with mujoco.viewer.launch_passive(
        u._model, u._data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        for f in files:
            ep = {k: v for k, v in np.load(f).items()}
            seed = int(ep["seed"])
            np.random.seed(seed)
            env.reset(seed=seed)
            print(
                f"{f.name}: object={object_name} seed={seed} steps={len(ep['actions'])}", flush=True
            )
            success = False
            show(viewer, ep, 0, success, f.stem)
            for t, action in enumerate(ep["actions"]):
                if not viewer.is_running():
                    return
                t0 = time.time()
                _, _, _, _, info = env.step(action)
                success = success or bool(info["success"])
                show(viewer, ep, t + 1, success, f.stem)
                time.sleep(max(0.0, dt - (time.time() - t0)))
            print(f"  replayed success={success} (recorded {bool(ep['success'])})", flush=True)
            last = f == files[-1]
            end = float("inf") if last and not a.close_when_done else time.time() + a.pause
            while viewer.is_running() and time.time() < end:
                viewer.sync()
                time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()
    # The passive viewer's GLFW/OpenGL teardown segfaults at interpreter exit on this
    # machine (after all work is done); exit without running it.
    sys.stdout.flush()
    os._exit(0)
