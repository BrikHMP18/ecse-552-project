"""Local web app: watch a policy rollout live, as the policy sees it.

    MUJOCO_GL=egl uv run --with gradio python scripts/play_policy.py   # http://localhost:7860

Choose a run, a checkpoint, an object and an initial state; "Run" resets the environment
and plays one rollout through `evaluate.rollout`, the same code as the reported numbers,
streamed in real time (10 Hz): the front, side and wrist cameras and the simulator's four
success conditions at every step. "Stop" ends the rollout; afterwards the whole rollout can
be replayed as a video.

Initial states come from seeds: the start of a demonstration the checkpoint trained on, of
one in its validation split, or a new one from the validation rollout seeds (900,000+).
Test seeds (1,000,000+) are never offered. The result says whether the object and the start
were in the checkpoint's training data.
"""

# ruff: noqa: E501
import csv
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from action_free.collect import DEFAULT_ENV, make_env
from action_free.evaluate import DEFAULT_EXECUTION, VALIDATION_BASE_SEED, load_policy, rollout
from action_free.objects import OBJECTS
from action_free.replay import tile, write_video

OUT = Path("outputs/play")
STARTS = ["new (validation rollout seeds)", "training demo", "validation demo"]
WORKER = ThreadPoolExecutor(1)  # the EGL context and the models live in one thread
cache: dict = {}  # the last policy and environment, reused while the choices do not change
stop = threading.Event()
LIGHT = "() => document.body.classList.remove('dark')"  # the pastel palette is a light one
THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Roboto"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "monospace"],
).set(
    body_background_fill="#f8f6fc",
    block_background_fill="#ffffff",
    block_label_background_fill="#e6dffa",
    block_label_text_color="#4a4063",
    block_title_background_fill="#e6dffa",
    block_title_text_color="#4a4063",
    button_primary_background_fill="#c9bcf2",
    button_primary_background_fill_hover="#b8a9e8",
    button_primary_text_color="#2f2a3d",
    button_secondary_background_fill="#d7e8f7",
    button_secondary_background_fill_hover="#c5dcf0",
    button_secondary_text_color="#2f2a3d",
    button_cancel_background_fill="#f6cccc",
    button_cancel_background_fill_hover="#efb8b8",
    button_cancel_text_color="#6b1f1f",
)


class LiveEnv:
    """Passes an environment through to `rollout`, publishing every frame at real-time pace."""

    def __init__(self, env, frames: queue.Queue, speed: float):
        self.env, self.unwrapped, self.frames, self.dt = env, env.unwrapped, frames, 0.1 / speed
        self.steps = 0

    def publish(self, ob):
        self.frames.put((big(ob), self.steps, self.unwrapped.goal_marker_conditions(0)))

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        self.publish(ob)
        return ob, info

    def step(self, action):
        t0 = time.perf_counter()
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.publish(ob)
        time.sleep(max(0.0, self.dt - (time.perf_counter() - t0)))
        return ob, reward, terminated, truncated or stop.is_set(), info


def big(ob: np.ndarray) -> np.ndarray:
    return tile(ob).repeat(2, axis=0).repeat(2, axis=1)


def conditions(c: dict) -> str:
    """The four success conditions as coloured chips."""
    chip = "display:inline-block;margin:6px 6px 0 0;padding:3px 10px;border-radius:999px;font-size:0.9em;color:#2f2a3d"
    return "".join(
        f"<span style='{chip};background:{'#cdebd3' if v else '#f6cccc'}'>{k}: {'yes' if v else 'no'}</span>"
        for k, v in c.items()
    )


def banner(text: str, color: str, c: dict | None = None) -> str:
    return (
        f"<div style='padding:10px 14px;border-radius:10px;font-weight:600;font-size:1.1em;"
        f"color:#2f2a3d;background:{color}'>{text}</div>" + (conditions(c) if c else "")
    )


def demo_seeds(checkpoint: str, object_name: str, split: str) -> list[int]:
    """Seeds of the demonstrations of an object in a checkpoint's train or val split."""
    data = json.loads((Path(checkpoint).parent / "act_run.json").read_text())["data"]
    seeds = []
    for c in data["collections"]:
        if c["object"] != object_name:
            continue
        name = Path(c["dir"]).name
        with open(Path("data") / name / "initial_states.csv") as f:
            seed_of = {r["file"]: int(r["seed"]) for r in csv.DictReader(f)}
        seeds += [seed_of[Path(x).name] for x in data[f"{split}_files"] if x.startswith(name + "/")]
    return seeds


def run(checkpoint, object_name, seed, start, execution, speed, frames: queue.Queue):
    """One rollout in the worker thread; returns the replay video, the banner and the details."""
    stop.clear()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cache.get("checkpoint") != checkpoint:
        cache.update(checkpoint=checkpoint, policy=load_policy(checkpoint, device))
    policy = cache["policy"]
    data = policy.run["data"]
    scene = data.get("scene_name") or "random_yaw"
    if cache.get("env_key") != (object_name, scene):
        if "env" in cache:
            cache["env"].close()
        cache.update(
            env_key=(object_name, scene), env=make_env(DEFAULT_ENV, object_name, scene=scene)
        )
    env = LiveEnv(cache["env"], frames, speed)
    r = rollout(env, policy, seed, device, keep_frames=True, execution=execution)
    video = write_video(np.stack([big(o) for o in r["frames"]]), OUT / "rollout.mp4")

    trained = sorted({c["object"] for c in data["collections"]})
    xy = np.array([r["start"]["cube_x"], r["start"]["cube_y"]])
    nearest = np.linalg.norm(np.array(data["train_initial_states"])[:, :2] - xy, axis=1).min()
    outcome = "SUCCESS" if r["success"] else "STOPPED" if stop.is_set() else "FAIL"
    color = {"SUCCESS": "#cdebd3", "STOPPED": "#e4e4ea", "FAIL": "#f6cccc"}[outcome]
    text = f"{outcome} in {r['length']} steps ({r['length'] / 10:.1f} s) · seed {seed}"
    details = f"""| | |
|---|---|
| Object | **{object_name}**: {"in the training data" if object_name in trained else "**NOT in the training data**"} (trained on {", ".join(trained)}) |
| Start | {start}; nearest training start {nearest * 100:.1f} cm away |
| Highest object position | {r["end"]["max_cube_z"] * 100:.1f} cm ({"lifted" if r["end"]["max_cube_z"] > 0.06 else "never lifted"}) |
| Inference latency (p95) | {np.percentile(r["latencies_s"], 95) * 1000:.0f} ms |"""
    return str(video), banner(text, color, r["end"]["conditions"]), details


def trained_on(run_dir: str) -> str:
    data = json.loads((Path(run_dir) / "act_run.json").read_text())["data"]
    parts = [
        f"{c['num_episodes']} {c['object']} ({c['actions']} actions)" for c in data["collections"]
    ]
    return "Trained on " + " + ".join(parts)


def play(run_dir, name, object_name, start, index, execution, speed):
    """Yields (live image, banner, details, replay video) while the rollout runs."""
    if not run_dir:
        raise gr.Error("No run with checkpoints under outputs/.")
    checkpoint, object_name = str(Path(run_dir) / f"{name}.ckpt"), object_name.split(":")[0]
    if start == STARTS[0]:
        seed = VALIDATION_BASE_SEED + int(index)
    else:
        seeds = demo_seeds(checkpoint, object_name, "train" if start == STARTS[1] else "val")
        if not seeds:
            raise gr.Error(f"This checkpoint has no {start} of object {object_name}.")
        seed = seeds[int(index) % len(seeds)]
    stop.set()  # ends a rollout that is still running
    frames: queue.Queue = queue.Queue()
    job = WORKER.submit(run, checkpoint, object_name, seed, start, execution, float(speed), frames)
    yield None, banner(f"Loading · seed {seed}", "#cfe3ff"), "", None
    while not (job.done() and frames.empty()):
        try:
            image, step, c = frames.get(timeout=0.2)
        except queue.Empty:
            continue
        while not frames.empty():  # the browser fell behind: show the newest frame
            image, step, c = frames.get()
        text = f"Running · step {step} ({step / 10:.1f} s) · seed {seed}"
        yield image, banner(text, "#cfe3ff", c), gr.skip(), gr.skip()
    video, text, details = job.result()
    yield gr.skip(), text, details, video


def play_next(run_dir, name, object_name, start, index, execution, speed):
    for out in play(run_dir, name, object_name, start, int(index) + 1, execution, speed):
        yield (int(index) + 1, *out)


def main() -> None:
    runs = sorted({str(p.parent) for p in Path("outputs").glob("**/selected.ckpt")})
    objects = [f"{o.name}: {o.description}" for o in OBJECTS.values()]
    with gr.Blocks(title="ECSE 552 policy rollouts") as app:
        gr.Markdown(
            "## Policy rollouts\nPick a run, an object and an initial state; the policy acts live "
            "from the three camera images, with the same code as the reported evaluations."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                with gr.Group():
                    run_dir = gr.Dropdown(runs, value=runs[0] if runs else None, label="Run")
                    name = gr.Radio(
                        ["selected", "policy_best", "policy_last"],
                        value="selected",
                        label="Checkpoint",
                    )
                info = gr.Markdown(trained_on(runs[0]) if runs else "")
                with gr.Group():
                    object_name = gr.Dropdown(objects, value=objects[0], label="Object")
                    start = gr.Radio(STARTS, value=STARTS[0], label="Initial state")
                    index = gr.Number(
                        0,
                        precision=0,
                        minimum=0,
                        maximum=99_999,
                        label="Episode #",
                        info="Position in the chosen list of initial states; each one is another start.",
                    )
                run_button = gr.Button("Run", variant="primary", size="lg")
                with gr.Row():
                    next_button = gr.Button("Next episode", min_width=120)
                    stop_button = gr.Button("Stop", variant="stop", min_width=120)
                with gr.Accordion("Advanced", open=False):
                    speed = gr.Slider(0.25, 4, value=1, step=0.25, label="Speed (1 = real time)")
                    execution = gr.Textbox(
                        DEFAULT_EXECUTION, label="Execution: chunk, replan:N, ensemble:M"
                    )
            with gr.Column(scale=2):
                badge = gr.HTML(banner("Choose a run and press Run", "#e4e4ea"))
                image = gr.Image(
                    label="Live: front · side · wrist", interactive=False, show_label=True
                )
                details = gr.Markdown()
                with gr.Accordion("Replay of the finished rollout", open=False):
                    video = gr.Video(label="Front · side · wrist")
        inputs = [run_dir, name, object_name, start, index, execution, speed]
        outputs = [image, badge, details, video]
        run_dir.change(trained_on, run_dir, info)
        run_button.click(play, inputs, outputs)
        next_button.click(play_next, inputs, [index, *outputs])
        stop_button.click(stop.set)
    app.launch(theme=THEME, js=LIGHT)


if __name__ == "__main__":
    main()
