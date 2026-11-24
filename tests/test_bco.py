"""Iterative BCO (g): rollout seeds, volume paths, and rollouts recorded as IDM episodes."""

import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from action_free.actions import to_absolute, to_delta
from action_free.bco import ROLLOUT_BASE_SEED, resolve, rollout_seeds, write_rollout
from action_free.collect import make_env
from action_free.evaluate import rollout
from action_free.idm import load_transitions, split_files

STATE_ENV = "block-cpu-lite-single-task1-v0"  # no rendering: the frames are state vectors


def test_rollout_seeds_follow_data_toml_and_overlap_no_other_range():
    seeds = tomllib.loads(Path("configs/data.toml").read_text())["seeds"]
    assert seeds["bco_rollouts"] == ROLLOUT_BASE_SEED
    r1, r2 = rollout_seeds(1), rollout_seeds(2)
    assert len(r1) == len(r2) == 200 and r1[-1] < r2[0]
    assert r1[0] > seeds["target_F_videos"] + 3 * 200  # F: 200 demos, at most 3 attempts each
    assert r2[-1] < seeds["validation"]
    assert rollout_seeds(1, 2) == r1[:2]


def test_volume_paths():
    assert resolve("bco:rollouts/D-r1-seed{seed}", 2) == "/bco/runs/rollouts/D-r1-seed2"
    assert resolve("bco:labels/x", 0, smoke=True) == "/bco/smoke/labels/x"
    assert (
        resolve("out:idm_bco_D_r1/seed{seed}/best.pt", 1) == "/outputs/idm_bco_D_r1/seed1/best.pt"
    )
    assert (
        resolve("data/D-deterministic-n200-seed400000", 0) == "data/D-deterministic-n200-seed400000"
    )


class FixedTarget:
    """A policy that always predicts the same absolute target, enough to drive the robot."""

    action_mode = "absolute"
    uses_robot_state = False

    def __init__(self, target: np.ndarray, chunk: int = 4):
        self.target, self.cfg = target, SimpleNamespace(chunk=chunk)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tensor(np.tile(self.target, (1, self.cfg.chunk, 1)), dtype=torch.float32)


def test_recorded_rollout_holds_the_actions_sent_and_the_state_before_each():
    env = make_env(STATE_ENV, "D")
    target = np.array([0.0, -1.5, 1.5, -1.5, -1.5, 0.0, 0.5], dtype=np.float32)
    try:
        r = rollout(env, FixedTarget(target), 800_000, torch.device("cpu"), record=True)
    finally:
        env.close()
    t = r["length"]
    assert len(r["frames"]) == t + 1 and r["actions"].shape == (t, 7) and len(r["qpos"]) == t
    # The action sent at each step is the target made relative to the recorded state.
    np.testing.assert_allclose(r["actions"], to_delta(target, r["qpos"]), atol=1e-5)
    assert isinstance(r["healthy"], bool)


def test_written_rollout_is_an_idm_episode(tmp_path):
    rng = np.random.default_rng(0)
    steps = 6
    r = {
        "frames": rng.integers(0, 255, (steps + 1, 3, 16, 16, 3), dtype=np.uint8),
        "actions": rng.uniform(-1, 1, (steps, 7)).astype(np.float32),
        "qpos": rng.uniform(-1, 1, (steps, 21)).astype(np.float32),
        "qvel": np.zeros((steps, 20), dtype=np.float32),
        "seed": 800_000,
        "success": False,
        "length": steps,
        "healthy": True,
    }
    write_rollout(tmp_path, 0, r)
    t = load_transitions(split_files(tmp_path), image_size=16)
    assert len(t.index) == steps
    # Labelled as a source demo: the action sent, made absolute with the robot's state.
    torch.testing.assert_close(t.targets, torch.from_numpy(to_absolute(r["actions"], r["qpos"])))
