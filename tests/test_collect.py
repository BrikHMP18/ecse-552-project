"""Collection is reproducible per seed and stores aligned episode arrays.

Uses the state-based variant of the lite single-block task, which needs no rendering.
"""

import json

import numpy as np

from action_free.collect import CollectConfig, collect, make_env, run_episode
from action_free.objects import OBJECTS

STATE_ENV = "block-cpu-lite-single-task1-v0"
VISUAL_ENV = "visual-block-cpu-lite-single-task1-v0"


def test_same_seed_same_episode():
    env = make_env(STATE_ENV)
    first, meta_a = run_episode(env, STATE_ENV, seed=7)
    second, meta_b = run_episode(env, STATE_ENV, seed=7)
    other, _ = run_episode(env, STATE_ENV, seed=8)
    env.close()

    assert meta_a == meta_b
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["observations"][0], other["observations"][0])


def test_collect_writes_aligned_episodes(tmp_path):
    summary = collect(CollectConfig(out_dir=tmp_path, env_name=STATE_ENV, num_episodes=2))

    assert summary["num_kept"] == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    kept = [e for e in manifest["episodes"] if e["kept"]]
    assert [e["file"] for e in kept] == ["episode_00000.npz", "episode_00001.npz"]

    for entry in kept:
        ep = np.load(tmp_path / entry["file"])
        steps = int(ep["length"])
        assert ep["actions"].shape == (steps, 7)
        assert ep["observations"].shape[0] == steps + 1  # includes the final frame
        assert np.all(np.abs(ep["actions"]) <= 1.0)
        assert int(ep["seed"]) == entry["seed"]
        assert ep["terminals"][-1] and not ep["terminals"][:-1].any()


def _cube_rgb(env):
    u = env.unwrapped
    return u._model.geom(u._cube_geom_ids_list[0][0]).rgba[:3].copy()


def test_objects_follow_proposal_colours_and_sizes():
    """Red: source, B, C; blue: A, D. Cubes of 6 cm; blocks of 9 x 6 x 6 and 9 x 7 x 6 cm."""
    sizes = {"source": (6, 6, 6), "A": (6, 6, 6), "B": (9, 6, 6), "C": (9, 7, 6), "D": (9, 7, 6)}
    assert set(sizes) == set(OBJECTS)
    for name, size in sizes.items():
        env = make_env(STATE_ENV, name)
        env.reset(seed=0)
        u = env.unwrapped
        red, _, blue = _cube_rgb(env)
        assert (blue > red) == (name in ("A", "D"))
        np.testing.assert_allclose(u._model.geom_size[u._cube_geom_ids_list[0][0]] * 200, size)
        env.close()


def test_same_seed_same_initial_state_for_every_object():
    states = {}
    for name in OBJECTS:
        env = make_env(STATE_ENV, name)
        states[name] = []
        for seed in range(5):
            env.reset(seed=seed)
            states[name].append(env.unwrapped._data.qpos.copy())
        env.close()
    for name in OBJECTS:
        np.testing.assert_array_equal(states[name], states["source"])
    assert not np.array_equal(states["source"][0], states["source"][1])


def test_blocks_render():
    for name, red_block in (("B", True), ("C", True), ("D", False)):
        env = make_env(VISUAL_ENV, name)
        ob, _ = env.reset(seed=0)
        env.close()
        assert ob.shape == (3, 224, 224, 3)
        r, g, b = (ob[0, ..., c].astype(int) for c in range(3))  # front camera
        block = (r > g + 80) & (r > b + 80) if red_block else (b > r + 80) & (b > g + 40)
        assert block.sum() > 50


def test_expert_moves_the_blocks():
    for name in ("B", "C"):
        env = make_env(STATE_ENV, name)
        assert all(run_episode(env, STATE_ENV, seed)[1]["success"] for seed in range(3))
        env.close()


def test_deterministic_expert_never_turns_the_first_grasp():
    """Stock OCBench turns the cube grasp by 90 degrees in 30% of episodes; ours never does."""
    from action_free.collect import make_expert

    env = make_env(STATE_ENV)
    for seed in range(10):
        np.random.seed(seed)
        ob, info = env.reset(seed=seed)
        expert = make_expert(env, STATE_ENV)
        expert.reset(ob, info, transition_idx=0)
        for t in range(60):
            ob, _, _, _, info = env.step(np.clip(expert.select_action(ob, info, t), -1, 1))
            if info["privileged/block_0_pos"][2] > 0.04:  # lifted
                break
        assert abs(np.sin(info["proprio/effector_yaw"][0])) < 0.26  # within 15 degrees
    env.close()


def test_success_uses_the_block_size():
    """2 cm off the goal, the 6 cm cube is inside the 12 cm square; the 9 cm block is not."""
    inside = {}
    for name in ("source", "B"):
        env = make_env(STATE_ENV, name)
        env.reset(seed=0)
        inside[name] = _place_cube(env, (0.445, 0.0, 0.03)).goal_marker_conditions(0)["inside"]
        env.close()
    assert inside == {"source": True, "B": False}


def _place_cube(env, xyz):
    import mujoco

    u = env.unwrapped
    u._data.joint("object_joint_0").qpos[:7] = [*xyz, 1.0, 0.0, 0.0, 0.0]
    u._data.joint("object_joint_0").qvel[:] = 0.0
    mujoco.mj_forward(u._model, u._data)
    return u


def test_goal_marker_is_in_every_scene():
    for name in OBJECTS:
        env = make_env(STATE_ENV, name)
        env.reset(seed=0)
        names = {env.unwrapped._model.geom(g).name for g in range(env.unwrapped._model.ngeom)}
        env.close()
        assert {"goal_marker_px", "goal_marker_nx", "goal_marker_py", "goal_marker_ny"} <= names


def test_cube_never_starts_on_the_marker():
    env = make_env(STATE_ENV)
    goal = np.array([0.425, 0.0])
    for seed in range(50):
        env.reset(seed=seed)
        xy = env.unwrapped._data.joint("object_joint_0").qpos[:2]
        assert np.abs(xy - goal).max() > 0.06 + 0.01 + 0.03 * np.sqrt(2)
    env.close()


def test_strict_success_conditions():
    env = make_env(STATE_ENV)
    env.reset(seed=0)
    cond = _place_cube(env, (0.425, 0.0, 0.03)).goal_marker_conditions(0)
    assert cond == {"inside": True, "resting": True, "released": True, "still": True}
    assert not _place_cube(env, (0.425, 0.0, 0.08)).goal_marker_conditions(0)["resting"]
    assert not _place_cube(env, (0.475, 0.0, 0.03)).goal_marker_conditions(0)["inside"]
    env.close()


def test_success_needs_one_second_of_holding():
    env = make_env(STATE_ENV)
    env.reset(seed=0)
    _place_cube(env, (0.425, 0.0, 0.03))
    hold_steps = round(1.0 / env.unwrapped._control_timestep)  # 10 at 10 Hz
    successes = [bool(env.step(np.zeros(7))[4]["success"]) for _ in range(hold_steps)]
    env.close()
    assert successes == [False] * (hold_steps - 1) + [True]
