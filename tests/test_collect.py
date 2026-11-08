"""Collection is reproducible per seed and stores aligned episode arrays.

Uses the state-based variant of the lite single-block task, which needs no rendering.
"""

import json

import numpy as np

from action_free.collect import CollectConfig, collect, make_env, run_episode

STATE_ENV = "block-cpu-lite-single-task1-v0"


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


def test_objects_follow_proposal_colours():
    """Source is the red cube and target A the blue cube, as in the proposal."""
    rgb = {}
    for name in ("source", "A"):
        env = make_env(STATE_ENV, name)
        env.reset(seed=0)
        rgb[name] = _cube_rgb(env)
        env.close()
    assert rgb["source"][0] > rgb["source"][2]  # red dominates blue
    assert rgb["A"][2] > rgb["A"][0]  # blue dominates red


def test_unimplemented_targets_fail_loudly():
    import pytest

    for name in ("B", "C", "D"):
        with pytest.raises(NotImplementedError):
            make_env(STATE_ENV, name)
