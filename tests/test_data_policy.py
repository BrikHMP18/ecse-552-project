"""Episode splits, model inputs, exact replay, initial states, actions, and the ACT adapter."""

import numpy as np
import pytest
import torch

from action_free.collect import CollectConfig, collect, make_env, run_episode
from action_free.data import (
    split_of,
)
from action_free.objects import SCENE_KWARGS, env_kwargs

STATE_ENV = "block-cpu-lite-single-task1-v0"


def test_splits_are_80_20_and_stable_as_the_set_grows():
    for n in (100, 200, 300):
        counts = {s: sum(split_of(i) == s for i in range(n)) for s in ("train", "val")}
        assert counts == {"train": 0.8 * n, "val": 0.2 * n}
    # Assignments depend only on the episode index, so 100 demos are a subset of 200 and 300.
    blocks = [[split_of(10 * b + k) for k in range(10)] for b in range(30)]
    assert all(sorted(b) == sorted(["train"] * 8 + ["val"] * 2) for b in blocks)
    assert len({tuple(b) for b in blocks}) > 1  # the permutation differs between blocks


@pytest.fixture(scope="module")
def tiny_collection(tmp_path_factory):
    out = tmp_path_factory.mktemp("episodes")
    collect(CollectConfig(out_dir=out, env_name=STATE_ENV, num_episodes=10))
    return out


def test_replaying_stored_actions_reproduces_the_episode_exactly():
    env = make_env(STATE_ENV)
    for seed in range(5):
        ep, meta = run_episode(env, STATE_ENV, seed)
        np.random.seed(seed)
        env.reset(seed=seed)
        qpos, success = [], False
        for action in ep["actions"]:  # float32, as stored
            _, _, _, _, info = env.step(action)
            qpos.append(info["prev_qpos"])
            success = success or bool(info["success"])
        np.testing.assert_array_equal(np.asarray(qpos, np.float32), ep["qpos"])
        assert success == meta["success"]
    env.close()


def test_initial_states_use_the_start_area_and_a_fixed_arm_pose():
    (x0, y0), (x1, y1) = SCENE_KWARGS["object_sampling_bounds"]
    env = make_env(STATE_ENV)
    u = env.unwrapped
    arm, effector = [], []
    for seed in range(30):
        _, info = env.reset(seed=seed)
        x, y = u._data.joint("object_joint_0").qpos[:2]
        assert x0 <= x <= x1 and y0 <= y <= y1
        arm.append(u._data.qpos[u._arm_joint_ids].copy())
        effector.append(info["proprio/effector_pos"].copy())
    env.close()
    np.testing.assert_allclose(np.ptp(np.asarray(arm), axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(effector[0], SCENE_KWARGS["arm_init_pose"][:3], atol=5e-3)


def test_initial_states_csv_matches_the_simulator(tiny_collection):
    from action_free.initial_states import current, read_csv

    rows = read_csv(tiny_collection)
    assert len(rows) == 10
    env = make_env(STATE_ENV)
    for r in rows:
        env.reset(seed=int(r["seed"]))
        live = current(env)
        for k in ("cube_x", "cube_y", "cube_yaw_mod90_deg", "arm_q0"):
            assert abs(live[k] - r[k]) < 1e-4, (k, live[k], r[k])
    env.close()


def test_evaluation_flags_out_of_distribution_starts_and_objects():
    from action_free.evaluate import compare_to_training

    train = {
        "object": "source",
        "scene": json_roundtrip(env_kwargs("source")),
        "train_initial_states": [[0.40, 0.25, 10.0], [0.45, 0.30, 50.0]],
    }
    episodes = [
        {"start": {"cube_x": 0.40, "cube_y": 0.25}},  # a training start
        {"start": {"cube_x": 0.40, "cube_y": -0.25}},  # other side of the table
    ]
    out = compare_to_training(train, "A", episodes)
    assert not out["same_object"] and set(out["scene_differences"]) == {"cube_colors"}
    assert [e["start"]["inside_train_area"] for e in episodes] == [True, False]
    assert episodes[0]["start"]["nearest_train_xy_cm"] == 0.0
    assert episodes[1]["start"]["nearest_train_xy_cm"] > 40
    assert out["starts_inside_train_area"] == 0.5


def json_roundtrip(x):
    import json

    return json.loads(json.dumps(x))


def test_collection_records_expert_detours(tiny_collection):
    import json

    from action_free.initial_states import read_csv

    manifest = json.loads((tiny_collection / "manifest.json").read_text())
    assert all("num_detours" in e for e in manifest["episodes"])
    assert all(r["num_detours"] != "" for r in read_csv(tiny_collection))


def _fake_image_collection(root, num_episodes=10, steps=12, size=32):
    """Tiny episodes with random images, enough to run the training loop on CPU."""
    import json

    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(num_episodes):
        np.savez_compressed(
            root / f"episode_{i:05d}.npz",
            observations=rng.integers(0, 255, (steps + 1, 3, size, size, 3), dtype=np.uint8),
            actions=rng.uniform(-1, 1, (steps, 7)).astype(np.float32),
            qpos=rng.uniform(-1, 1, (steps, 21)).astype(np.float32),
            seed=i,
            success=True,
            length=steps,
        )
    manifest = {"config": {"env_name": "fake", "object_name": "source"}, "episodes": []}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_checkpoint_selection_rule():
    from action_free.evaluate import choose_checkpoint

    assert choose_checkpoint(0.80, 0.85) == "last"
    assert choose_checkpoint(0.85, 0.80) == "best"
    assert choose_checkpoint(0.80, 0.80) == "best"  # ties go to best.pt


def test_action_scales_match_ocbench():
    from action_free.actions import ARM_SCALE, GRIPPER_QPOS, GRIPPER_SCALE

    env = make_env(STATE_ENV)
    u = env.unwrapped
    np.testing.assert_allclose(u._joint_action_delta, [*ARM_SCALE, GRIPPER_SCALE])
    assert u._gripper_opening_joint_id == GRIPPER_QPOS and list(u._arm_joint_ids) == list(range(6))
    env.close()


def test_absolute_targets_round_trip_to_the_stored_deltas():
    from action_free.actions import GRIPPER_SCALE, gripper_opening, to_absolute, to_delta

    env = make_env(STATE_ENV)
    ep, _ = run_episode(env, STATE_ENV, seed=3)
    env.close()
    targets = to_absolute(ep["actions"], ep["qpos"])
    back = to_delta(targets, ep["qpos"])
    np.testing.assert_allclose(back[:, :6], ep["actions"][:, :6], atol=1e-6)
    # The gripper target is clipped to [0, 1], as the environment does; the effect is the same.
    opening = gripper_opening(ep["qpos"])
    effect = lambda a: np.clip(opening + a[:, 6] * GRIPPER_SCALE, 0, 1)  # noqa: E731
    np.testing.assert_allclose(effect(back), effect(ep["actions"]), atol=1e-6)
    assert (targets[:, 6] >= 0).all() and (targets[:, 6] <= 1).all()


def test_execution_modes_parse():
    from action_free.evaluate import parse_execution

    assert parse_execution("chunk", 10) == ("replan", 10)
    assert parse_execution("replan:1", 10) == ("replan", 1)
    assert parse_execution("ensemble:0.1", 10) == ("ensemble", 0.1)
    for bad in ("replan:0", "replan:11", "smooth"):
        with pytest.raises(ValueError):
            parse_execution(bad, 10)


def test_act_config_matches_upstream_cli():
    import tomllib
    from pathlib import Path

    from action_free.act_original import _argv, _cli, _import_upstream, act_args

    cfg = tomllib.loads(Path("configs/act_deterministic.toml").read_text())
    args = act_args("ckpt", 0, cfg)
    assert args["chunk_size"] == 20 and args["num_epochs"] == 2000 and args["kl_weight"] == 10
    _import_upstream()
    from detr.main import get_args_parser

    with _argv(_cli(args)):
        parsed = get_args_parser().parse_args()  # upstream re-parses sys.argv when building
    assert parsed.chunk_size == 20 and parsed.policy_class == "ACT"


def test_export_writes_act_layout(tmp_path):
    import h5py

    from action_free.act_original import ACTION_DIM, CAMERAS, STATE_DIM, export
    from action_free.actions import to_absolute

    data = _fake_image_collection(tmp_path / "data", num_episodes=3)
    files = sorted(data.glob("episode_*.npz"))
    index = export(files, tmp_path / "act")
    assert [e["file"] for e in index] == [f.name for f in files]
    ep = np.load(files[1])
    with h5py.File(tmp_path / "act" / "episode_1.hdf5") as h:
        assert h.attrs["sim"]
        assert set(h["observations/images"]) == set(CAMERAS)
        np.testing.assert_array_equal(h["observations/images/side"][()], ep["observations"][:12, 1])
        assert h["action"].shape == (12, STATE_DIM)
        np.testing.assert_allclose(
            h["action"][:, :ACTION_DIM], to_absolute(ep["actions"], ep["qpos"])
        )
        assert not h["action"][:, ACTION_DIM:].any() and not h["observations/qpos"][()].any()


def test_export_refuses_test_collections(tmp_path):
    import json

    from action_free.act_original import export

    data = _fake_image_collection(tmp_path / "data", num_episodes=1)
    m = json.loads((data / "manifest.json").read_text())
    m["config"]["role"] = "test"
    (data / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="held-out test"):
        export(sorted(data.glob("episode_*.npz")), tmp_path / "act")


def test_upstream_dataset_handles_variable_length_episodes(tmp_path):
    from action_free.act_original import (
        CAMERAS,
        MAX_EPISODE_LEN,
        STATE_DIM,
        _import_upstream,
        _patch_variable_length,
        export,
    )  # fmt: skip

    data = _fake_image_collection(tmp_path / "data", num_episodes=4)
    files = sorted(data.glob("episode_*.npz"))
    export(files, tmp_path / "act")
    utils = _import_upstream()
    _patch_variable_length(utils)
    stats = utils.get_norm_stats(tmp_path / "act", 4)
    assert stats["action_mean"].shape == (STATE_DIM,) and (stats["action_std"] >= 1e-2).all()
    ds = utils.EpisodicDataset([0, 1], str(tmp_path / "act"), list(CAMERAS), stats)
    image, qpos, action, is_pad = ds[0]
    assert image.shape == (3, 3, 32, 32) and qpos.shape == (STATE_DIM,)
    assert action.shape == (MAX_EPISODE_LEN, STATE_DIM) and is_pad.shape == (MAX_EPISODE_LEN,)
    assert is_pad[12:].all()  # nothing beyond the 12-step episode counts in the loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="upstream ACT trains with .cuda()")
def test_original_act_trains_and_loads(tmp_path):
    import json

    from action_free.act_original import ActPolicy, export, train

    data = _fake_image_collection(tmp_path / "data")
    export(sorted(data.glob("episode_*.npz")), tmp_path / "act")
    cfg = {"num_epochs": 2, "batch_size": 2, "lr": 1e-5, "kl_weight": 10, "chunk_size": 5,
           "hidden_dim": 64, "dim_feedforward": 64}  # fmt: skip
    record = train(tmp_path / "act", tmp_path / "run", 10, 0, {**cfg, "use_robot_state": True})
    run = tmp_path / "run"
    for name in ("policy_best.ckpt", "policy_last.ckpt", "dataset_stats.pkl", "act_run.json"):
        assert (run / name).exists(), name
    assert sorted(record["split"]["train"] + record["split"]["val"]) == list(range(10))
    assert len(json.loads((run / "history.json").read_text())) == 2
    policy = ActPolicy(run / "policy_last.ckpt", torch.device("cuda"))
    images = torch.randint(0, 255, (1, 3, 32, 32, 3), dtype=torch.uint8, device="cuda")
    assert policy.uses_robot_state  # robot-state reference path
    assert policy.predict(images, np.zeros(21)).shape == (1, 5, 7)
    with pytest.raises(ValueError, match="robot state"):
        policy.predict(images)


def test_detour_files_reads_the_annotations(tmp_path):
    import csv

    from action_free.data import detour_files
    from action_free.initial_states import FIELDS

    with open(tmp_path / "initial_states.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i in range(4):
            row = {k: 0.0 for k in FIELDS if k.startswith(("cube_", "arm_"))}
            w.writerow({**row, "file": f"episode_{i:05d}.npz", "num_detours": int(i == 2)})
    assert detour_files(tmp_path) == {"episode_00002.npz"}


def test_fixed_yaw_scene_and_deterministic_expert():
    from action_free.initial_states import current

    yaws = {}
    for scene in ("random_yaw", "fixed_yaw"):
        env = make_env(STATE_ENV, scene=scene)
        yaws[scene] = []
        for seed in range(10):
            env.reset(seed=seed)
            yaws[scene].append(current(env)["cube_yaw_deg"])
        env.close()
    assert all(abs(y) < 1e-6 for y in yaws["fixed_yaw"])
    assert max(yaws["random_yaw"]) - min(yaws["random_yaw"]) > 30  # stock behaviour unchanged
    env = make_env(STATE_ENV, scene="fixed_yaw")
    metas = [run_episode(env, STATE_ENV, s, expert="deterministic")[1] for s in range(5)]
    env.close()
    assert all(m["num_detours"] == 0 for m in metas)
    assert len({round(m["speed"], 6) for m in metas}) == 1  # fixed speed profile


def test_export_with_robot_state(tmp_path):
    import h5py

    from action_free.act_original import export
    from action_free.actions import gripper_opening

    data = _fake_image_collection(tmp_path / "data", num_episodes=1)
    f = sorted(data.glob("episode_*.npz"))
    export(f, tmp_path / "act", use_robot_state=True)
    ep = np.load(f[0])
    with h5py.File(tmp_path / "act" / "episode_0.hdf5") as h:
        q = h["observations/qpos"][()]
    np.testing.assert_allclose(q[:, :6], ep["qpos"][:, :6])
    np.testing.assert_allclose(q[:, 6], gripper_opening(ep["qpos"]), rtol=1e-6)
    assert not q[:, 7:].any()
