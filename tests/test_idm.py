"""IDM data: transitions within episodes, labels, splits, resizing, action-free loading, DANN."""

import json

import numpy as np
import pytest
import torch
from test_data_policy import _fake_image_collection

from action_free.actions import to_absolute
from action_free.data import split_of
from action_free.idm import TransitionDataset, load_frames, load_transitions, split_files


def test_transitions_pair_consecutive_frames_of_one_episode(tmp_path):
    data = _fake_image_collection(tmp_path / "source", num_episodes=3, steps=5, size=16)
    files = sorted(data.glob("episode_*.npz"))
    t = load_transitions(files, image_size=16)
    assert len(t.index) == 15 and len(t.frames) == 18
    ds = TransitionDataset(t)
    for i in range(len(ds)):
        ep, step = int(t.episode[i]), i % 5
        obs = torch.from_numpy(np.load(files[ep])["observations"]).permute(0, 1, 4, 2, 3)
        torch.testing.assert_close(ds[i]["obs"], obs[step])
        torch.testing.assert_close(ds[i]["next_obs"], obs[step + 1])
    # The last frame of an episode is never an o_t.
    assert not set(t.index.tolist()) & {5, 11, 17}


def test_labels_are_the_absolute_targets(tmp_path):
    data = _fake_image_collection(tmp_path / "source", num_episodes=2, steps=4, size=16)
    files = sorted(data.glob("episode_*.npz"))
    t = load_transitions(files, image_size=16)
    expected = np.concatenate(
        [to_absolute(*(np.load(f)[k] for k in ("actions", "qpos"))) for f in files]
    )
    np.testing.assert_array_equal(t.targets.numpy(), expected)


def test_action_free_videos_load_without_actions_or_states(tmp_path):
    data = _fake_image_collection(tmp_path / "A", num_episodes=2, steps=4, size=16)
    for f in data.glob("episode_*.npz"):  # keep only the images
        obs = np.load(f)["observations"]
        np.savez_compressed(f, observations=obs)
    t = load_transitions(sorted(data.glob("episode_*.npz")), image_size=16, labeled=False)
    assert t.targets is None and len(t.index) == 8
    assert "target" not in TransitionDataset(t)[0]


def test_resize_keeps_layout_and_values(tmp_path):
    data = _fake_image_collection(tmp_path / "source", num_episodes=1, steps=2, size=32)
    f = next(data.glob("episode_*.npz"))
    assert load_frames(f, 16).shape == (3, 3, 3, 16, 16)
    assert load_frames(f, 32).dtype == torch.uint8
    obs = np.full((3, 3, 32, 32, 3), 77, np.uint8)
    np.savez_compressed(f, observations=obs)
    assert (load_frames(f, 16) == 77).all()  # a flat image stays flat


def test_split_files_follow_the_episode_split(tmp_path):
    data = _fake_image_collection(tmp_path / "source", num_episodes=10, steps=2, size=8)
    train, val = split_files(data, "train"), split_files(data, "val")
    assert len(train) == 8 and len(val) == 2 and not set(train) & set(val)
    assert all(split_of(int(f.stem.split("_")[1])) == "val" for f in val)
    assert split_files(data) == sorted(train + val)


def test_idm_shapes_and_target_scale():
    from action_free.idm import IDM

    torch.manual_seed(0)
    model = IDM(pretrained=False).eval()
    obs = torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8)
    next_obs = torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8)
    assert model.features(obs, next_obs).shape == (2, 3 * 512)
    assert model(obs, next_obs).shape == (2, 7)

    targets = torch.randn(100, 7) * 3 + 5
    targets[:, 4] = 1.0  # a joint that never moves
    model.set_target_stats(targets)
    assert model.target_std[4] == 1e-3
    expected = model(obs, next_obs) * model.target_std + model.target_mean
    torch.testing.assert_close(model.predict(obs, next_obs), expected)


def test_idm_first_layer_starts_from_the_pretrained_filters():
    from torchvision.models import resnet18

    from action_free.idm import IDM

    model = IDM(pretrained=False)
    w = model.encoder.conv1.weight
    assert w.shape == (64, 9, 7, 7)
    torch.testing.assert_close(w[:, 0:3], w[:, 3:6])
    assert (w[:, 6:9] == 0).all()
    # A still pair (o_t = o_t+1) gets the same first-layer response as one image in a ResNet.
    ref = resnet18().conv1
    with torch.no_grad():
        ref.weight.copy_(2 * w[:, 0:3])
        x = torch.randn(1, 3, 32, 32)
        torch.testing.assert_close(model.encoder.conv1(torch.cat([x, x, x - x], 1)), ref(x))


def test_training_writes_checkpoints_and_per_component_errors(tmp_path):
    from action_free.idm import ACTION_NAMES, load_idm, predict_transitions, train_idm

    data = _fake_image_collection(tmp_path / "source", num_episodes=10, steps=4, size=32)
    cfg = {
        "collections": [str(data)],
        "image_size": 32,
        "pretrained": False,
        "epochs": 2,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
    }
    best = train_idm(cfg, tmp_path / "run")
    assert {p.name for p in (tmp_path / "run").iterdir()} == {
        "best.pt",
        "last.pt",
        "history.json",
        "config.json",
    }
    assert set(best["val"]["mae"]) == set(ACTION_NAMES)

    model = load_idm(tmp_path / "run" / "best.pt", torch.device("cpu"))
    t = load_transitions(split_files(data, "val"), image_size=32)
    pred = predict_transitions(model, t, torch.device("cpu"))
    assert pred.shape == (len(t.index), 7)
    # The checkpoint keeps the target statistics, so predictions are absolute targets.
    assert abs(pred.mean() - t.targets.mean()) < 1.0


def test_selection_metric_ignores_the_joint_that_never_moves():
    from action_free.idm import MIN_STD, errors

    target = torch.zeros(4, 7)
    std = torch.tensor([1.0, 2.0, 1.0, 1.0, MIN_STD, 1.0, 0.5])
    pred = target.clone()
    pred[:, 1] = 1.0  # 0.5 standardized on one of six moving joints
    pred[:, 4] = 0.01  # 10 standardized on the fixed joint: must not count
    e = errors(pred, target, std)
    assert abs(e["smae"] - 0.5 / 6) < 1e-6
    assert e["mae"]["shoulder"] == 1.0


def test_gradient_reversal_is_identity_forward_and_negates_backward():
    from action_free.idm import GradReverse

    x = torch.randn(4, 5, requires_grad=True)
    y = GradReverse.apply(x, 0.5)
    torch.testing.assert_close(y, x)
    (3 * y).sum().backward()
    torch.testing.assert_close(x.grad, torch.full((4, 5), -0.5 * 3))


def test_dann_lambda_grows_from_zero_to_lambda_max():
    from action_free.idm import dann_lambda

    assert dann_lambda(0.0, 0.5, 10) == 0.0
    assert abs(dann_lambda(1.0, 0.5, 10) - 0.5) < 1e-3
    assert dann_lambda(0.2, 0.5, 10) < dann_lambda(0.5, 0.5, 10)


def test_discriminator_returns_one_logit_per_pair():
    from action_free.idm import DomainDiscriminator

    assert DomainDiscriminator(3 * 512, 16)(torch.randn(5, 3 * 512)).shape == (5,)


def test_dann_training_with_an_images_only_target(tmp_path):
    from action_free.idm import load_idm, train_idm

    source = _fake_image_collection(tmp_path / "source", num_episodes=10, steps=4, size=32)
    target = _fake_image_collection(tmp_path / "A", num_episodes=12, steps=4, size=32)  # new images
    for f in target.glob("episode_*.npz"):  # keep only the images (R1)
        obs = np.load(f)["observations"]
        np.savez_compressed(f, observations=obs)
    cfg = {
        "collections": [str(source)],
        "target": str(target),
        "image_size": 32,
        "pretrained": False,
        "epochs": 2,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
        "dann_lambda": 0.5,
        "dann_gamma": 10,
        "discriminator_hidden": 16,
        "discriminator_lr": 1e-3,
    }
    last = train_idm(cfg, tmp_path / "run")
    assert {p.name for p in (tmp_path / "run").iterdir()} == {
        "best.pt",
        "last.pt",
        "history.json",
        "config.json",
    }
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert last["epoch"] == 2 and len(history) == 2
    for h in history:
        assert 0 <= h["tsr"] <= 1 and 0 <= h["ttr"] <= 1
        assert {"lambda", "domain_loss"} <= set(h)
    assert "discriminator" in torch.load(tmp_path / "run" / "best.pt")
    load_idm(tmp_path / "run" / "best.pt", torch.device("cpu"))


def test_dann_stops_when_the_target_has_no_full_batch(tmp_path):
    from action_free.idm import train_idm

    source = _fake_image_collection(tmp_path / "source", num_episodes=10, steps=4, size=32)
    target = _fake_image_collection(tmp_path / "A", num_episodes=10, steps=1, size=32)  # 8 pairs
    cfg = {
        "collections": [str(source)],
        "target": str(target),
        "image_size": 32,
        "pretrained": False,
        "epochs": 1,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
        "dann_lambda": 0.5,
        "dann_gamma": 10,
        "discriminator_hidden": 16,
        "discriminator_lr": 1e-3,
    }
    with pytest.raises(ValueError, match="fewer train pairs"):
        train_idm(cfg, tmp_path / "run")
