"""LAPA (h): the VQ latent action model, latent labels, the policy stage and its loader."""

import json

import numpy as np
import torch
from test_data_policy import _fake_image_collection

from action_free.idm import load_transitions, split_files
from action_free.lapa import (
    LapaPolicy,
    LatentActionModel,
    export_lapa,
    load_lam,
    train_lam,
    train_lapa,
    z_labels,
)


def _tiny_lam(tmp_path):
    """A LAM trained for two epochs on a tiny fake collection; returns (checkpoint, data)."""
    data = _fake_image_collection(tmp_path / "source", num_episodes=10, steps=6, size=32)
    cfg = {
        "collections": [str(data)],
        "image_size": 32,
        "pretrained": False,
        "epochs": 2,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
        "codebook": 16,
        "commitment": 0.25,
        "latent_dim": 16,
    }
    best = train_lam(cfg, tmp_path / "lam")
    return tmp_path / "lam" / "best.pt", data, best


def test_lam_shapes_and_quantization():
    model = LatentActionModel(codebook=16, latent_dim=16, pretrained=False).eval()
    obs = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    next_obs = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    recon, z_e, z_q, indices = model(obs, next_obs)
    assert recon.shape == (2, 3, 3, 32, 32)
    assert indices.shape == (2,) and ((0 <= indices) & (indices < 16)).all()
    torch.testing.assert_close(z_q, model.vq.codebook(indices))  # straight-through
    z_e.retain_grad()
    recon.sum().backward()
    assert z_e.grad is not None  # gradients reach the pre-quantization features


def test_lam_encoder_starts_from_the_pretrained_filters():
    model = LatentActionModel(pretrained=False)
    w = model.encoder.conv1.weight
    assert w.shape == (64, 9, 7, 7)
    torch.testing.assert_close(w[:, 0:3], w[:, 3:6])
    assert (w[:, 6:9] == 0).all()


def test_train_lam_writes_outputs_and_z_labels(tmp_path):
    lam_ckpt, data, best = _tiny_lam(tmp_path)
    run = lam_ckpt.parent
    assert {p.name for p in run.iterdir()} == {"best.pt", "last.pt", "history.json", "config.json"}
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 2 and all("val_recon" in h for h in history)
    assert best["epoch"] == min(history, key=lambda h: h["val_recon"])["epoch"]

    model = load_lam(lam_ckpt, torch.device("cpu"))
    t = load_transitions(split_files(data, "val"), image_size=32, labeled=False)
    z = z_labels(model, t, torch.device("cpu"))
    assert z.shape == (len(t.index),) and ((0 <= z) & (z < 16)).all()


def test_policy_stage_trains_and_loads(tmp_path):
    from action_free.evaluate import load_policy

    lam_ckpt, _, _ = _tiny_lam(tmp_path)
    source = _fake_image_collection(tmp_path / "source-policy", num_episodes=10, steps=6, size=32)
    target = _fake_image_collection(tmp_path / "A", num_episodes=10, steps=6, size=32)
    turned = _fake_image_collection(tmp_path / "turned", num_episodes=10, steps=6, size=32)
    for f in target.glob("episode_*.npz"):  # action-free video: images, no actions or states
        ep = np.load(f)
        np.savez_compressed(
            f,
            observations=ep["observations"],
            seed=ep["seed"],
            success=ep["success"],
            length=ep["length"],
        )
    cfg = {
        "collections": [str(source)],
        "targets": [str(target)],
        "turned": [str(turned)],
        "lam": str(lam_ckpt),
        "image_size": 32,
        "decoder_image_size": 32,
        "pretrained": False,
        "num_epochs": 2,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "chunk_size": 5,
        "hidden_dim": 64,
        "dim_feedforward": 64,
        "enc_layers": 2,
        "dec_layers": 2,
        "nheads": 2,
        "decoder_hidden": 32,
        "action_weight": 1.0,
        "split_seed": 1,
        "num_workers": 0,
        "seed": 0,
    }
    out = tmp_path / "run"
    export_lapa(
        [
            (sorted(source.glob("episode_*.npz")), "source", True),
            (sorted(target.glob("episode_*.npz")), "target", False),
            (sorted(turned.glob("episode_*.npz")), "turned", True),
        ],
        out / "dataset",
    )
    record = train_lapa(cfg, out / "dataset", out)
    assert {p.name for p in out.iterdir()} >= {
        "policy_best.ckpt",
        "policy_last.ckpt",
        "lapa_run.json",
        "history.json",
        "config.json",
        "dataset",
    }
    history = json.loads((out / "history.json").read_text())
    assert len(history) == 2 and all("val_loss" in h for h in history)
    assert record["best_epoch"] in (1, 2)
    split = record["split"]
    # Source + target split 80/20 by episode; the turned demos sit unsplit in train.
    assert len(split["train"]) + len(split["val"]) == 30  # source + target + turned
    assert sum("turned" in f for f in split["val"]) == 0
    assert sum("turned" in f for f in split["train"]) == 10
    assert record["data"]["object"] == "source"
    assert "scene_name" in record["data"] and "train_initial_states" in record["data"]

    policy = LapaPolicy(out / "policy_best.ckpt", torch.device("cpu"))
    assert policy.action_mode == "absolute" and not policy.uses_robot_state
    images = torch.randint(0, 255, (1, 3, 32, 32, 3), dtype=torch.uint8)
    assert policy.predict(images).shape == (1, 5, 7)
    loaded = load_policy(str(out / "policy_best.ckpt"), torch.device("cpu"))
    assert isinstance(loaded, LapaPolicy)
