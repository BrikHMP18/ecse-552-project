"""The LAPA labeler (h): the LAQ (official code, scaled down), frame pairs, the action
decoder g, and the pseudo-label format ACT consumes (docs/experiments/lapa.md)."""

import json

import numpy as np
import torch
from test_data_policy import _fake_image_collection

from action_free.actions import to_absolute
from action_free.idm import split_files
from action_free.lapa import (
    load_g,
    load_laq,
    load_pairs,
    make_laq,
    reference_parameters,
    train_g,
    train_laq,
    video_pair,
    z_codes,
    z_embedding,
)

# The tiny LAQ configuration for CPU tests: the official architecture at image 32 /
# patch 4 (8x8 grid, as the official 256/32), dim 64, 2 + 2 layers, 2 heads.
TINY_LAQ = {
    "collections": [],
    "image_size": 32,
    "patch_size": 4,
    "dim": 64,
    "spatial_depth": 2,
    "temporal_depth": 2,
    "heads": 2,
    "dim_head": 64,
    "codebook": 8,
    "code_seq_len": 4,
    "quant_dim": 32,
    "stride": 2,
    "epochs": 2,
    "batch_size": 8,
    "lr": 1e-3,
    "grad_clip": 0.5,
    "record_reference": False,
    "seed": 0,
}


def _tiny_laq(tmp_path, stride=2, num_episodes=10):
    """A LAQ trained for two epochs on a tiny fake collection; returns (checkpoint, data)."""
    data = _fake_image_collection(tmp_path / "source", num_episodes=num_episodes, steps=6)
    cfg = {**TINY_LAQ, "collections": [str(data)], "stride": stride}
    best = train_laq(cfg, tmp_path / "laq")
    return tmp_path / "laq" / "best.pt", data, best


def test_laq_shapes_codes_and_reconstruction():
    model = make_laq(TINY_LAQ, torch.device("cpu")).eval()
    obs = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    next_obs = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    v = video_pair(obs, next_obs)
    assert v.shape == (2, 9, 2, 32, 32)  # 3 cameras x 3 colours, two frames
    loss, unique = model(v)
    assert loss.shape == () and isinstance(unique, int)
    codes = model.inference(v, return_only_codebook_ids=True)
    assert codes.shape == (2, 4) and ((0 <= codes) & (codes < 8)).all()  # 8^4 latent actions
    recon = model.inference(v)
    assert recon.shape == (2, 9, 32, 32)  # the reconstructed x_{t+H} (temporal dim squeezed)


def test_laq_constructs_on_the_actual_device():
    # The official model hardcodes NSVQ(device='cuda'); the adapter patches it, so CPU
    # construction works and the NSVQ knows its real device.
    model = make_laq(TINY_LAQ, torch.device("cpu"))
    assert model.vq.device == torch.device("cpu")


def test_nsvq_codebook_replacement_runs():
    model = make_laq(TINY_LAQ, torch.device("cpu"))
    obs = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    model(video_pair(obs, obs))  # accumulate codebook usage counters
    # The official counter is advisory (plain indexed +=, not a scatter: duplicate codes
    # in a batch are counted once). Only that it counts and resets is asserted.
    assert 1 <= model.vq.codebooks_used.sum().item() <= 2 * 4
    model.vq.replace_unused_codebooks(2)  # the official early-training maintenance step
    assert model.vq.codebooks_used.sum().item() == 0


def test_pairs_use_the_stride_and_share_frames(tmp_path):
    from action_free.idm import load_frames

    data = _fake_image_collection(tmp_path / "data", num_episodes=8, steps=6)
    files = sorted(data.glob("episode_*.npz"))
    pairs = load_pairs(files, image_size=32, stride=3)
    assert len(pairs.index) == 8 * (7 - 3)  # 7 frames, 4 pairs per episode
    # Episode 1 starts at frame 7; its pair i is (frame 7 + i, frame 7 + i + 3), each
    # frame stored once in the shared list.
    ep1 = load_frames(files[1], 32)
    for i in range(4):
        k = int(pairs.index[4 + i])
        torch.testing.assert_close(pairs.frames[k], ep1[i])
        torch.testing.assert_close(pairs.frames[k + 3], ep1[i + 3])
        assert int(pairs.episode[4 + i]) == 1
    # Labeled pairs at stride 1 carry the action of the step they start at.
    one = load_pairs(files, image_size=32, stride=1)
    ep0 = np.load(files[0])
    np.testing.assert_allclose(one.targets[4], to_absolute(ep0["actions"], ep0["qpos"])[4])


def test_train_laq_smoke_writes_outputs_and_records_parameters(tmp_path):
    ckpt, data, best = _tiny_laq(tmp_path)
    run = ckpt.parent
    assert {p.name for p in run.iterdir()} == {"best.pt", "last.pt", "history.json", "config.json"}
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 2 and all("val_recon" in h for h in history)
    assert best["epoch"] == min(history, key=lambda h: h["val_recon"])["epoch"]
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["parameters"] > 0 and cfg["trainable"] == cfg["parameters"]
    assert "reference_parameters" not in cfg  # record_reference=False in the smoke config

    model = load_laq(ckpt, torch.device("cpu"))
    pairs = load_pairs(split_files(data, "val"), image_size=32, stride=2, labeled=False)
    z = z_codes(model, pairs, torch.device("cpu"))
    assert z.shape == (len(pairs.index), 4) and ((0 <= z) & (z < 8)).all()
    emb = z_embedding(model, z)
    assert emb.shape == (len(pairs.index), 4 * 32)
    torch.testing.assert_close(emb, model.vq.codebooks[z].flatten(1))


def test_reference_parameters_counts_a_configured_model():
    # The official-scale count (300M-class) is too heavy for CPU tests; the same function
    # at tiny size proves the counting path.
    n = reference_parameters(dim=64, spatial_depth=2, temporal_depth=2, heads=2, image_size=32)
    assert n > 0


def test_g_trains_loads_and_predicts(tmp_path):
    ckpt, _, _ = _tiny_laq(tmp_path)
    source = _fake_image_collection(tmp_path / "source-g", num_episodes=8, steps=6)
    turned = _fake_image_collection(tmp_path / "turned", num_episodes=8, steps=6)
    cfg = {
        "collections": [str(source), str(turned)],
        "laq": str(ckpt),
        "image_size": 32,
        "pretrained": False,
        "epochs": 2,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
    }
    out = tmp_path / "g"
    best = train_g(cfg, out)
    assert {p.name for p in out.iterdir()} == {"best.pt", "last.pt", "history.json", "config.json"}
    assert best["epoch"] in (1, 2) and "smae" in best["val"]
    decoder = load_g(out / "best.pt", torch.device("cpu"))
    laq = load_laq(ckpt, torch.device("cpu"))
    torch.testing.assert_close(decoder.z_codebook, laq.vq.codebooks.detach())
    obs = torch.randint(0, 256, (3, 3, 3, 32, 32), dtype=torch.uint8)
    z = torch.tensor([[1, 2, 3, 4], [0, 7, 6, 5], [3, 3, 3, 3]])
    assert decoder.predict(obs, z).shape == (3, 7)


def test_label_collection_writes_the_act_pseudo_label_format(tmp_path):
    import h5py

    from action_free.act_original import ACTION_DIM, export_dataset

    ckpt, _, _ = _tiny_laq(tmp_path)
    target = _fake_image_collection(tmp_path / "A", num_episodes=4, steps=6)
    files = sorted(target.glob("episode_*.npz"))
    for f in files:  # action-free videos: images only, no actions or states
        ep = np.load(f)
        np.savez_compressed(
            f,
            observations=ep["observations"],
            seed=ep["seed"],
            success=ep["success"],
            length=ep["length"],
        )
    laq = load_laq(ckpt, torch.device("cpu"))
    decoder = load_g(_tiny_g_for_label_test(tmp_path, ckpt), torch.device("cpu"))
    labels = tmp_path / "labels"
    from action_free.lapa import label_collection

    label_collection(laq, decoder, files, 32, labels, torch.device("cpu"))
    (labels / "labels.json").write_text('{"laq": "fake"}')
    for f in files:
        a = np.load(labels / f"{f.stem}.npy")
        assert a.shape == (6, ACTION_DIM) and a.dtype == np.float32
    # The ACT exporter consumes the folder unchanged (labels used, stored actions never).
    export_dataset([(files, labels)], tmp_path / "act")
    with h5py.File(tmp_path / "act" / "episode_1.hdf5") as h:
        np.testing.assert_array_equal(
            h["action"][:, :ACTION_DIM], np.load(labels / "episode_00001.npy")
        )


def _tiny_g_for_label_test(tmp_path, laq_ckpt):
    source = _fake_image_collection(tmp_path / "source-label", num_episodes=8, steps=6)
    cfg = {
        "collections": [str(source)],
        "laq": str(laq_ckpt),
        "image_size": 32,
        "pretrained": False,
        "epochs": 2,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
    }
    train_g(cfg, tmp_path / "g-label")
    return tmp_path / "g-label" / "best.pt"


def test_action_decoder_constructor_matches_the_laq_dims(tmp_path):
    # g's checkpoint records the LAQ's codebook dims, so load_g rebuilds it standalone.
    ckpt, _, _ = _tiny_laq(tmp_path)
    source = _fake_image_collection(tmp_path / "source-dims", num_episodes=8, steps=6)
    cfg = {
        "collections": [str(source)],
        "laq": str(ckpt),
        "image_size": 32,
        "pretrained": False,
        "epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
    }
    train_g(cfg, tmp_path / "g-dims")
    ckpt_data = torch.load(tmp_path / "g-dims" / "best.pt", map_location="cpu")
    assert ckpt_data["config"]["codebook"] == 8
    assert ckpt_data["config"]["quant_dim"] == 32
    assert ckpt_data["config"]["code_seq_len"] == 4
    decoder = load_g(tmp_path / "g-dims" / "best.pt", torch.device("cpu"))
    assert decoder.z_codebook.shape == (8, 32)
