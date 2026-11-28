"""The LAPA labeler (method h): the original latent action quantization model plus an
action decoder (docs/experiments/lapa.md, notes/plan-lapa.md).

Stage 1 (`train_laq`) is the latent action quantization (LAQ) model of the original LAPA
paper (Ye et al., ICLR 2025, arXiv:2410.11758). The model itself is the official code,
vendored unedited at `third_party/lapa` (github.com/LatentActionPretraining/LAPA): a
C-ViViT variant whose encoder reads a frame pair (x_t, x_{t+H}) and produces a discrete
latent action z_t of `code_seq_len` tokens from a `codebook`-entry codebook (8 x 4 = 8^4
latent actions, NSVQ with codebook replacement), and whose decoder reconstructs x_{t+H}
from x_t and z_t by cross-attention (stop-grad on x_t). It trains on the images of the
source demos, the turned demos and the target videos; no action is ever read (R1, R3).
The frame window H is `stride` steps (0.6 s at 10 Hz, the paper's fps rule) for training;
the labels use adjacent frames (H = 1), the paper's own labeling procedure (§3.2: "use the
encoder ... as an inverse dynamics model to label all frames x_t, given frame x_{t+1}").

The action decoder g (`train_g`) is our extension. The original maps latent to robot
actions by fine-tuning the latent policy (stages 2-3); those stages produce a *policy*
(image + instruction -> a_t), never a per-transition label, and stage 2 is omitted here by
decision (notes/plan-lapa.md). The labeler therefore implements the conversion directly:
g(o_t, z_t) -> the absolute target a_t, trained on the labeled source and turned demos
(the role of `idm_turned` in method (b)). `label_collection` writes the pseudo-labels in
the IDM labeler's format (episode_XXXXX.npy (T, 7) + labels.json), which
`act_original.export_dataset` consumes unchanged.

Adaptations vs the official code, all disclosed in docs/experiments/lapa.md:
- 3 cameras stacked as 9 channels (`channels=9`; the original is single-view);
- 128 px with dim 512 and 4 + 4 layers instead of 256 px with dim 1024 and 8 + 8 (cost;
  the paper's own Fig. 5a ablates model scale; the reference size is recorded untrained);
- the NSVQ is constructed on the actual device: the official model hardcodes
  `device='cuda'` and cannot construct on CPU-only machines (`_patch_nsvq_device`).
Everything else is the official architecture and objective.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from action_free.act_original import ACTION_DIM
from action_free.idm import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    absolute_targets,
    errors,
    load_frames,
    select_device,
    split_files,
)

ACTION_STD_FLOOR = 1e-2  # g's target std floor, as ACT's normalization statistics

LAQ_ROOT = Path(
    os.environ.get("LAQ_ROOT", Path(__file__).resolve().parents[2] / "third_party" / "lapa" / "laq")
)

# Official code (vendored at third_party/lapa, github.com/LatentActionPretraining/LAPA,
# MIT license): laq_model.latent_action_quantization, .nsvq, .attention. Unedited.
_OFFICIAL_PARAMS = dict(
    dim=1024, quant_dim=32, codebook_size=8, image_size=256, patch_size=32,
    spatial_depth=8, temporal_depth=8, dim_head=64, heads=16, code_seq_len=4,
)  # fmt: skip  # the paper's configuration (train_sthv2.py), channels adapted to 9


def _import_upstream():
    """Import the vendored `laq_model` modules (the official LAQ, unedited).

    The package's own __init__ imports the official trainer, which pulls wandb and
    accelerate (not our dependencies); only the model modules are used, so the package is
    registered without executing its __init__ (the vendored files stay untouched).
    """
    import types
    import warnings

    from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

    for p in (LAQ_ROOT,):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if "laq_model" not in sys.modules:
        pkg = types.ModuleType("laq_model")
        pkg.__path__ = [str(LAQ_ROOT / "laq_model")]
        sys.modules["laq_model"] = pkg
    # The vendored attention.py uses PEP 484 hints that beartype deprecates on 3.12.
    warnings.filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)
    import laq_model.latent_action_quantization as laq_mod

    return laq_mod


def _patch_nsvq_device():
    """Let the official model construct on the actual device.

    `LatentActionQuantization.__init__` hardcodes NSVQ(device='cuda'), which cannot
    construct on CPU-only machines; it looks the class up from its own module globals, so
    a subclass with a pass-through device is installed there. The vendored code is
    otherwise untouched.
    """
    laq_mod = _import_upstream()
    if laq_mod.NSVQ.__name__ == "DeviceNSVQ":
        return laq_mod
    base = laq_mod.NSVQ

    class DeviceNSVQ(base):
        """Construct on the available device: the official model always passes
        device='cuda', which cannot construct on CPU-only machines; the substitute
        matches what the adapter later moves the model to (`select_device`)."""

        def __init__(self, *args, device="cuda", **kwargs):
            real = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            super().__init__(*args, device=real, **kwargs)

    DeviceNSVQ.__name__ = "DeviceNSVQ"
    DeviceNSVQ.__module__ = base.__module__
    laq_mod.NSVQ = DeviceNSVQ
    return laq_mod


LAQ_MODULE = _patch_nsvq_device()
LatentActionQuantization = LAQ_MODULE.LatentActionQuantization


def laq_kwargs(cfg: dict) -> dict:
    """The scaled-down LAQ configuration: 9 channels (the three cameras stacked), the rest
    from the config (configs/laq.toml)."""
    return {
        "dim": cfg["dim"],
        "quant_dim": cfg["quant_dim"],
        "codebook_size": cfg["codebook"],
        "image_size": cfg["image_size"],
        "patch_size": cfg["patch_size"],
        "spatial_depth": cfg["spatial_depth"],
        "temporal_depth": cfg["temporal_depth"],
        "dim_head": cfg["dim_head"],
        "heads": cfg["heads"],
        "code_seq_len": cfg["code_seq_len"],
        "channels": 9,  # three cameras, 3 colour channels each
    }


def make_laq(cfg: dict, device: torch.device) -> LatentActionQuantization:
    return LatentActionQuantization(**laq_kwargs(cfg)).to(device)


def count_parameters(model: nn.Module) -> dict:
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def reference_parameters(**overrides) -> int:
    """Parameter count of the official-scale LAQ (256 px, dim 1024, 8 + 8 layers, 16
    heads, 9 channels), constructed untrained and never trained: the reference against
    which the scaled-down model's recorded count is compared."""
    ref = LatentActionQuantization(**{**_OFFICIAL_PARAMS, "channels": 9, **overrides})
    return count_parameters(ref)["parameters"]


@dataclass
class Pairs:
    """Frame pairs (x_t, x_{t+stride}) of several episodes, each frame stored once (as
    `idm.Transitions`); `index[i] + stride` is the second frame of pair i."""

    frames: torch.Tensor  # (frames, cameras, 3, S, S) uint8, episodes concatenated
    index: torch.Tensor  # (N,) frame of x_t
    episode: torch.Tensor  # (N,) position of the pair's episode in `files`
    files: list[Path]
    stride: int
    targets: torch.Tensor | None  # (N, 7) absolute targets; None for action-free videos


def load_pairs(files: list[Path], image_size: int, stride: int, labeled: bool = True) -> Pairs:
    """(x_t, x_{t+stride}) pairs of every episode. Labeled pairs are meaningful only at
    stride 1 (the target of pair t is the action at step t)."""
    frames, index, episode, targets, start = [], [], [], [], 0
    for i, f in enumerate(files):
        x = load_frames(f, image_size)
        steps = len(x) - stride
        if steps < 1:
            raise ValueError(f"{f.name}: {len(x)} frames, no pair with stride {stride}")
        frames.append(x)
        index.append(torch.arange(start, start + steps))
        episode.append(torch.full((steps,), i))
        if labeled:
            targets.append(torch.from_numpy(absolute_targets(f))[:steps])
        start += len(x)
    return Pairs(
        torch.cat(frames),
        torch.cat(index),
        torch.cat(episode),
        list(files),
        stride,
        torch.cat(targets) if labeled else None,
    )


class PairDataset(Dataset):
    """(x_t, x_{t+stride}) as uint8 images, plus the target when labeled."""

    def __init__(self, pairs: Pairs):
        self.p = pairs

    def __len__(self) -> int:
        return len(self.p.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        k = self.p.index[i]
        item = {"obs": self.p.frames[k], "next_obs": self.p.frames[k + self.p.stride]}
        if self.p.targets is not None:
            item["target"] = self.p.targets[i]
        return item


def video_pair(obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """uint8 (B, cams, 3, S, S) twice -> (B, 9, 2, S, S) float32 in [0, 1]: the three
    cameras stacked on the channel dim, the two frames on the temporal dim (the LAQ's
    single-view video format; `channels=9` is the camera adaptation)."""
    x = torch.stack([obs, next_obs], 1).float() / 255  # (B, 2, cams, 3, S, S)
    return x.flatten(2, 3).transpose(1, 2)  # (B, 9, 2, S, S)


def video_frame(obs: torch.Tensor) -> torch.Tensor:
    """uint8 (B, cams, 3, S, S) -> (B, 9, 1, S, S) float32 in [0, 1]."""
    return (obs.float() / 255).flatten(1, 2).unsqueeze(2)


@torch.no_grad()
def z_codes(
    model: LatentActionQuantization,
    pairs: Pairs,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Latent actions z_t (N, 4), one 4-token code per transition, from the frozen LAQ
    encoder (the paper's labeling procedure)."""
    model.eval()
    loader = DataLoader(PairDataset(pairs), batch_size=batch_size)
    out = []
    for b in loader:
        v = video_pair(b["obs"].to(device), b["next_obs"].to(device))
        out.append(model.inference(v, return_only_codebook_ids=True).cpu())
    return torch.cat(out)


@torch.no_grad()
def reconstruction_mse(
    model: LatentActionQuantization,
    pairs: Pairs,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Mean squared reconstruction error of the validation pairs, in [0, 1] pixels."""
    model.eval()
    loader = DataLoader(PairDataset(pairs), batch_size=batch_size)
    total, n = 0.0, 0
    for b in loader:
        obs, next_obs = b["obs"].to(device), b["next_obs"].to(device)
        recon = model.inference(video_pair(obs, next_obs))  # (B, 9, S, S): official format
        y = video_frame(next_obs).squeeze(2)
        total += F.mse_loss(recon, y).item() * len(obs)
        n += len(obs)
    return total / n


def train_laq(cfg: dict, out_dir: Path, max_episodes: int | None = None) -> dict:
    """Pretrain the LAQ on the train split of cfg["collections"], read as images only
    (labeled=False); keep the epoch with the lowest validation reconstruction loss
    (best.pt). Follows the official recipe (laq/train_sthv2.py): Adam lr 1e-4, no weight
    decay, gradient clip 0.5; the NSVQ codebook replacement schedule runs inside the
    vendored forward on the global step. Writes history.json and config.json (with the
    trained parameter count and, unless cfg["record_reference"] is false, the official-
    scale reference count) in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = select_device()
    size = cfg["image_size"]

    def files(split: str) -> list[Path]:
        return [f for c in cfg["collections"] for f in split_files(Path(c), split)[:max_episodes]]

    train = load_pairs(files("train"), size, cfg["stride"], labeled=False)
    val = load_pairs(files("val"), size, cfg["stride"], labeled=False)
    model = make_laq(cfg, device)
    loader = DataLoader(
        PairDataset(train), batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), eps=1e-8)
    record_cfg = {**cfg, "device": str(device), **count_parameters(model)}
    if cfg.get("record_reference", True):
        record_cfg["reference_parameters"] = reference_parameters()
    (out_dir / "config.json").write_text(json.dumps(record_cfg, indent=2))

    history, best, step = [], float("inf"), 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        start = time.time()
        total = 0.0
        for b in loader:
            v = video_pair(b["obs"].to(device), b["next_obs"].to(device))
            loss, _ = model(v, step=step)  # NSVQ substitution; replacement by schedule
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            step += 1
            total += loss.item()
        v = reconstruction_mse(model, val, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(loader),
                "val_recon": v,
                "seconds": time.time() - start,
            }
        )
        print(f"epoch {epoch}: train {total / len(loader):.4f}, val recon {v:.4f}")
        checkpoint = {"model": model.state_dict(), "config": cfg, "epoch": epoch}
        if v < best:
            best = v
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(checkpoint, out_dir / "last.pt")
    return min(history, key=lambda h: h["val_recon"])


def load_laq(path: Path, device: torch.device) -> LatentActionQuantization:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = make_laq(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.config = cfg
    return model.eval()


def z_embedding(model: LatentActionQuantization, codes: torch.Tensor) -> torch.Tensor:
    """Token codes (N, 4) -> the NSVQ quantized embeddings (N, 4 * quant_dim): the same
    conditioning the official decoder cross-attends to (one shared 8-entry codebook across
    the 4 token positions)."""
    return model.vq.codebooks[codes].flatten(1)


class GDataset(Dataset):
    """(o_t, z_t) -> a_t items: each pair's first frame, its precomputed latent code and
    its absolute target."""

    def __init__(self, pairs: Pairs, z: torch.Tensor):
        self.p = pairs
        self.z = z

    def __len__(self) -> int:
        return len(self.p.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = {"obs": self.p.frames[self.p.index[i]], "z": self.z[i]}
        if self.p.targets is not None:
            item["target"] = self.p.targets[i]
        return item


class ActionDecoder(nn.Module):
    """g(o_t, z_t): the current frame and a transition's latent action -> the absolute
    target a_t. Our extension (notes/plan-lapa.md): the original's stage 3 fine-tunes a
    policy on (image, instruction), which cannot label an observed transition; g is
    trained on the labeled demos only (source + turned, the role of `idm_turned` in (b)).
    It has its own ResNet-18 (three colour channels per camera, 128 px); z enters as the
    NSVQ quantized embedding (4 tokens x quant_dim), frozen into the model so g alone
    decodes the LAQ's z space. Targets are standardized with training statistics stored in
    the model, so predict returns absolute targets."""

    def __init__(
        self,
        codebook: int,
        quant_dim: int,
        code_seq_len: int,
        cameras: int = 3,
        hidden: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.head = nn.Sequential(
            nn.Linear(cameras * 512 + code_seq_len * quant_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ACTION_DIM),
        )
        self.register_buffer("z_codebook", torch.zeros(codebook, quant_dim))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))
        self.register_buffer("target_mean", torch.zeros(ACTION_DIM))
        self.register_buffer("target_std", torch.ones(ACTION_DIM))

    def set_codebook(self, codebook: torch.Tensor) -> None:
        """Freeze the LAQ's NSVQ codebook into g (the z space g decodes)."""
        self.z_codebook.copy_(codebook)

    def set_target_stats(self, targets: torch.Tensor) -> None:
        """Standardize with the labeled training targets; the floor matches ACT's stats."""
        self.target_mean.copy_(targets.mean(0))
        self.target_std.copy_(targets.std(0).clamp_min(ACTION_STD_FLOOR))

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """uint8 (B, cams, 3, S, S) and codes (B, 4) -> standardized targets (B, 7)."""
        x = (obs.float() / 255 - self.mean) / self.std
        b, c = x.shape[:2]
        feat = self.encoder(x.flatten(0, 1)).view(b, c * 512)
        emb = self.z_codebook[z].flatten(1)  # (B, 4 * quant_dim)
        return self.head(torch.cat([feat, emb], dim=1))

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Absolute targets (B, 7)."""
        return self(obs, z) * self.target_std + self.target_mean


@torch.no_grad()
def predict_g(
    decoder: ActionDecoder,
    pairs: Pairs,
    z: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Absolute targets (N, 7) of every pair, in order."""
    decoder.eval()
    loader = DataLoader(GDataset(pairs, z), batch_size=batch_size)
    out = [decoder.predict(b["obs"].to(device), b["z"].to(device)).cpu() for b in loader]
    return torch.cat(out)


def train_g(cfg: dict, out_dir: Path, max_episodes: int | None = None) -> dict:
    """Train g on the train split of cfg["collections"] (the source and turned demos) with
    z labels from the frozen LAQ of cfg["laq"] on adjacent frames (the same pairs the
    labeler uses); keep the epoch with the lowest validation standardized MAE (best.pt),
    as the IDM. The IDM recipe (train_idm): AdamW lr 3e-4 cosine, weight decay 1e-4.
    Writes history.json and config.json (the LAQ checkpoint's config for provenance) in
    out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = select_device()
    laq = load_laq(Path(cfg["laq"]), device)
    laq_cfg = laq.config
    size = cfg["image_size"]

    def files(split: str) -> list[Path]:
        return [f for c in cfg["collections"] for f in split_files(Path(c), split)[:max_episodes]]

    train = load_pairs(files("train"), size, 1)
    val = load_pairs(files("val"), size, 1)
    z_train = z_codes(laq, train, device)
    z_val = z_codes(laq, val, device)
    decoder = ActionDecoder(
        laq_cfg["codebook"],
        laq_cfg["quant_dim"],
        laq_cfg["code_seq_len"],
        pretrained=cfg["pretrained"],
    ).to(device)
    decoder.set_codebook(laq.vq.codebooks.detach())
    decoder.set_target_stats(train.targets.to(device))
    std = decoder.target_std.cpu()

    loader = DataLoader(
        GDataset(train, z_train), batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    record_cfg = {**cfg, "device": str(device), "laq_config": laq_cfg}
    (out_dir / "config.json").write_text(json.dumps(record_cfg, indent=2))

    history, best, step = [], float("inf"), 0
    for epoch in range(1, cfg["epochs"] + 1):
        decoder.train()
        start = time.time()
        total = 0.0
        for b in loader:
            y = (b["target"].to(device) - decoder.target_mean) / decoder.target_std
            loss = F.mse_loss(decoder(b["obs"].to(device), b["z"].to(device)), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sched.step()
            step += 1
            total += loss.item()
        v = errors(predict_g(decoder, val, z_val, device), val.targets, std)
        history.append(
            {
                "epoch": epoch,
                "train_mse": total / len(loader),
                "val": v,
                "seconds": time.time() - start,
            }
        )
        print(
            f"epoch {epoch}: train {total / len(loader):.4f}, "
            f"val {v['mse']:.4f}, val sMAE {v['smae']:.4f}"
        )
        checkpoint = {
            "decoder": decoder.state_dict(),
            # The LAQ's dims, so the checkpoint reconstructs g without the LAQ.
            "config": {
                **cfg,
                "codebook": laq_cfg["codebook"],
                "quant_dim": laq_cfg["quant_dim"],
                "code_seq_len": laq_cfg["code_seq_len"],
                "laq_config": laq_cfg,
            },
            "laq": cfg["laq"],
            "epoch": epoch,
        }
        if v["smae"] < best:
            best = v["smae"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(checkpoint, out_dir / "last.pt")
    return min(history, key=lambda h: h["val"]["smae"])


def load_g(path: Path, device: torch.device) -> ActionDecoder:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    decoder = ActionDecoder(
        cfg["codebook"], cfg["quant_dim"], cfg["code_seq_len"], pretrained=False
    )
    decoder.load_state_dict(ckpt["decoder"])
    return decoder.to(device).eval()


def label_collection(
    laq: LatentActionQuantization,
    decoder: ActionDecoder,
    files: list[Path],
    image_size: int,
    out: Path,
    device: torch.device,
) -> None:
    """Pseudo-labels: one `episode_XXXXX.npy`, (T, 7) float32 absolute targets, per
    episode, the IDM labeler's format (act_original._pseudo_labels). The episodes are read
    as images only (R1): the LAQ labels every adjacent pair with its 4-token code and g
    decodes (o_t, z_t) to the target a_t."""
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        pairs = load_pairs([f], image_size, 1, labeled=False)
        z = z_codes(laq, pairs, device)
        a = predict_g(decoder, pairs, z, device)
        np.save(out / f"{f.stem}.npy", a.numpy())
