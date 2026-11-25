"""Latent action pretraining (LAPA, method h) and its data (docs/experiments/lapa.md).

Stage 1 (`train_lam`) pretrains a latent action model (LAM), a VQ-VAE whose encoder reads
two consecutive frames and predicts a discrete latent action z and whose decoder
reconstructs the next frame from o_t and z. It trains on the images of the source demos and
the target videos; no action is ever read in this stage (R1, R3).

Stage 2 (`train_lapa`) trains the policy and its action decoder. The policy is ACT's
transformer without the VAE (the outputs are discrete): the current frame -> a chunk of
`chunk_size` latent actions. It trains on the source demos and the target videos, whose
supervision is the latent action the frozen LAM encoder gives each frame pair (latent
pseudo-labels; cross-entropy). The action decoder g(o_t, z) has its own ResNet-18 and an MLP
head over the absolute targets ACT uses; it trains on the source demos and the 100
turned-grasp demos only, so the turned demos never enter the policy's z-head (they play the
role of `idm_turned` in method (b)). The target videos are read through their images only
(R1, R3).

The episodes of stage 2 come from `export_lapa`'s HDF5 layout (`episode_<i>.hdf5` with the
camera images and, for labeled episodes, the absolute targets; `episodes.json`): images are
read once per item, as upstream ACT does.

`LapaPolicy` presents a trained policy to `evaluate.rollout` with the interface of
`act_original.ActPolicy`: `predict(images)` returns absolute targets (B, chunk, 7). A chunk
is decoded step by step against the current frame, g(o_t, z_i) for the i-th chunk step; the
executed action is dominated by the freshest query's first step, as ACT's temporal
ensembling weights it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from action_free.act_original import ACTION_DIM, CAMERAS
from action_free.actions import to_absolute
from action_free.idm import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TransitionDataset,
    Transitions,
    load_transitions,
    select_device,
    split_files,
)
from action_free.initial_states import read_csv

ACTION_STD_FLOOR = 1e-2  # as ACT's normalization statistics


def resize_uint8(x: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 (..., 3, H, W) -> (..., 3, size, size), resized with antialiasing."""
    if x.shape[-1] == size:
        return x
    flat = x.flatten(0, -4)
    out = F.interpolate(flat.float(), size=(size, size), mode="bilinear", antialias=True)
    return out.round().clamp(0, 255).to(torch.uint8).unflatten(0, x.shape[:-3])


class VQ(nn.Module):
    """Vector quantization: the nearest codebook entry, straight-through for the input."""

    def __init__(self, size: int, dim: int):
        super().__init__()
        self.codebook = nn.Embedding(size, dim)
        self.codebook.weight.data.uniform_(-1 / size, 1 / size)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, dim) -> z_q (N, dim) and the code indices (N,)."""
        d = (
            z_e.pow(2).sum(1, keepdim=True)
            + self.codebook.weight.pow(2).sum(1)
            - 2 * z_e @ self.codebook.weight.t()
        )
        indices = d.argmin(1)
        z_q = self.codebook(indices)
        return z_e + (z_q - z_e).detach(), indices


class _Decoder(nn.Module):
    """Shared across cameras: o_t and z_q -> reconstructed o_{t+1} (per camera)."""

    def __init__(self, dim: int, channels: tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        stem, c = [], 3
        for ch in channels:  # five halvings: S -> S/32
            stem += [nn.Conv2d(c, ch, 4, 2, 1), nn.ReLU()]
            c = ch
        self.stem = nn.Sequential(*stem)
        self.z_proj = nn.Conv2d(dim, channels[-1], 1)
        up, c = [], channels[-1]
        for ch in (*channels[-2::-1], 16):  # five doublings: S/32 -> S
            up += [nn.ConvTranspose2d(c, ch, 4, 2, 1), nn.ReLU()]
            c = ch
        up.append(nn.Conv2d(c, 3, 3, 1, 1))
        self.up = nn.Sequential(*up)

    def forward(self, obs: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        b, cam = obs.shape[:2]
        x = obs.flatten(0, 1)  # (B*cam, 3, S, S)
        z = z_q.repeat_interleave(cam, 0)[:, :, None, None]  # broadcast over the bottleneck
        return self.up(self.stem(x) + self.z_proj(z)).unflatten(0, (b, cam))


class LatentActionModel(nn.Module):
    """VQ-VAE over frame pairs: (o_t, o_{t+1}) -> z; (o_t, z) -> reconstructed o_{t+1}.

    The encoder is the IDM's ResNet-18 (9 channels: the two frames and their difference),
    shared across cameras; the decoder reconstructs each camera's next frame.
    """

    def __init__(
        self,
        codebook: int = 128,
        latent_dim: int = 64,
        cameras: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old = backbone.conv1
        conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # As the IDM: a still pair gives the pretrained response; the difference
            # channels start at zero and learn motion filters from there.
            conv1.weight.zero_()
            conv1.weight[:, 0:3] = old.weight / 2
            conv1.weight[:, 3:6] = old.weight / 2
        backbone.conv1 = conv1
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.pre = nn.Linear(cameras * 512, latent_dim)
        self.vq = VQ(codebook, latent_dim)
        self.decoder = _Decoder(latent_dim)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))

    def encode(self, obs: torch.Tensor, next_obs: torch.Tensor) -> tuple:
        """uint8 (B, cam, 3, S, S) twice -> z_e, z_q (B, dim) and the code indices (B,)."""
        o = (obs.float() / 255 - self.mean) / self.std
        o_next = (next_obs.float() / 255 - self.mean) / self.std
        x = torch.cat([o, o_next, o_next - o], dim=2)
        b, c = x.shape[:2]
        z_e = self.pre(self.encoder(x.flatten(0, 1)).view(b, c * 512))
        z_q, indices = self.vq(z_e)
        return z_e, z_q, indices

    def decode(self, obs: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        """uint8 (B, cam, 3, S, S) and (B, dim) -> reconstructed o_{t+1} (B, cam, 3, S, S)."""
        o = (obs.float() / 255 - self.mean) / self.std
        return self.decoder(o, z_q)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor):
        z_e, z_q, indices = self.encode(obs, next_obs)
        return self.decode(obs, z_q), z_e, z_q, indices

    @torch.no_grad()
    def indices(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Latent actions z (B,) of the transitions."""
        return self.encode(obs, next_obs)[2]


def reconstruction_mse(
    model: LatentActionModel, transitions: Transitions, device: torch.device, batch_size: int = 256
) -> float:
    """Mean squared reconstruction error of the validation transitions, in [0, 1] pixels."""
    model.eval()
    loader = DataLoader(TransitionDataset(transitions), batch_size=batch_size)
    total, n = 0.0, 0
    for b in loader:
        obs, next_obs = b["obs"].to(device), b["next_obs"].to(device)
        recon = model.decode(obs, model.encode(obs, next_obs)[1])
        y = (next_obs.float() / 255 - model.mean) / model.std
        total += F.mse_loss(recon, y).item() * len(obs)
        n += len(obs)
    return total / n


def train_lam(cfg: dict, out_dir: Path, max_episodes: int | None = None) -> dict:
    """Pretrain the LAM on the train split of cfg["collections"], read as images only;
    keep the epoch with the lowest validation reconstruction loss (best.pt). Writes
    history.json and config.json in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = select_device()
    size = cfg["image_size"]

    def files(split: str) -> list[Path]:
        return [f for c in cfg["collections"] for f in split_files(Path(c), split)[:max_episodes]]

    train = load_transitions(files("train"), size, labeled=False)
    val = load_transitions(files("val"), size, labeled=False)
    model = LatentActionModel(cfg["codebook"], cfg["latent_dim"], pretrained=cfg["pretrained"]).to(
        device
    )
    loader = DataLoader(
        TransitionDataset(train), batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    (out_dir / "config.json").write_text(json.dumps({**cfg, "device": str(device)}, indent=2))

    history, best, step = [], float("inf"), 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        start = time.time()
        total = recon_total = commit_total = 0.0
        counts = torch.zeros(model.vq.codebook.weight.shape[0])
        for b in loader:
            obs, next_obs = b["obs"].to(device), b["next_obs"].to(device)
            recon, z_e, z_q, indices = model(obs, next_obs)
            y = (next_obs.float() / 255 - model.mean) / model.std
            recon_loss = F.mse_loss(recon, y)
            commit_loss = F.mse_loss(z_e.detach(), z_q) + cfg["commitment"] * F.mse_loss(
                z_e, z_q.detach()
            )
            loss = recon_loss + commit_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sched.step()
            step += 1
            total += loss.item()
            recon_total += recon_loss.item()
            commit_total += commit_loss.item()
            counts += torch.bincount(indices.cpu(), minlength=len(counts))
        p = counts / counts.sum()
        perp = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))
        v = reconstruction_mse(model, val, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(loader),
                "train_recon": recon_total / len(loader),
                "train_commit": commit_total / len(loader),
                "val_recon": v,
                "perplexity": perp,
                "seconds": time.time() - start,
            }
        )
        print(
            f"epoch {epoch}: train {total / len(loader):.4f} "
            f"(recon {recon_total / len(loader):.4f}), val recon {v:.4f}"
        )
        checkpoint = {"model": model.state_dict(), "config": cfg, "epoch": epoch}
        if v < best:
            best = v
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(checkpoint, out_dir / "last.pt")
    return min(history, key=lambda h: h["val_recon"])


def load_lam(path: Path, device: torch.device) -> LatentActionModel:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = LatentActionModel(cfg["codebook"], cfg["latent_dim"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.config = cfg
    return model.to(device).eval()


@torch.no_grad()
def z_labels(
    model: LatentActionModel,
    transitions: Transitions,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Latent labels z (N,) of every transition, in order (the frozen LAM's codes)."""
    model.eval()
    loader = DataLoader(TransitionDataset(transitions), batch_size=batch_size)
    return torch.cat(
        [model.indices(b["obs"].to(device), b["next_obs"].to(device)).cpu() for b in loader]
    )


def export_lapa(collections: list[tuple[list[Path], str, bool]], out_dir: Path) -> list[dict]:
    """Episodes as HDF5 for the policy stage: one `episode_<i>.hdf5` per episode in file
    order, with the camera images (lzf, as `act_original.export_dataset`) and, for labeled
    episodes, the absolute targets. `episodes.json` lists id, file, dir, seed, length, role
    (source, target or turned) and `labeled`. Target videos keep no action dataset, so
    their stored actions and states are never read (R1, R3)."""
    import h5py

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index, i = [], 0
    for files, role, labeled in collections:
        for f in map(Path, files):
            manifest = f.parent / "manifest.json"
            if (
                manifest.exists()
                and json.loads(manifest.read_text())["config"].get("role") == "test"
            ):
                raise ValueError(f"{f} belongs to a held-out test collection; never train on it.")
            with np.load(f) as ep:
                steps = int(ep["length"])
                obs = ep["observations"][:steps]
                seed = int(ep["seed"])
                action = None if not labeled else to_absolute(ep["actions"], ep["qpos"])
            with h5py.File(out_dir / f"episode_{i}.hdf5", "w") as h:
                images = h.create_group("observations/images")
                for k, cam in enumerate(CAMERAS):
                    images.create_dataset(
                        cam, data=obs[:, k], chunks=(1, *obs.shape[2:]), compression="lzf"
                    )
                if action is not None:
                    h.create_dataset("action", data=action)
            index.append(
                {
                    "id": i,
                    "file": f.name,
                    "dir": str(f.parent),
                    "seed": seed,
                    "length": steps,
                    "role": role,
                    "labeled": labeled,
                }
            )
            i += 1
    (out_dir / "episodes.json").write_text(json.dumps(index, indent=2))
    return index


class LatentPolicy(nn.Module):
    """The current frame -> a chunk of latent actions: ACT's transformer with the action
    head replaced by K-way logits. Unlike ACT there is no VAE: the outputs are discrete."""

    def __init__(
        self,
        codebook: int,
        cameras: int = 3,
        chunk: int = 20,
        hidden: int = 512,
        ff: int = 3200,
        enc_layers: int = 4,
        dec_layers: int = 7,
        nheads: int = 8,
        pretrained: bool = True,
    ):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(512, hidden, 1)
        self.pos = nn.Parameter(torch.zeros(1, cameras, hidden))
        layer = nn.TransformerEncoderLayer(hidden, nheads, ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers)
        dlayer = nn.TransformerDecoderLayer(hidden, nheads, ff, batch_first=True)
        self.decoder = nn.TransformerDecoder(dlayer, dec_layers)
        self.query_embed = nn.Embedding(chunk, hidden)
        self.head = nn.Linear(hidden, codebook)
        self.register_buffer("tgt_mask", torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), 1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """uint8 (B, cam, 3, S, S) -> logits (B, chunk, K)."""
        x = (obs.float() / 255 - self.mean) / self.std
        b, c = x.shape[:2]
        feat = self.backbone(x.flatten(0, 1)).view(b, c, 512)
        proj = self.input_proj(feat.flatten(0, 1)[:, :, None, None])[..., 0, 0]
        src = proj.view(b, c, -1) + self.pos  # (B, cam, hidden)
        tgt = self.query_embed.weight[None].expand(b, -1, -1)  # (B, chunk, hidden)
        out = self.decoder(tgt, src, tgt_mask=self.tgt_mask)
        return self.head(out)


class ActionDecoder(nn.Module):
    """g(o_t, z): the current frame and a latent action -> the absolute target a_t. It has
    its own ResNet-18 (three colour channels), so the turned demos train it without
    touching the policy's encoder. Targets are standardized with training statistics stored
    in the model, so `predict` returns absolute targets."""

    def __init__(self, codebook: int, cameras: int = 3, hidden: int = 512, pretrained: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.codebook = codebook
        self.head = nn.Sequential(
            nn.Linear(cameras * 512 + codebook, hidden), nn.ReLU(), nn.Linear(hidden, ACTION_DIM)
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))
        self.register_buffer("target_mean", torch.zeros(ACTION_DIM))
        self.register_buffer("target_std", torch.ones(ACTION_DIM))

    def set_target_stats(self, targets: torch.Tensor) -> None:
        """Standardize with the labeled training targets; the floor matches ACT's stats."""
        self.target_mean.copy_(targets.mean(0))
        self.target_std.copy_(targets.std(0).clamp_min(ACTION_STD_FLOOR))

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """uint8 (B, cam, 3, S, S) and codes (B,) -> standardized targets (B, 7)."""
        x = (obs.float() / 255 - self.mean) / self.std
        b, c = x.shape[:2]
        feat = self.encoder(x.flatten(0, 1)).view(b, c * 512)
        one_hot = F.one_hot(z, self.codebook).float()
        return self.head(torch.cat([feat, one_hot], dim=1))

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Absolute targets (B, 7)."""
        return self(obs, z) * self.target_std + self.target_mean


def describe_run(collections: list[tuple[Path, str]], split: dict) -> dict:
    """Run description for out-of-distribution checks (evaluate.compare_to_training): the
    source collection's scene and initial states, plus one entry per collection.
    `collections` is (dir, role); `split` maps "train"/"val" to file names."""
    info = []
    for d, role in collections:
        manifest = Path(d) / "manifest.json"
        m = json.loads(manifest.read_text()) if manifest.exists() else {}
        cfg = m.get("config", {})
        info.append(
            {
                "dir": str(d),
                "object": cfg.get("object_name", "source"),
                "scene_name": cfg.get("scene", "random_yaw"),
                "expert": cfg.get("expert", "stochastic"),
                "scene": (m.get("object") or {}).get("env_kwargs"),
                "role": role,
            }
        )
    source = info[0]
    rows = read_csv(Path(source["dir"])) or []
    train = set(split["train"])
    return {
        **{k: source[k] for k in ("object", "scene_name", "expert", "scene")},
        "source_dir": source["dir"],
        "collections": info,
        "train_files": list(split["train"]),
        "val_files": list(split["val"]),
        "train_initial_states": [
            [r["cube_x"], r["cube_y"], r["cube_yaw_mod90_deg"]] for r in rows if r["file"] in train
        ],
    }


@dataclass
class _Roles:
    """Whether an episode contributes to the policy's cross-entropy and the decoder's MSE."""

    policy: bool
    decoder: bool


ROLES = {"source": _Roles(True, True), "target": _Roles(True, False), "turned": _Roles(False, True)}


class LapaDataset(Dataset):
    """One item: the frame at a chunk start and the chunk's latent labels (padded); the
    absolute targets of the same steps when the episode is labeled. The start is sampled at
    random, or fixed at 0 for validation."""

    def __init__(
        self, dataset_dir: Path, ids: list[int], z: dict, chunk: int, deterministic: bool = False
    ):
        self.dir = Path(dataset_dir)
        self.index = json.loads((self.dir / "episodes.json").read_text())
        self.ids = ids
        self.z = z
        self.chunk = chunk
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        import h5py

        e = self.index[self.ids[i]]
        with h5py.File(self.dir / f"episode_{e['id']}.hdf5", "r") as h:
            obs = np.stack([h["observations/images"][cam][()] for cam in CAMERAS], 1)
            actions = h["action"][()] if "action" in h else None
        t = 0 if self.deterministic else np.random.randint(len(self.z[e["id"]]))
        z = self.z[e["id"]][t : t + self.chunk]
        mask = torch.zeros(self.chunk, dtype=torch.bool)
        mask[: len(z)] = True
        item = {
            "obs": torch.from_numpy(obs[t].copy()),  # (cam, H, W, 3) uint8
            "z": F.pad(z, (0, self.chunk - len(z))),
            "z_mask": mask,
            "actions": None,
            "ce": ROLES[e["role"]].policy,
            "mse": ROLES[e["role"]].decoder,
        }
        if actions is not None:
            a = actions[t : t + self.chunk]
            a = np.pad(a, ((0, self.chunk - len(a)), (0, 0)))
            item["actions"] = torch.from_numpy(a)
        return item


def collate(items: list[dict]) -> dict:
    out = {k: torch.stack([it[k] for it in items]) for k in ("obs", "z", "z_mask")}
    out["ce"] = torch.tensor([it["ce"] for it in items], dtype=torch.bool)
    out["mse"] = torch.tensor([it["mse"] for it in items], dtype=torch.bool)
    actions = torch.zeros(len(items), items[0]["z"].shape[0], ACTION_DIM)
    lab = torch.zeros_like(out["z_mask"])
    for i, it in enumerate(items):
        if it["actions"] is not None:
            actions[i] = it["actions"]
            lab[i] = True
    return {**out, "actions": actions, "lab": lab}


@torch.no_grad()
def compute_z(
    lam: LatentActionModel, dataset_dir: Path, index: list[dict], device: torch.device
) -> dict:
    """Latent labels z (T,) per episode id, from the frozen LAM on each pair of frames."""
    import h5py

    size = lam.config["image_size"]
    z = {}
    for e in index:
        with h5py.File(Path(dataset_dir) / f"episode_{e['id']}.hdf5", "r") as h:
            obs = np.stack([h["observations/images"][cam][()] for cam in CAMERAS], 1)
        frames = resize_uint8(torch.from_numpy(obs).permute(0, 1, 4, 2, 3), size)
        steps = len(frames) - 1
        out = []
        for i0 in range(0, steps, 128):
            i1 = min(i0 + 128, steps)
            out.append(
                lam.indices(frames[i0:i1].to(device), frames[i0 + 1 : i1 + 1].to(device)).cpu()
            )
        z[e["id"]] = torch.cat(out)
    return z


def train_lapa(cfg: dict, dataset_dir: Path, out_dir: Path) -> dict:
    """Train the policy and the action decoder of method (h) on the episodes exported by
    `export_lapa`. cfg["lam"] is a LAM checkpoint: its frozen encoder gives the latent
    pseudo-labels. The policy trains on the source demos and the target videos (latent
    labels), the decoder on the source demos and the turned demos; the split is a random
    80/20 by episode over source plus target (cfg["split_seed"]), the turned demos unsplit
    in train. Writes policy_best.ckpt (lowest validation loss), policy_last.ckpt,
    lapa_run.json, history.json and config.json in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = select_device()
    lam = load_lam(Path(cfg["lam"]), device)
    cfg = {**cfg, "codebook": lam.config["codebook"]}  # the codebook is the LAM's
    index = json.loads((Path(dataset_dir) / "episodes.json").read_text())
    z = compute_z(lam, dataset_dir, index, device)

    policy_ids = [e["id"] for e in index if ROLES[e["role"]].policy]
    turned_ids = [e["id"] for e in index if e["role"] == "turned"]
    rng = np.random.default_rng(cfg["split_seed"])
    order = rng.permutation(policy_ids)
    n_val = round(0.2 * len(policy_ids))
    val_ids = sorted(order[:n_val].tolist())
    train_ids = [i for i in policy_ids if i not in set(val_ids)] + turned_ids
    if len(train_ids) < cfg["batch_size"]:
        raise ValueError(
            f"{dataset_dir} has fewer train episodes than batch_size {cfg['batch_size']}"
        )

    policy = LatentPolicy(
        cfg["codebook"],
        chunk=cfg["chunk_size"],
        hidden=cfg["hidden_dim"],
        ff=cfg["dim_feedforward"],
        enc_layers=cfg["enc_layers"],
        dec_layers=cfg["dec_layers"],
        nheads=cfg["nheads"],
        pretrained=cfg["pretrained"],
    ).to(device)
    decoder = ActionDecoder(
        cfg["codebook"], hidden=cfg["decoder_hidden"], pretrained=cfg["pretrained"]
    ).to(device)
    # The decoder's target statistics come from the labeled train episodes only.
    import h5py

    labeled = []
    for i in train_ids:
        if not index[i]["labeled"]:
            continue
        with h5py.File(dataset_dir / f"episode_{i}.hdf5", "r") as h:
            labeled.append(torch.from_numpy(h["action"][()]))
    decoder.set_target_stats(torch.cat(labeled))

    ds = LapaDataset(dataset_dir, train_ids, z, cfg["chunk_size"])
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=cfg.get("num_workers", 2),
    )
    val_ds = LapaDataset(dataset_dir, val_ids, z, cfg["chunk_size"], deterministic=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=collate)
    params = list(policy.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = cfg["num_epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    (out_dir / "config.json").write_text(json.dumps({**cfg, "device": str(device)}, indent=2))

    def losses(batch: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        o_p = resize_uint8(batch["obs"].permute(0, 1, 4, 2, 3), cfg["image_size"]).to(device)
        logits = policy(o_p)
        ce_mask = batch["z_mask"] & batch["ce"][:, None]
        ce = (
            F.cross_entropy(logits[ce_mask], batch["z"][ce_mask].to(device))
            if ce_mask.any()
            else None
        )
        lab = batch["lab"] & batch["mse"][:, None]
        mse = None
        if lab.any():
            o_g = resize_uint8(batch["obs"].permute(0, 1, 4, 2, 3), cfg["decoder_image_size"]).to(
                device
            )
            sel = lab.view(-1)
            o_rep = o_g.repeat_interleave(cfg["chunk_size"], 0)[sel]
            z_flat = batch["z"].view(-1)[sel].to(device)
            y = (batch["actions"][lab].to(device) - decoder.target_mean) / decoder.target_std
            mse = F.mse_loss(decoder(o_rep, z_flat), y)
        return ce, mse

    history, best, step = [], float("inf"), 0
    for epoch in range(1, cfg["num_epochs"] + 1):
        policy.train()
        decoder.train()
        start = time.time()
        train_ce = train_mse = 0.0
        for batch in loader:
            ce, mse = losses(batch)
            loss = (ce if ce is not None else 0) + cfg["action_weight"] * (
                mse if mse is not None else 0
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sched.step()
            step += 1
            train_ce += ce.item() if ce is not None else 0
            train_mse += mse.item() if mse is not None else 0
        policy.eval()
        decoder.eval()
        val_ce = val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ce, mse = losses(batch)
                val_ce += ce.item() if ce is not None else 0
                val_mse += mse.item() if mse is not None else 0
        val_ce, val_mse = val_ce / len(val_loader), val_mse / len(val_loader)
        v = val_ce + cfg["action_weight"] * val_mse
        history.append(
            {
                "epoch": epoch,
                "train_ce": train_ce / len(loader),
                "train_mse": train_mse / len(loader),
                "val_ce": val_ce,
                "val_mse": val_mse,
                "val_loss": v,
                "seconds": time.time() - start,
            }
        )
        print(
            f"epoch {epoch}: train ce {train_ce / len(loader):.4f}, "
            f"mse {train_mse / len(loader):.4f}, "
            f"val ce {val_ce:.4f}, mse {val_mse:.4f}"
        )
        checkpoint = {
            "policy": policy.state_dict(),
            "decoder": decoder.state_dict(),
            "config": cfg,
            "epoch": epoch,
        }
        if v < best:
            best = v
            torch.save(checkpoint, out_dir / "policy_best.ckpt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(checkpoint, out_dir / "policy_last.ckpt")

    def names(ids: list[int]) -> list[str]:
        return [f"{Path(index[i]['dir']).name}/{index[i]['file']}" for i in ids]

    split = {"train": names(train_ids), "val": names(val_ids)}
    collections = []
    for e in index:
        d = Path(e["dir"])
        if not any(c[0] == d for c in collections):
            collections.append((d, e["role"]))
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    record = {
        "chunk_size": cfg["chunk_size"],
        "num_epochs": cfg["num_epochs"],
        "best_epoch": best_epoch,
        "last_epoch": cfg["num_epochs"] - 1,
        "best_val_loss": min(h["val_loss"] for h in history),
        "split": split,
        "data": describe_run(collections, split),
        "config": cfg,
    }
    (out_dir / "lapa_run.json").write_text(json.dumps(record, indent=2))
    return record


class LapaPolicy:
    """A trained LAPA policy with the interface `evaluate.rollout` uses: images only,
    absolute targets (B, chunk, 7) per call. A chunk's latent actions are decoded against
    the current frame; ACT's temporal ensembling weights the freshest query most."""

    action_mode = "absolute"

    def __init__(self, ckpt_path: Path, device: torch.device):
        ckpt_path = Path(ckpt_path)
        run = json.loads((ckpt_path.parent / "lapa_run.json").read_text())
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["config"]
        policy = LatentPolicy(
            cfg["codebook"],
            chunk=cfg["chunk_size"],
            hidden=cfg["hidden_dim"],
            ff=cfg["dim_feedforward"],
            enc_layers=cfg["enc_layers"],
            dec_layers=cfg["dec_layers"],
            nheads=cfg["nheads"],
            pretrained=False,
        )
        policy.load_state_dict(ckpt["policy"])
        decoder = ActionDecoder(cfg["codebook"], hidden=cfg["decoder_hidden"], pretrained=False)
        decoder.load_state_dict(ckpt["decoder"])
        self.policy = policy.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.device = device
        self.run = run
        self.cfg = SimpleNamespace(chunk=run["chunk_size"])
        self.uses_robot_state = False
        self.decoder_size = cfg["decoder_image_size"]

    @torch.no_grad()
    def predict(self, images: torch.Tensor, qpos: np.ndarray | None = None) -> torch.Tensor:
        """uint8 (B, cam, H, W, 3) -> absolute targets (B, chunk, 7)."""
        x = images.permute(0, 1, 4, 2, 3)  # (B, cam, 3, H, W)
        z = self.policy(x).argmax(-1)  # (B, chunk)
        g = resize_uint8(x, self.decoder_size)
        out = self.decoder.predict(g.repeat_interleave(self.cfg.chunk, 0), z.view(-1))
        return out.view(len(x), self.cfg.chunk, ACTION_DIM)
