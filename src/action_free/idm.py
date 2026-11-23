"""Inverse dynamics model (IDM) and its data.

Our ACT trains on image-action pairs. However, we do not have actions, only videos.
The goal of IDM is to predict the absolute target a_t from the images o_t and o_t+1 alone,
two consecutive images. It trains on labeled source episodes. Target videos are read through
their images only (labeled=False), so their actions and states never reach it and they are
read separately, and only to measure label error.

Images are resized to 'image_size' (128 by default, 224 keeps the stored resolution). Splits
follow 'data.split_of': train and validation by episode within a collection, and the held-out
source collection is the test set.

With cfg["target"], training is domain-adversarial (DANN, method c): a discriminator tells
source from target features, and a gradient reversal layer pushes the features to make them
indistinguishable. The target videos are read through their images only (labeled=False).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from action_free.actions import to_absolute
from action_free.data import episode_files, split_of

IMAGE_SIZE = 128
MIN_STD = 1e-3  # floor of the target std: the joint that never moves (wrist2) stays at it
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ACTION_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3", "gripper")


def split_files(data_dir: Path, split: str | None = None) -> list[Path]:
    """Episode files of a collection in one split ("train", "val"), or all of them (None)."""
    files = episode_files(data_dir)
    if split is None:
        return files
    return [f for f in files if split_of(int(f.stem.split("_")[1])) == split]


def load_frames(f: Path, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """An episode's images, (T + 1, cameras, 3, S, S) uint8, resized with antialiasing."""
    with np.load(f) as ep:
        x = torch.from_numpy(ep["observations"]).permute(0, 1, 4, 2, 3)
    if x.shape[-1] != image_size:
        steps, cameras = x.shape[:2]
        x = F.interpolate(
            x.flatten(0, 1).float(), size=(image_size, image_size), mode="bilinear", antialias=True
        )
        x = x.round().clamp(0, 255).to(torch.uint8).unflatten(0, (steps, cameras))
    return x


def absolute_targets(f: Path) -> np.ndarray:
    """The true absolute targets (T, 7) of an episode: IDM labels for source, evaluation only
    for targets."""
    with np.load(f) as ep:
        return to_absolute(ep["actions"], ep["qpos"])


@dataclass
class Transitions:
    """Transitions (o_t, o_t+1) of several episodes, each frame stored once.
    This avoids storing twice in pairs (o_t, o_t+1) then (o_t+1, o_t+2)"""

    frames: torch.Tensor  # (frames, cameras, 3, S, S) uint8, episodes concatenated
    index: torch.Tensor  # (N,) frame of o_t; o_t+1 is the next frame of the same episode
    episode: torch.Tensor  # (N,) position of the transition's episode in `files`
    files: list[Path]
    targets: torch.Tensor | None  # (N, 7) absolute targets; None for action-free videos


def load_transitions(
    files: list[Path], image_size: int = IMAGE_SIZE, labeled: bool = True
) -> Transitions:
    frames, index, episode, targets, start = [], [], [], [], 0
    # start tracks where current episode begins in the long frame list
    for i, f in enumerate(files):
        x = load_frames(f, image_size)
        steps = len(x) - 1
        frames.append(x)
        index.append(torch.arange(start, start + steps))
        episode.append(torch.full((steps,), i))
        if labeled:
            targets.append(torch.from_numpy(absolute_targets(f)))
        start += len(x)
    return Transitions(
        torch.cat(frames),
        torch.cat(index),
        torch.cat(episode),
        list(files),
        torch.cat(targets) if labeled else None,
    )


class TransitionDataset(Dataset):
    """(o_t, o_t+1) as uint8 images, plus the target when labeled."""

    def __init__(self, transitions: Transitions):
        self.t = transitions

    def __len__(self) -> int:
        return len(self.t.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        k = self.t.index[i]
        item = {"obs": self.t.frames[k], "next_obs": self.t.frames[k + 1]}
        if self.t.targets is not None:
            item["target"] = self.t.targets[i]
        return item


class IDM(nn.Module):
    """(o_t, o_t+1) from every camera -> absolute target a_t.

    Each camera's two frames (each has 3 color channels) and their difference (also 3 colour
    channels) are stacked as 9 channels and encoded by one pre-trained ResNet-18 shared
    across cameras; the camera features are concatenated (features, where the DANN
    discriminator attaches) and an MLP regresses the target. Targets are standardized with
    source training statistics stored in the model, so predict returns absolute targets.
    Training difference channels works as motion detectors.
    """

    def __init__(
        self, action_dim: int = 7, cameras: int = 3, hidden: int = 512, pretrained: bool = True
    ):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old = backbone.conv1
        conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Half the pretrained filter on each frame: a still pair gives the pretrained response.
            # The difference channels start at zero and learn motion filters from there.
            conv1.weight.zero_()
            conv1.weight[:, 0:3] = old.weight / 2
            conv1.weight[:, 3:6] = old.weight / 2
        backbone.conv1 = conv1
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.head = nn.Sequential(
            nn.Linear(cameras * 512, hidden), nn.ReLU(), nn.Linear(hidden, action_dim)
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))
        self.register_buffer("target_mean", torch.zeros(action_dim))
        self.register_buffer("target_std", torch.ones(action_dim))

    def set_target_stats(self, targets: torch.Tensor, min_std: float = MIN_STD) -> None:
        """Standardize with source training targets; min_std covers the joint that never moves."""
        self.target_mean.copy_(targets.mean(0))
        self.target_std.copy_(targets.std(0).clamp_min(min_std))

    def features(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """uint8 (B, cameras, 3, S, S) twice -> (B, cameras * 512)."""
        o = (obs.float() / 255 - self.mean) / self.std
        o_next = (next_obs.float() / 255 - self.mean) / self.std
        x = torch.cat([o, o_next, o_next - o], dim=2)  # (B, cameras, 9, S, S)
        b, c = x.shape[:2]
        return self.encoder(x.flatten(0, 1)).view(b, c * 512)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Standardized targets (B, action_dim): what the loss compares."""
        return self.head(self.features(obs, next_obs))

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Absolute targets (B, action_dim): the pseudo-labels."""
        return self(obs, next_obs) * self.target_std + self.target_mean


class GradReverse(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by -lambd in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad, None


class DomainDiscriminator(nn.Module):
    """IDM features (B, features) -> one logit per pair (B,): source = 0, target = 1."""

    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def dann_lambda(p: float, lambda_max: float, gamma: float) -> float:
    """Weight of the reversed gradient at training progress p in [0, 1]: 0 at the start,
    close to lambda_max at the end."""
    return lambda_max * (2 / (1 + math.exp(-gamma * p)) - 1)


def repeat_forever(loader: DataLoader):
    """Batches of a loader without end, reshuffled at every pass."""
    while True:
        for batch in loader:
            yield batch


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_transitions(
    model: IDM, transitions: Transitions, device: torch.device, batch_size: int = 256
) -> torch.Tensor:
    """Absolute targets (N, 7) for every transition, in order."""
    model.eval()
    loader = DataLoader(TransitionDataset(transitions), batch_size=batch_size)
    preds = [model.predict(b["obs"].to(device), b["next_obs"].to(device)).cpu() for b in loader]
    return torch.cat(preds)


@torch.no_grad()
def domain_accuracy(
    model: IDM,
    discriminator: DomainDiscriminator,
    transitions: Transitions,
    domain: int,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Share of the pairs that the discriminator assigns to their domain (0 source, 1 target)."""
    model.eval()
    loader = DataLoader(TransitionDataset(transitions), batch_size=batch_size)
    correct = 0
    for b in loader:
        logits = discriminator(model.features(b["obs"].to(device), b["next_obs"].to(device)))
        correct += ((logits > 0) == (domain == 1)).sum().item()
    return correct / len(transitions.index)


def errors(pred: torch.Tensor, target: torch.Tensor, std: torch.Tensor) -> dict:
    """Mean absolute error per component (radians; gripper in [0, 1]); `smae`, the MAE of
    standardized targets averaged over the components that move (the selection metric); and
    `mse`, the mean squared error of standardized targets (the training loss)."""
    mae = (pred - target).abs().mean(0)
    smae = (mae / std)[std > MIN_STD].mean().item()
    mse = (((pred - target) / std) ** 2).mean().item()
    return {"smae": smae, "mse": mse, "mae": dict(zip(ACTION_NAMES, mae.tolist()))}


def train_idm(cfg: dict, out_dir: Path, max_episodes: int | None = None) -> dict:
    """Train on the train split of cfg["collections"]; keep the epoch with the lowest
    standardized MAE on their validation split (best.pt). Writes history.json in out_dir.

    With cfg["target"] (DANN): each source batch is paired with a target batch of the same
    size, the action loss stays on source only, and best.pt is the last epoch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = select_device()
    size = cfg["image_size"]
    dann = "target" in cfg

    def files(split: str) -> list[Path]:
        return [f for c in cfg["collections"] for f in split_files(Path(c), split)[:max_episodes]]

    train, val = load_transitions(files("train"), size), load_transitions(files("val"), size)
    model = IDM(pretrained=cfg["pretrained"]).to(device)
    model.set_target_stats(train.targets.to(device))
    std = model.target_std.cpu()

    loader = DataLoader(
        TransitionDataset(train), batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    if dann:
        # Images only: the target's actions and states are never read (R1).
        target = Path(cfg["target"])
        target_train = load_transitions(
            split_files(target, "train")[:max_episodes], size, labeled=False
        )
        target_val = load_transitions(
            split_files(target, "val")[:max_episodes], size, labeled=False
        )
        target_loader = DataLoader(
            TransitionDataset(target_train),
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        target_batches = repeat_forever(target_loader)
        discriminator = DomainDiscriminator(
            model.head[0].in_features, cfg["discriminator_hidden"]
        ).to(device)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg["discriminator_lr"])
        discriminator_loss = nn.BCEWithLogitsLoss()
    (out_dir / "config.json").write_text(json.dumps({**cfg, "device": str(device)}, indent=2))

    history, best, step = [], float("inf"), 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        start, total, domain_total = time.time(), 0.0, 0.0
        for b in loader:
            obs, next_obs = b["obs"].to(device), b["next_obs"].to(device)
            y = (b["target"].to(device) - model.target_mean) / model.target_std
            n = len(y)  # the first n pairs are source
            if dann:
                # One features() pass on both domains, so batch norm sees both.
                t = next(target_batches)
                obs = torch.cat([obs, t["obs"].to(device)])
                next_obs = torch.cat([next_obs, t["next_obs"].to(device)])
            features = model.features(obs, next_obs)
            mse = F.mse_loss(model.head(features[:n]), y)  # action loss on source only
            loss = mse
            if dann:
                lambd = dann_lambda(step / total_steps, cfg["dann_lambda"], cfg["dann_gamma"])
                domain = torch.cat([torch.zeros(n), torch.ones(len(features) - n)]).to(device)
                logits = discriminator(GradReverse.apply(features, lambd))
                domain_loss = discriminator_loss(logits, domain)
                loss = mse + domain_loss
                domain_total += domain_loss.item()
                d_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if dann:
                d_optimizer.step()
            sched.step()
            step += 1
            total += mse.item()
        v = errors(predict_transitions(model, val, device), val.targets, std)
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
        checkpoint = {"model": model.state_dict(), "config": cfg, "epoch": epoch}
        if dann:
            # TSR / TTR: share of source / target validation pairs classified correctly.
            tsr = domain_accuracy(model, discriminator, val, 0, device)
            ttr = domain_accuracy(model, discriminator, target_val, 1, device)
            history[-1].update(
                {"lambda": lambd, "domain_loss": domain_total / len(loader), "tsr": tsr, "ttr": ttr}
            )
            print(
                f"  lambda {lambd:.3f}, domain loss {domain_total / len(loader):.4f}, "
                f"TSR {tsr:.3f}, TTR {ttr:.3f}"
            )
            checkpoint["discriminator"] = discriminator.state_dict()
        if dann or v["smae"] < best:
            best = v["smae"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(checkpoint, out_dir / "last.pt")
    if dann:
        return history[-1]
    return min(history, key=lambda h: h["val"]["smae"])


def load_idm(path: Path, device: torch.device) -> IDM:
    ckpt = torch.load(path, map_location="cpu")
    model = IDM(pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
