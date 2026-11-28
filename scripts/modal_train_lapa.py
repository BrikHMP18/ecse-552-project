"""Train the LAPA labeler (method h) on Modal: one A10G container per stage and seed
(docs/experiments/lapa.md, notes/plan-lapa.md).

Stage 1, the latent action quantization model (configs/laq.toml): pretrains on the images
of the source demos, the turned demos and the target videos, downloaded once from the
public dataset pinned in `configs/data.toml` ([hub]) into the volume `ecse552-lapa-data`.
Each run calls `lapa.train_laq` and writes best.pt, last.pt, history.json and config.json
(with the parameter counts) to `ecse552-lapa-outputs:/laq/seed<seed>`.

Stage 2, the action decoder g (configs/lapa_g.toml): trains g on the source and turned
demos with z labels from the frozen LAQ of the same seed (`laq/seed<seed>/best.pt` in the
outputs volume), and writes best.pt, last.pt, history.json and config.json to
`ecse552-lapa-outputs:/lapa_g/seed<seed>`.

Stage 3, the labels: `--label` writes the pseudo-labels of A, D and E with the LAQ and g
of one seed to `ecse552-lapa-outputs:/labels/<target>-deterministic-lapa-seed<seed>`. A
team member downloads them and uploads the folders to the dataset repository as
`labels/<name>/`; ACT then trains with `scripts/modal_train.py --config
configs/act_lapa_{A,D,E}.toml`.

Stage detection: a config with a `laq` key is the g stage; otherwise the LAQ stage. No
secrets are needed.

    modal run scripts/modal_train_lapa.py --config configs/laq.toml --smoke
    modal run --detach scripts/modal_train_lapa.py --config configs/laq.toml --seeds 0
    modal run --detach scripts/modal_train_lapa.py --config configs/lapa_g.toml --seeds 0
    modal run scripts/modal_train_lapa.py --label laq/seed0/best.pt,lapa_g/seed0/best.pt
    modal volume get ecse552-lapa-outputs labels data/labels/
"""

import os
import tomllib
from pathlib import Path

import modal

GPU = "A10G"
app = modal.App("ecse552-lapa")
data_volume = modal.Volume.from_name("ecse552-lapa-data", create_if_missing=True)
out_volume = modal.Volume.from_name("ecse552-lapa-outputs", create_if_missing=True)

# The targets the labeler labels (docs/experiments/lapa.md, fixed in advance).
LABEL_COLLECTIONS = (
    "A-deterministic-n200-seed100000",
    "D-deterministic-n200-seed400000",
    "E-deterministic-n200-seed500000",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"LAQ_ROOT": "/root/third_party/lapa/laq"})
    .uv_pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "numpy==2.2.6",
        "huggingface_hub==2.2.0",
        "h5py==3.16.0",
        # the vendored LAQ's imports (versions from uv.lock)
        "einops==0.8.2",
        "beartype==0.23.1",
    )
    # Bake the ImageNet ResNet-18 weights into the image.
    .run_commands(
        'python -c "from torchvision.models import resnet18, ResNet18_Weights; '
        'resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"'
    )
    # Local dirs last: Modal adds them at container start, not at build time.
    .add_local_dir("third_party/lapa/laq", "/root/third_party/lapa/laq", ignore=[".git"])
    .add_local_dir("src/action_free", "/root/action_free")
)


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600)
def download(collections: list[str], dataset: str, revision: str) -> None:
    """Download the collections that are not in the volume yet (`data/<name>` -> `/data/<name>`)."""
    from huggingface_hub import snapshot_download

    missing = [Path(c).name for c in collections if not (Path("/data") / Path(c).name).exists()]
    if missing:
        snapshot_download(
            dataset,
            repo_type="dataset",
            revision=revision,
            local_dir="/data",
            allow_patterns=[f"{name}/*" for name in missing],
        )
        data_volume.commit()
    print(f"downloaded {missing or 'nothing'}; in the volume: {sorted(os.listdir('/data'))}")


@app.function(
    image=image,
    gpu=GPU,
    memory=32768,
    timeout=8 * 3600,
    volumes={"/root/data": data_volume, "/outputs": out_volume},
)
def train(cfg: dict, run: str, max_episodes: int | None) -> dict:
    from action_free.lapa import train_g, train_laq

    os.chdir("/root")  # the config's `data/<name>` paths resolve to the data volume
    out_dir = Path("/outputs") / run
    try:
        if "laq" in cfg:  # the g stage: its LAQ checkpoint sits in the outputs volume
            cfg = {**cfg, "laq": str(Path("/outputs") / cfg["laq"].format(seed=cfg["seed"]))}
            result = train_g(cfg, out_dir, max_episodes)
        else:
            result = train_laq(cfg, out_dir, max_episodes)
    finally:
        out_volume.commit()
    return result


@app.function(
    image=image,
    gpu=GPU,
    memory=32768,
    timeout=4 * 3600,
    volumes={"/root/data": data_volume, "/outputs": out_volume},
)
def label_targets(laq_ckpt: str, g_ckpt: str, max_episodes: int | None) -> list[str]:
    """Pseudo-label A, D and E with the LAQ and g checkpoints (paths in the volume)."""
    import torch

    from action_free.data import episode_files
    from action_free.lapa import label_collection, load_g, load_laq

    os.chdir("/root")
    device = torch.device("cuda")
    laq = load_laq(Path("/outputs") / laq_ckpt, device)
    decoder = load_g(Path("/outputs") / g_ckpt, device)
    seed = laq.config["seed"]
    outs = []
    try:
        for c in LABEL_COLLECTIONS:
            files = episode_files(Path("data") / c, max_episodes)
            out = Path("/outputs") / "labels" / f"{c.split('-n')[0]}-lapa-seed{seed}"
            label_collection(laq, decoder, files, laq.config["image_size"], out, device)
            outs.append(str(out))
            print(f"labeled {len(files)} episodes of {c} into {out}")
    finally:
        out_volume.commit()
    return outs


@app.local_entrypoint()
def main(
    config: str = "configs/laq.toml",
    seeds: str = "0",
    label: str = "",
    smoke: bool = False,
) -> None:
    if label:
        laq_ckpt, g_ckpt = label.split(",")
        if smoke:
            raise SystemExit("--smoke is for the training stages.")
        print(label_targets.remote(laq_ckpt, g_ckpt, None))
        return
    cfg = tomllib.loads(Path(config).read_text())
    hub = tomllib.loads(Path("configs/data.toml").read_text())["hub"]
    stage = "g" if "laq" in cfg else "laq"
    download.remote(cfg["collections"], hub["dataset"], hub["revision"])

    name = Path(config).stem + ("-smoke" if smoke else "")
    max_episodes = 10 if smoke else None
    if smoke:
        cfg["epochs"] = 1
        if stage == "laq":
            cfg["record_reference"] = False  # the 300M reference count in a smoke run
        else:
            cfg["laq"] = cfg["laq"].replace("laq/", "laq-smoke/")  # the smoke LAQ's run
    run_root = f"{name}/seed"
    jobs = [({**cfg, "seed": int(s)}, f"{run_root}{s}", max_episodes) for s in seeds.split(",")]
    calls = [train.spawn(*job) for job in jobs]  # independent of this client
    for (_, run, _), call in zip(jobs, calls):
        print(f"spawned {run} ({call.object_id})")
    for (_, run, _), call in zip(jobs, calls):
        result = call.get()
        if stage == "laq":
            print(f"{run}: best epoch {result['epoch']}, val recon {result['val_recon']:.4f}")
        else:
            print(f"{run}: best epoch {result['epoch']}, val sMAE {result['val']['smae']:.4f}")
