"""Train the IDM on Modal: one A10G container per seed.

The config's collections (and its DANN target, if any) are downloaded once from the public
Hugging Face dataset pinned in `configs/data.toml` ([hub]) into the volume
`ecse552-idm-data`. Each run calls
`idm.train_idm` and writes best.pt, last.pt, history.json and config.json to
`ecse552-idm-outputs:/<config>/seed<seed>`. No secrets are needed.

Iterative BCO (g, docs/experiments/bco.md): `bco:` and `out:` paths in `collections` and
`init` are read from the volumes (`action_free.bco.resolve`), not downloaded; with a
`[label]` table the run then labels that collection into `[label] out` in `ecse552-bco`.

Runs are spawned, so with --detach they finish even if this terminal closes once the
"spawned" lines are printed (the summary is then only in the volume's history.json).

    modal run scripts/modal_train_idm.py --config configs/idm.toml --smoke
    modal run --detach scripts/modal_train_idm.py --config configs/idm.toml --seeds 0
    modal volume get ecse552-idm-outputs idm/seed0 outputs/idm/
"""

import os
import tomllib
from pathlib import Path

import modal

GPU = "A10G"
app = modal.App("ecse552-idm")
data_volume = modal.Volume.from_name("ecse552-idm-data", create_if_missing=True)
out_volume = modal.Volume.from_name("ecse552-idm-outputs", create_if_missing=True)
bco_volume = modal.Volume.from_name("ecse552-bco", create_if_missing=True)
VOLUME_PREFIXES = ("bco:", "out:")  # paths read from the volumes, not downloaded

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.15.0", "torchvision==0.29.2", "numpy==2.2.6", "huggingface_hub==2.2.0"
    )
    # Bake the ImageNet ResNet-18 weights into the image.
    .run_commands(
        'python -c "from torchvision.models import resnet18, ResNet18_Weights; '
        'resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"'
    )
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
    timeout=4 * 3600,
    volumes={"/root/data": data_volume, "/outputs": out_volume, "/bco": bco_volume},
)
def train(cfg: dict, run: str, max_episodes: int | None, smoke: bool = False) -> dict:
    from action_free.bco import resolve
    from action_free.idm import label_collection, load_idm, select_device, split_files, train_idm

    os.chdir("/root")  # the config's `data/<name>` paths resolve to the data volume
    out_dir = Path("/outputs") / run
    s = cfg["seed"]
    cfg = {**cfg, "collections": [resolve(c, s, smoke) for c in cfg["collections"]]}
    if cfg.get("init"):
        cfg["init"] = resolve(cfg["init"], s, smoke)
    try:
        result = train_idm(cfg, out_dir, max_episodes)
    finally:
        out_volume.commit()
    if "label" in cfg:
        files = split_files(Path(cfg["label"]["collection"]))[:max_episodes]
        out = Path(resolve(cfg["label"]["out"], s, smoke))
        device = select_device()
        model = load_idm(out_dir / "best.pt", device)
        label_collection(model, files, cfg["image_size"], out, device)
        bco_volume.commit()
        print(f"labelled {len(files)} episodes into {out}")
    return result


@app.local_entrypoint()
def main(config: str = "configs/idm.toml", seeds: str = "0", smoke: bool = False) -> None:
    cfg = tomllib.loads(Path(config).read_text())
    hub = tomllib.loads(Path("configs/data.toml").read_text())["hub"]
    collections = cfg["collections"] + ([cfg["target"]] if "target" in cfg else [])
    collections += [cfg["label"]["collection"]] if "label" in cfg else []
    collections = [c for c in collections if not c.startswith(VOLUME_PREFIXES)]
    download.remote(collections, hub["dataset"], hub["revision"])

    name = Path(config).stem + ("-smoke" if smoke else "")
    if smoke:
        cfg["epochs"] = 1
    jobs = [
        ({**cfg, "seed": int(s)}, f"{name}/seed{s}", 10 if smoke else None, smoke)
        for s in seeds.split(",")
    ]
    calls = [train.spawn(*job) for job in jobs]  # independent of this client
    for (_, run, *_), call in zip(jobs, calls):
        print(f"spawned {run} ({call.object_id})")
    for (_, run, *_), call in zip(jobs, calls):
        kept = call.get()
        mae = ", ".join(f"{k} {v:.4f}" for k, v in kept["val"]["mae"].items())
        which = "last" if "target" in cfg else "best"  # DANN keeps the last epoch
        print(
            f"{run}: {which} epoch {kept['epoch']}, "
            f"validation sMAE {kept['val']['smae']:.4f}; {mae}"
        )
