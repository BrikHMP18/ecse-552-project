"""Train LAPA (method h) on Modal: one A10G container per seed (docs/experiments/lapa.md).

Stage 1, the latent action model (configs/lam.toml): pretrains on the images of the source
demos and the target videos, downloaded once from the public dataset pinned in
`configs/data.toml` ([hub]) into the volume `ecse552-lapa-data`. Each run calls
`lapa.train_lam` and writes best.pt, last.pt, history.json and config.json to
`ecse552-lapa-outputs:/lam/seed<seed>`.

Stage 2, the policy (configs/lapa_{A,D,E}.toml): `prepare` exports the episodes (source,
target, turned) to the policy stage's HDF5 layout once, seed-independent, into the data
volume. Each run computes the latent pseudo-labels with the frozen LAM of the same seed
(`lam/seed<seed>/best.pt` in the outputs volume) and calls `lapa.train_lapa`, which writes
policy_best.ckpt, policy_last.ckpt, lapa_run.json, history.json and config.json to
`ecse552-lapa-outputs:/<config>/n300/seed<seed>`. A config with a `lam` key is the policy
stage; without, the LAM stage. No secrets are needed.

Runs are spawned, so with --detach they finish even if this terminal closes once the
"spawned" lines are printed (the summary is then only in the volume's history.json).

    modal run scripts/modal_train_lapa.py --config configs/lam.toml --smoke
    modal run --detach scripts/modal_train_lapa.py --config configs/lam.toml --seeds 0,1,2
    modal run --detach scripts/modal_train_lapa.py --config configs/lapa_A.toml --seeds 0,1,2
    modal volume get ecse552-lapa-outputs lapa_A outputs/lapa_A/
"""

import json
import os
import tomllib
from pathlib import Path

import modal

GPU = "A10G"
app = modal.App("ecse552-lapa")
data_volume = modal.Volume.from_name("ecse552-lapa-data", create_if_missing=True)
out_volume = modal.Volume.from_name("ecse552-lapa-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "numpy==2.2.6",
        "huggingface_hub==2.2.0",
        "h5py==3.16.0",
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


@app.function(image=image, volumes={"/data": data_volume}, timeout=2 * 3600)
def prepare(roles: list[dict], out: str, max_episodes: int | None) -> str:
    """Export the episodes to the policy stage's HDF5 layout once (seed-independent)."""
    from action_free.data import episode_files
    from action_free.lapa import export_lapa

    collections = []
    for r in roles:
        files = episode_files(Path("/data") / Path(r["dir"]).name, max_episodes)
        collections.append((files, r["role"], r["labeled"]))
    out_dir = Path("/data") / "hdf5" / out
    if not (out_dir / "episodes.json").exists():
        export_lapa(collections, out_dir)
        data_volume.commit()
    index = json.loads((out_dir / "episodes.json").read_text())
    print(f"{len(index)} episodes in {out_dir}")
    return f"hdf5/{out}"


@app.function(
    image=image,
    gpu=GPU,
    memory=32768,
    timeout=8 * 3600,
    volumes={"/root/data": data_volume, "/outputs": out_volume},
)
def train(cfg: dict, run: str, dataset_dir: str | None, max_episodes: int | None) -> dict:
    import shutil

    from action_free.lapa import train_lam, train_lapa

    os.chdir("/root")  # the config's `data/<name>` paths resolve to the data volume
    out_dir = Path("/outputs") / run
    try:
        if dataset_dir is None:
            result = train_lam(cfg, out_dir, max_episodes)
        else:
            cfg = {**cfg, "lam": str(Path("/outputs") / cfg["lam"].format(seed=cfg["seed"]))}
            local = Path("/tmp/lapa_data")
            shutil.copytree(Path("/root/data") / dataset_dir, local, dirs_exist_ok=True)
            result = train_lapa(cfg, local, out_dir)
    finally:
        out_volume.commit()
    return result


@app.local_entrypoint()
def main(config: str = "configs/lam.toml", seeds: str = "0", smoke: bool = False) -> None:
    cfg = tomllib.loads(Path(config).read_text())
    hub = tomllib.loads(Path("configs/data.toml").read_text())["hub"]
    stage = "policy" if "lam" in cfg else "lam"
    if stage == "lam":
        collections = cfg["collections"]
    else:
        collections = cfg["collections"] + cfg.get("targets", []) + cfg.get("turned", [])
    download.remote(collections, hub["dataset"], hub["revision"])

    name = Path(config).stem + ("-smoke" if smoke else "")
    max_episodes = 10 if smoke else None
    dataset_dir = None
    if stage == "policy":
        roles = (
            [{"dir": c, "role": "source", "labeled": True} for c in cfg["collections"]]
            + [{"dir": c, "role": "target", "labeled": False} for c in cfg.get("targets", [])]
            + [{"dir": c, "role": "turned", "labeled": True} for c in cfg.get("turned", [])]
        )
        dataset_dir = prepare.remote(roles, name, max_episodes)
    if smoke:
        cfg["num_epochs" if stage == "policy" else "epochs"] = 1
        if stage == "policy":
            cfg["lam"] = cfg["lam"].replace("lam/", "lam-smoke/")  # the smoke LAM's run
    run_root = f"{name}/{'n300/' if stage == 'policy' else ''}seed"
    jobs = [
        ({**cfg, "seed": int(s)}, f"{run_root}{s}", dataset_dir, max_episodes)
        for s in seeds.split(",")
    ]
    calls = [train.spawn(*job) for job in jobs]  # independent of this client
    for (_, run, _, _), call in zip(jobs, calls):
        print(f"spawned {run} ({call.object_id})")
    for (_, run, _, _), call in zip(jobs, calls):
        result = call.get()
        if dataset_dir is None:
            print(f"{run}: best epoch {result['epoch']}, val recon {result['val_recon']:.4f}")
        else:
            print(
                f"{run}: best epoch {result['best_epoch']}, val loss {result['best_val_loss']:.4f}"
            )
