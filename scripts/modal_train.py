"""Train original-ACT baselines on Modal: one A10G container per (data size, seed).

Data comes from the public Hugging Face dataset pinned in `configs/data.toml` ([hub]): the
source collection and the config's optional `[[targets]]` (stored actions for the oracle (d),
pseudo-labels for (b)/(c)) are downloaded and exported once to ACT's HDF5 layout in the Modal
volume `ecse552-data` (`act_original.export_dataset`). Each run trains upstream ACT
(`act_original.train`) in `ecse552-outputs:/<config>/n<episodes>/seed<seed>` and pushes
`policy_best.ckpt`, `policy_last.ckpt`, `dataset_stats.pkl`, `act_run.json`, `history.json`,
the loss plots and `provenance.json` to the same path of a public Hugging Face model repo
(default `Autobrik/ecse552-<config>`). Upstream's every-100-epoch checkpoints stay in the
volume (learning curves). Losses go to W&B through the Modal secret `wandb-secret`.

    modal run scripts/modal_train.py::check                      # GPU, keys, upstream imports
    modal run scripts/modal_train.py --smoke                     # 2 epochs, 10 demos, no push
    modal run --detach scripts/modal_train.py --num-episodes 300 --seeds 0
    modal run --detach scripts/modal_train.py --num-episodes 100,200,300 --seeds 0   # sweep
    modal run --detach scripts/modal_train.py --config configs/act_oracle_A.toml --seeds 0  # (d)

Checkpoint selection and evaluation run locally (`scripts/select_checkpoint.py`,
`scripts/evaluate_policy.py`), or on Modal with `scripts/modal_rollouts.py`.

Secrets: `ECSE552_MODAL_SECRETS` lists the Modal secrets to attach (default
`huggingface-secret,wandb-secret`; empty for none). Without `WANDB_API_KEY` nothing is logged
to W&B; without a Hugging Face token use `--no-push` (smoke runs never push).
Labels named `bco:<path>` (iterative BCO, docs/experiments/bco.md) are read from the volume
`ecse552-bco`, and `lapa:<path>` (LAPA labeler, docs/experiments/lapa.md) from the volume
`ecse552-lapa-outputs`, not downloaded.
"""

import os
import subprocess
import tomllib
from pathlib import Path

import modal

GPU = "A10G"  # Proposal: one NVIDIA A10G.
app = modal.App("ecse552-act")
data_volume = modal.Volume.from_name("ecse552-data", create_if_missing=True)
out_volume = modal.Volume.from_name("ecse552-outputs", create_if_missing=True)
bco_volume = modal.Volume.from_name("ecse552-bco", create_if_missing=True)
lapa_volume = modal.Volume.from_name("ecse552-lapa-outputs", create_if_missing=True)
SECRET_NAMES = os.environ.get("ECSE552_MODAL_SECRETS", "huggingface-secret,wandb-secret")
secrets = [modal.Secret.from_name(n) for n in SECRET_NAMES.split(",") if n]
PUSH_FILES = (
    "policy_best.ckpt",
    "policy_last.ckpt",
    "dataset_stats.pkl",
    "act_run.json",
    "history.json",
    "provenance.json",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libegl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "numpy==2.2.6",
        "huggingface_hub==2.2.0",
        "wandb==0.31.0",
        # original ACT's imports (versions from uv.lock)
        "h5py==3.16.0",
        "einops==0.8.2",
        "ipython==8.39.0",
        "opencv-python-headless==5.0.0.93",
        "matplotlib==3.11.2",
        "tqdm==4.70.1",
        "dm-control==1.0.48",
        "mujoco==3.15.0",
    )
    # ECSE552_MODAL_SECRETS too, so the container attaches the same secrets as the client.
    .env(
        {
            "ACT_ROOT": "/root/third_party/act",
            "MUJOCO_GL": "egl",
            "MPLBACKEND": "Agg",
            "ECSE552_MODAL_SECRETS": SECRET_NAMES,
        }
    )
    # Bake the ImageNet ResNet-18 weights into the image.
    .run_commands(
        'python -c "from torchvision.models import resnet18, ResNet18_Weights; '
        'resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"'
    )
    .add_local_dir("third_party/act", "/root/third_party/act", ignore=[".git"])
    .add_local_dir("src/action_free", "/root/action_free")
)

MODEL_CARD = """---
license: mit
tags:
- robotics
- imitation-learning
- act
datasets:
- {dataset}
---

# {repo}

Original ACT ([tonyzhaozh/act](https://github.com/tonyzhaozh/act), unmodified) trained on
[{dataset}](https://huggingface.co/datasets/{dataset}) at revision `{revision}`, for the McGill
ECSE 552 project *Learning to Manipulate a New Object from Action-Free Videos*
([code](https://github.com/BrikHMP18/ecse-552-project), config `{config}`, adapter
`src/action_free/act_original.py`). Differences from ACT: images only (robot state is zeroed),
our 7 action dimensions zero-padded to upstream's 14, variable-length episodes, chunk 20 at
10 Hz (2 s, as ACT's 100 at 50 Hz). Actions are absolute joint targets. Runs on several
collections (oracle (d), pseudo-labelled targets for (b)/(c)) list them in `act_run.json`
(`data.collections`).

Each `n<demos>/seed<k>/` holds one run: `policy_best.ckpt` (lowest validation loss),
`policy_last.ckpt`, `selected.ckpt` and `selection.json` (the better of the two on source
validation rollouts, the one evaluated on test), `dataset_stats.pkl`, `act_run.json`
(arguments, split, data), `history.json` (losses per epoch) and `provenance.json`.
Load with `action_free.act_original.ActPolicy("n300/seed0/selected.ckpt", device)`.
"""


def _hf_whoami() -> str:
    from huggingface_hub import whoami

    return whoami()["name"]


@app.function(image=image, gpu=GPU, timeout=600, secrets=secrets)
def check() -> dict:
    import os

    import torch

    from action_free.act_original import CAMERAS, UPSTREAM_POLICY, _argv, _import_upstream

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    info = {
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hf_user": _hf_whoami(),
        "wandb_key": bool(os.environ.get("WANDB_API_KEY")),
    }
    _import_upstream()
    import imitate_episodes  # noqa: F401  (all upstream imports, incl. dm_control)
    from policy import ACTPolicy

    cfg = {"lr": 1e-5, "num_queries": 20, "kl_weight": 10, "hidden_dim": 512,
           "dim_feedforward": 3200, "camera_names": list(CAMERAS), **UPSTREAM_POLICY}  # fmt: skip
    with _argv(["--ckpt_dir", "x", "--policy_class", "ACT", "--task_name", "sim_x",
                "--seed", "0", "--num_epochs", "1"]):  # fmt: skip
        policy = ACTPolicy(cfg).cuda()
    info["upstream_act_params_M"] = round(sum(p.numel() for p in policy.parameters()) / 1e6, 1)
    print(info)
    return info


@app.function(
    image=image,
    volumes={"/data": data_volume, "/bco": bco_volume, "/lapa": lapa_volume},
    timeout=3600,
    secrets=secrets,
)
def prepare(
    dataset: str,
    revision: str,
    source: str,
    sizes: list[int],
    targets: list[dict],
    exclude_detours: bool,
    use_robot_state: bool,
    model_repo: str | None,
    card: str,
    smoke: bool = False,
) -> dict:
    """Download the collections once, export ACT HDF5 per source size; create the model repo."""
    from huggingface_hub import HfApi, snapshot_download

    from action_free.act_original import export_dataset, plausible
    from action_free.bco import resolve
    from action_free.data import detour_files, episode_files

    def in_volume(actions: str) -> bool:
        return actions.startswith(("bco:", "lapa:"))

    def labels_path(actions: str) -> Path:
        """A target's labels folder in the container (`bco:`/`lapa:` volumes, else hub)."""
        if actions.startswith("lapa:"):
            return Path("/lapa") / actions.removeprefix("lapa:")
        return Path(resolve(actions, 0, smoke))

    local = Path(f"/data/{dataset.replace('/', '__')}/{revision}")
    meta = ("manifest.json", "provenance.txt", "packages.txt", "initial_states.csv")
    patterns = []
    for folder, n in [(source, max(sizes)), *((t["folder"], t["num_episodes"]) for t in targets)]:
        patterns += [f"{folder}/{m}" for m in meta]
        patterns += [f"{folder}/episode_{i:05d}.npz" for i in range(n)]
    patterns += [
        f"{t['actions']}/*"
        for t in targets
        if t["actions"] != "true" and not in_volume(t["actions"])
    ]
    patterns += [
        f"{t['exclude_like']}/*"
        for t in targets
        if t.get("exclude_like") and not in_volume(t["exclude_like"])
    ]
    snapshot_download(
        dataset, repo_type="dataset", revision=revision, allow_patterns=patterns, local_dir=local
    )
    exported = {}
    for n in sizes:
        files = episode_files(local / source, n)
        tag = f"n{n}"
        if exclude_detours:
            dropped = detour_files(local / source)
            files = [f for f in files if f.name not in dropped]
            tag += "-nodetours"
        if use_robot_state:
            tag += "-state"
        collections = [(files, None)]
        for t in targets:
            if t["actions"] == "true":
                labels = None
            elif in_volume(t["actions"]):
                labels = labels_path(t["actions"])  # {seed} is already filled in
            else:
                labels = local / t["actions"]
            target_files = episode_files(local / t["folder"], t["num_episodes"])
            tag += (
                f"+{t['folder']}-n{t['num_episodes']}-{'true' if labels is None else labels.name}"
            )
            if t.get("exclude_like"):
                # Matched samples (method h): keep the same episodes the reference run
                # kept, by its own label filter (docs/experiments/lapa.md).
                ref = local / t["exclude_like"]
                if t["max_label_step"]:
                    target_files = plausible(target_files, ref, t["max_label_step"])
                tag += f"-like{Path(t['exclude_like']).name}-step{t['max_label_step']}"
            elif t["max_label_step"]:
                target_files = plausible(target_files, labels, t["max_label_step"])
                tag += f"-step{t['max_label_step']}"
            collections.append((target_files, labels))
        out = local / "act_hdf5" / source / tag
        if not (out / "episodes.json").exists():
            export_dataset(collections, out, use_robot_state=use_robot_state)
            data_volume.commit()
        total = sum(len(c[0]) for c in collections)
        exported[n] = {"dataset_dir": str(out), "num_episodes": total}
        print(f"{tag}: {total} episodes in {out}")
    if model_repo:
        print(f"Hugging Face user: {_hf_whoami()}")
        api = HfApi()
        api.create_repo(model_repo, private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=model_repo,
            commit_message="Model card",
        )
    return exported


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": data_volume, "/outputs": out_volume},
    timeout=8 * 3600,
    memory=32768,
    secrets=secrets,
)
def train(cfg: dict, data: dict, run_path: str, seed: int, provenance: dict, model_repo):
    import json
    import os
    import shutil

    from action_free.act_original import train as act_train

    data_volume.reload()
    # Upstream reads every sample from disk; use the container's local disk, not the volume.
    local_data = Path("/tmp/act_data")
    shutil.copytree(data["dataset_dir"], local_data, dirs_exist_ok=True)
    out = Path("/outputs") / run_path
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    wandb_run = None
    if cfg.get("wandb_project") and os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity"),
            group=run_path.split("/")[0],
            name=run_path.replace("/", "-"),
            config={**cfg, **data, "seed": seed, **provenance},
            dir=str(out),
        )
    record = act_train(
        local_data,
        out,
        data["num_episodes"],
        seed,
        cfg,
        run_info={"provenance": provenance, "wandb_url": wandb_run.url if wandb_run else None},
        wandb_run=wandb_run,
    )
    if wandb_run is not None:
        wandb_run.summary.update({"best_epoch": record["best_epoch"]})
        wandb_run.finish()
    out_volume.commit()
    if model_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        rel = "/".join(Path(run_path).parts[1:])
        for path in [out / n for n in PUSH_FILES] + sorted(out.glob("train_val_*.png")):
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{rel}/{path.name}",
                repo_id=model_repo,
                commit_message=f"{rel}: {path.name}",
            )
        print(f"pushed {run_path} to https://huggingface.co/{model_repo}")
    return run_path


@app.function(image=image, timeout=9 * 3600)
def orchestrate(runs: list, model_repo: str | None) -> list:
    """Spawn one training container per run and wait for all. With --detach, Modal keeps only
    the last function triggered by the local process alive if it disconnects, so the runs are
    spawned from here."""
    calls = []
    for cfg, data, run_path, seed, provenance in runs:
        calls.append(train.spawn(cfg, data, run_path, seed, provenance, model_repo))
        print(f"launched {run_path}", flush=True)
    return [call.get() for call in calls]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def _hub_path(p: str) -> str:
    """A collection or labels folder as named in the dataset repository (locally under data/)."""
    parts = Path(p).parts
    return str(Path(*parts[1:])) if parts[:1] == ("data",) else str(p)


@app.local_entrypoint()
def main(
    config: str = "configs/act_deterministic.toml",
    num_episodes: str = "",
    seeds: str = "0",
    smoke: bool = False,
    no_push: bool = False,
    model_repo: str = "",
    wandb_entity: str = "",
):
    cfg = tomllib.loads(Path(config).read_text())
    hub = tomllib.loads(Path("configs/data.toml").read_text())["hub"]
    stem = Path(config).stem
    sizes = [int(n) for n in num_episodes.split(",")] if num_episodes else [cfg["num_episodes"]]
    if smoke:
        cfg.update(num_epochs=2)
        cfg.pop("wandb_project", None)
        sizes, no_push = [10], True
    if wandb_entity:
        cfg["wandb_entity"] = wandb_entity
    run_root = stem + ("-smoke" if smoke else "")
    repo = None if no_push else (model_repo or f"Autobrik/ecse552-{stem}".replace("_", "-"))
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if repo and dirty:
        raise SystemExit(f"Commit your changes before a run that publishes:\n{dirty}")

    source = _hub_path(cfg["data_dir"])

    def targets_for(seed: int) -> list[dict]:
        """The config's targets; `{seed}` in a labels folder is the training seed."""
        return [
            {
                "folder": _hub_path(t["dir"]),
                "num_episodes": 10 if smoke else int(t["num_episodes"]),
                "actions": "true"
                if t.get("actions", "true") == "true"
                else _hub_path(t["actions"]).format(seed=seed),
                "exclude_like": _hub_path(t["exclude_like"]).format(seed=seed)
                if t.get("exclude_like")
                else None,
                "max_label_step": t.get("max_label_step"),
            }
            for t in cfg.get("targets", [])
        ]

    provenance = {
        "project_commit": _git("rev-parse", "HEAD"),
        "project_dirty": bool(dirty),
        "ocbench_commit": _git("-C", "third_party/ocbench", "rev-parse", "HEAD"),
        "act_commit": _git("-C", "third_party/act", "rev-parse", "HEAD"),
        "config_file": config,
        "dataset": hub["dataset"],
        "dataset_revision": hub["revision"],
        "dataset_folder": source,
        "gpu": GPU,
        "model_repo": repo,
    }
    card = MODEL_CARD.format(
        repo=repo, dataset=hub["dataset"], revision=hub["revision"], config=config
    )
    exclude, state = cfg.get("exclude_detours", False), cfg.get("use_robot_state", False)
    runs, exported = [], {}
    for s in (int(x) for x in seeds.split(",")):
        targets = targets_for(s)
        key = str(targets)
        if key not in exported:
            exported[key] = prepare.remote(
                hub["dataset"],
                hub["revision"],
                source,
                sizes,
                targets,
                exclude,
                state,
                repo,
                card,
                smoke,
            )
        for n in sizes:
            run = (cfg, exported[key][n], f"{run_root}/n{n}/seed{s}", s)
            runs.append((*run, {**provenance, "targets": targets}))
    for run_path in orchestrate.remote(runs, repo):
        print("finished:", run_path)
