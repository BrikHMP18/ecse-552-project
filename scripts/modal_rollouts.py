"""Policy rollouts on Modal for iterative BCO (g, docs/experiments/bco.md): one T4 container
per training seed.

Steps (`--step`); paths take `{seed}`, `bco:` (volume `ecse552-bco`), `out:` (volume
`ecse552-outputs`, where `modal_train.py` writes the policies) and `idm:` (volume
`ecse552-idm-outputs`, where `modal_train_idm.py` writes the IDMs):

- `fetch`: copy files (`--files`; without it, the whole `--prefix` folder) of a public
  Hugging Face repo (`--repo-type model` or `dataset`) into a volume, e.g. the (b) policies,
  round 0. No token is needed.
- `collect`: N rollouts of a policy on the target from the round's seeds (800,000+); every
  episode is kept, with the actions the policy sent and the robot's joint positions.
- `select`: best or last checkpoint of an ACT run on 100 source validation rollouts.
- `test`: 100 rollouts of a selected checkpoint on the test seeds, once (refuses to repeat).
- `publish`: upload volume folders (`--out`, comma-separated) to a public Hugging Face
  dataset (`--repo`): `bco:<p>` to `<p>`, `out:<p>` to `out/<p>`, `idm:<p>` to `idm/<p>`,
  only the `--files` patterns if given; needs the Modal secret `huggingface-secret`.

    modal run scripts/modal_rollouts.py --step fetch --repo Autobrik/ecse552-act-idm-D \
        --prefix n300/seed{seed} --files selected.ckpt,act_run.json,dataset_stats.pkl \
        --out bco:policies/act_idm_D/seed{seed}
    modal run scripts/modal_rollouts.py --step collect --round 1 \
        --policy bco:policies/act_idm_D/seed{seed}/selected.ckpt --out bco:rollouts/D-r1-seed{seed}
    modal run scripts/modal_rollouts.py --step select --policy out:act_bco_D_r1/n300/seed{seed}
    modal run scripts/modal_rollouts.py --step test --object D \
        --policy out:act_bco_D_r1/n300/seed{seed}/selected.ckpt

--smoke runs 2 episodes per step under the volume's `smoke/` folder (test results under
`smoke_test_<object>/`). Only `publish` uses a secret; without one, set
`ECSE552_MODAL_SECRETS=`.
"""

import os
from pathlib import Path

import modal

GPU = "T4"  # rollouts are simulator-bound; bco.md: T4, training stays on A10G
app = modal.App("ecse552-rollouts")
bco_volume = modal.Volume.from_name("ecse552-bco", create_if_missing=True)
act_volume = modal.Volume.from_name("ecse552-outputs", create_if_missing=True)
idm_volume = modal.Volume.from_name("ecse552-idm-outputs", create_if_missing=True)
VOLUMES = {"/bco": bco_volume, "/outputs": act_volume, "/idm": idm_volume}
SECRET_NAMES = os.environ.get("ECSE552_MODAL_SECRETS", "huggingface-secret")
HF_SECRET = "huggingface-secret"
HF_SECRETS = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET in SECRET_NAMES.split(",") else []
# MuJoCo renders through EGL; point glvnd at the NVIDIA driver Modal mounts.
NVIDIA_EGL = '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}'

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libegl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "numpy==2.2.6",
        "huggingface_hub==2.2.0",
        # upstream ACT's and OCBench's imports (versions from uv.lock)
        "h5py==3.16.0",
        "einops==0.8.2",
        "ipython==8.39.0",
        "opencv-python-headless==5.0.0.93",
        "matplotlib==3.11.2",
        "tqdm==4.70.1",
        "dm-control==1.0.48",
        "mujoco==3.15.0",
        "gymnasium==1.4.0",
        "imageio==2.38.1",
        "lxml==6.1.3",
    )
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d",
        f"echo '{NVIDIA_EGL}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .env(
        {
            "ACT_ROOT": "/root/third_party/act",
            "MUJOCO_GL": "egl",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": "/root/third_party/ocbench",
            "ECSE552_MODAL_SECRETS": SECRET_NAMES,  # the container attaches the same secrets
        }
    )
    .add_local_dir("third_party/act", "/root/third_party/act", ignore=[".git"])
    .add_local_dir(
        "third_party/ocbench", "/root/third_party/ocbench", ignore=[".git", "**/__pycache__"]
    )
    .add_local_dir("src/action_free", "/root/action_free")
)


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def fetch(
    repo: str, prefix: str, files: list[str], out: str, seed: int, smoke: bool, repo_type: str
) -> list[str]:
    """Copy `prefix/<file>` (every file of `prefix/` if no file is named) of a public Hugging
    Face repo into the volume folder `out`."""
    import shutil

    from huggingface_hub import hf_hub_download, snapshot_download

    from action_free.bco import resolve

    prefix = prefix.format(seed=seed)
    dest = Path(resolve(out, seed, smoke))
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in files if f]
    if files:
        for f in files:
            path = f"{prefix}/{f}" if prefix else f
            shutil.copyfile(hf_hub_download(repo, path, repo_type=repo_type), dest / f)
    else:
        local = snapshot_download(repo, repo_type=repo_type, allow_patterns=f"{prefix}/*")
        shutil.copytree(Path(local) / prefix, dest, dirs_exist_ok=True)
    for v in (bco_volume, act_volume, idm_volume):
        v.commit()
    names = sorted(p.name for p in dest.iterdir())
    return names if len(names) <= 5 else [*names[:3], f"... {len(names)} files"]


@app.function(image=image, volumes=VOLUMES, timeout=6 * 3600, secrets=HF_SECRETS)
def publish(repo: str, folders: list[str], files: list[str], seed: int) -> list[str]:
    """Upload volume folders to a public dataset repo: `bco:<p>` to `<p>`, `out:<p>` to
    `out/<p>`, `idm:<p>` to `idm/<p>`; only the `files` patterns if any are given."""
    from huggingface_hub import HfApi

    from action_free.bco import resolve

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    commits = []
    for f in folders:
        path = f.format(seed=seed)
        in_repo = path.removeprefix("bco:").replace(":", "/", 1)
        info = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=resolve(path, seed),
            path_in_repo=in_repo,
            allow_patterns=[p for p in files if p] or None,
            commit_message=f"BCO (g): {in_repo}",
        )
        commits.append(info.oid)
    return commits


@app.function(image=image, gpu=GPU, memory=16384, cpu=2, volumes=VOLUMES, timeout=6 * 3600)
def collect(policy: str, object_name: str, round_: int, out: str, seed: int, smoke: bool) -> dict:
    from action_free.bco import collect_rollouts, resolve, rollout_seeds
    from action_free.evaluate import DEFAULT_EXECUTION

    seeds = rollout_seeds(round_, 2 if smoke else None)
    checkpoint, dest = resolve(policy, seed, smoke), Path(resolve(out, seed, smoke))
    try:
        m = collect_rollouts(checkpoint, object_name, seeds, dest, DEFAULT_EXECUTION)
    finally:
        bco_volume.commit()
    return {k: m[k] for k in ("num_episodes", "success_rate", "total_transitions")}


@app.function(image=image, gpu=GPU, memory=16384, cpu=2, volumes=VOLUMES, timeout=4 * 3600)
def select(run: str, seed: int, smoke: bool) -> dict:
    from action_free.bco import resolve
    from action_free.evaluate import select_checkpoint

    try:
        return select_checkpoint(Path(resolve(run, seed, smoke)), 2 if smoke else 100)
    finally:
        act_volume.commit()


@app.function(image=image, gpu=GPU, memory=16384, cpu=2, volumes=VOLUMES, timeout=4 * 3600)
def test(policy: str, object_name: str, seed: int, smoke: bool) -> dict:
    from action_free.bco import resolve
    from action_free.evaluate import TEST_BASE_SEED, EvalConfig, evaluate

    checkpoint = resolve(policy, seed, smoke)
    out = Path(checkpoint).parent / f"{'smoke_' if smoke else ''}test_{object_name}"
    if (out / "results.json").exists():
        raise FileExistsError(f"{out}: test seeds are used once per selected checkpoint")
    n = 2 if smoke else 100
    cfg = EvalConfig(checkpoint, object_name, n, TEST_BASE_SEED, out_dir=str(out))
    try:
        s = evaluate(cfg, log=lambda *a, **k: None)
    finally:
        act_volume.commit()
    return {k: s[k] for k in ("success_rate", "num_episodes", "latency_ms")}


@app.local_entrypoint()
def main(
    step: str,
    seeds: str = "0,1,2",
    policy: str = "",
    object: str = "D",
    round: int = 1,
    out: str = "",
    repo: str = "",
    files: str = "",
    prefix: str = "",
    repo_type: str = "model",
    smoke: bool = False,
) -> None:
    calls = {}
    for s in (int(x) for x in seeds.split(",")):
        if step == "fetch":
            calls[s] = fetch.spawn(repo, prefix, files.split(","), out, s, smoke, repo_type)
        elif step == "collect":
            calls[s] = collect.spawn(policy, object, round, out, s, smoke)
        elif step == "select":
            calls[s] = select.spawn(policy, s, smoke)
        elif step == "test":
            calls[s] = test.spawn(policy, object, s, smoke)
        elif step == "publish":
            calls[s] = publish.spawn(repo, out.split(","), files.split(","), s)
        else:
            raise SystemExit(f"unknown step {step!r}: fetch, collect, select, test or publish")
        print(f"spawned {step} seed {s} ({calls[s].object_id})")
    for s, call in calls.items():
        print(f"seed {s}: {call.get()}")
