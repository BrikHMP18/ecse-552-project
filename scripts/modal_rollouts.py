"""Policy rollouts on Modal for iterative BCO (g, docs/experiments/bco.md): one T4 container
per training seed.

Steps (`--step`); paths take `{seed}`, `bco:` (volume `ecse552-bco`) and `out:` (volume
`ecse552-outputs`, where `modal_train.py` writes the policies):

- `fetch`: copy files of a public Hugging Face model repo into the volume, e.g. the (b)
  policies, round 0. No token is needed.
- `collect`: N rollouts of a policy on the target from the round's seeds (800,000+); every
  episode is kept, with the actions the policy sent and the robot's joint positions.
- `select`: best or last checkpoint of an ACT run on 100 source validation rollouts.
- `test`: 100 rollouts of a selected checkpoint on the test seeds, once (refuses to repeat).

    modal run scripts/modal_rollouts.py --step fetch --repo Autobrik/ecse552-act-idm-D \
        --prefix n300/seed{seed} --files selected.ckpt,act_run.json,dataset_stats.pkl \
        --out bco:policies/act_idm_D/seed{seed}
    modal run scripts/modal_rollouts.py --step collect --round 1 \
        --policy bco:policies/act_idm_D/seed{seed}/selected.ckpt --out bco:rollouts/D-r1-seed{seed}
    modal run scripts/modal_rollouts.py --step select --policy out:act_bco_D_r1/n300/seed{seed}
    modal run scripts/modal_rollouts.py --step test --object D \
        --policy out:act_bco_D_r1/n300/seed{seed}/selected.ckpt

--smoke runs 2 episodes per step under the volume's `smoke/` folder (test results under
`smoke_test_<object>/`). No secrets are needed.
"""

from pathlib import Path

import modal

GPU = "T4"  # rollouts are simulator-bound; bco.md: T4, training stays on A10G
app = modal.App("ecse552-rollouts")
bco_volume = modal.Volume.from_name("ecse552-bco", create_if_missing=True)
act_volume = modal.Volume.from_name("ecse552-outputs", create_if_missing=True)
VOLUMES = {"/bco": bco_volume, "/outputs": act_volume}
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
        }
    )
    .add_local_dir("third_party/act", "/root/third_party/act", ignore=[".git"])
    .add_local_dir(
        "third_party/ocbench", "/root/third_party/ocbench", ignore=[".git", "**/__pycache__"]
    )
    .add_local_dir("src/action_free", "/root/action_free")
)


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def fetch(repo: str, prefix: str, files: list[str], out: str, seed: int, smoke: bool) -> list[str]:
    """Copy `prefix/<file>` of a public model repo into the volume folder `out`."""
    import shutil

    from huggingface_hub import hf_hub_download

    from action_free.bco import resolve

    dest = Path(resolve(out, seed, smoke))
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        path = f"{prefix.format(seed=seed)}/{f}" if prefix else f
        shutil.copyfile(hf_hub_download(repo, path), dest / f)
    bco_volume.commit()
    return sorted(p.name for p in dest.iterdir())


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
    smoke: bool = False,
) -> None:
    calls = {}
    for s in (int(x) for x in seeds.split(",")):
        if step == "fetch":
            calls[s] = fetch.spawn(repo, prefix, files.split(","), out, s, smoke)
        elif step == "collect":
            calls[s] = collect.spawn(policy, object, round, out, s, smoke)
        elif step == "select":
            calls[s] = select.spawn(policy, s, smoke)
        elif step == "test":
            calls[s] = test.spawn(policy, object, s, smoke)
        else:
            raise SystemExit(f"unknown step {step!r}: fetch, collect, select or test")
        print(f"spawned {step} seed {s} ({calls[s].object_id})")
    for s, call in calls.items():
        print(f"seed {s}: {call.get()}")
