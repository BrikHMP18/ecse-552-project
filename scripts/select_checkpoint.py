"""Pick each ACT run's checkpoint (policy_best or policy_last) with source validation rollouts.

    MUJOCO_GL=egl uv run python scripts/select_checkpoint.py outputs/act_deterministic/n300/seed0
    MUJOCO_GL=egl uv run python scripts/select_checkpoint.py outputs/act_deterministic/n*/seed0 \
        --push Autobrik/ecse552-act-deterministic

Rolls out both checkpoints on 100 source validation seeds with ACT's temporal ensembling,
keeps the one with higher success (ties to policy_best.ckpt) and writes selection.json and
selected.ckpt in each run directory. With --push, uploads both to the run's folder of the
Hugging Face model repository (path relative to outputs/<config>/).
"""

import argparse
from pathlib import Path

from action_free.evaluate import DEFAULT_EXECUTION, select_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", type=Path, nargs="+", help="Run directories with policy_*.ckpt")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--execution", default=DEFAULT_EXECUTION)
    p.add_argument("--push", default=None, help="Hugging Face model repo to upload to")
    a = p.parse_args()
    for run in a.runs:
        s = select_checkpoint(run, a.episodes, a.execution)
        v = s["validation_success"]
        print(
            f"{run}: selected {s['selected']} (epoch {s['selected_epoch']}); validation success "
            f"best {v['best']:.1%}, last {v['last']:.1%}"
        )
        if a.push:
            from huggingface_hub import HfApi

            prefix = "/".join(run.parts[-2:])  # e.g. n300/seed0
            api = HfApi()
            for name in ("selection.json", "selected.ckpt"):
                api.upload_file(
                    path_or_fileobj=str(run / name),
                    path_in_repo=f"{prefix}/{name}",
                    repo_id=a.push,
                    commit_message=f"{prefix}: {name}",
                )


if __name__ == "__main__":
    main()
