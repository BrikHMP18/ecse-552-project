"""Evaluate a policy checkpoint in closed loop from the shared evaluation initial states.

    MUJOCO_GL=egl uv run python scripts/evaluate_policy.py <run>/selected.ckpt --object source
    MUJOCO_GL=egl uv run python scripts/evaluate_policy.py <run>/selected.ckpt --object A \
        --video-episodes 5

Test seeds (default) give the final numbers; evaluate each selected checkpoint on them once.
Writes results.json (success, lengths, latency, distribution checks) and optional MP4s next
to the checkpoint, under <seeds>_<object>/, unless --out-dir is given.
"""

import argparse
from pathlib import Path

from action_free.evaluate import (
    DEFAULT_EXECUTION,
    TEST_BASE_SEED,
    VALIDATION_BASE_SEED,
    EvalConfig,
    evaluate,
)
from action_free.objects import OBJECTS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--object", default="source", choices=sorted(OBJECTS))
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument(
        "--seeds",
        choices=("test", "validation"),
        default="test",
        help="test: final numbers, used once per selected checkpoint; validation: decisions",
    )
    p.add_argument("--execution", default=DEFAULT_EXECUTION, help="chunk, replan:N or ensemble:M")
    p.add_argument("--scene", default=None, help="Default: the scene the policy was trained in")
    p.add_argument("--video-episodes", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args()

    base_seed = TEST_BASE_SEED if a.seeds == "test" else VALIDATION_BASE_SEED
    suffix = "" if a.execution == DEFAULT_EXECUTION else "_" + a.execution.replace(":", "")
    suffix += f"_{a.scene}" if a.scene else ""
    out = a.out_dir or a.checkpoint.parent / f"{a.seeds}_{a.object}{suffix}"
    s = evaluate(
        EvalConfig(
            checkpoint=str(a.checkpoint),
            object_name=a.object,
            num_episodes=a.num_episodes,
            base_seed=base_seed,
            video_episodes=a.video_episodes,
            execution=a.execution,
            scene=a.scene,
            out_dir=str(out),
        )
    )
    lat = s["latency_ms"]
    print(
        f"\n{a.object}: success {s['success_rate']:.1%} over {s['num_episodes']} episodes "
        f"(epoch {s['checkpoint_epoch']}); latency mean {lat['mean']:.1f} ms, "
        f"p95 {lat['p95']:.1f} ms, max {lat['max']:.1f} ms (R6: < 100 ms) -> {out}"
    )


if __name__ == "__main__":
    main()
