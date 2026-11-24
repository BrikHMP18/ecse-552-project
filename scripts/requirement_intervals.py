"""95% bootstrap intervals for R4 (G_T) and R5 (drop in source success) from test rollouts.

    uv run python scripts/requirement_intervals.py

Each replicate resamples, independently per run, its training seeds with replacement and then
the 100 test rollouts of each drawn seed, so the interval covers both seed and rollout
variation. Point estimates are the means over seeds, as in the report.
"""

import json
from pathlib import Path

import numpy as np

RUNS = Path("outputs")
BASELINE = "act_deterministic"
METHODS = {"(e)": lambda t: "act_augment", "(b)": "act_idm_{}".format}
TARGETS = ("A", "D", "E")
REPLICATES = 10_000


def outcomes(run: str, obj: str) -> np.ndarray:
    """Successes (seeds, rollouts) of a run's test rollouts on an object."""
    files = sorted((RUNS / run / "n300").glob(f"seed*/test_{obj}/results.json"))
    return np.array([[e["success"] for e in json.loads(f.read_text())["episodes"]] for f in files])


def bootstrap(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mean success of each replicate: seeds drawn with replacement, then rollouts."""
    seeds = rng.integers(0, len(x), (REPLICATES, len(x), 1))
    rollouts = rng.integers(0, x.shape[1], (REPLICATES, len(x), x.shape[1]))
    return x[seeds, rollouts].mean((1, 2))


def interval(x: np.ndarray, digits: int) -> str:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def main() -> None:
    rng = np.random.default_rng(0)
    base_source = outcomes(BASELINE, "source")
    print(f"{'target':<7} {'method':<7} {'G_T':>6} {'interval':>15} {'drop':>7} {'interval':>17}")
    for t in TARGETS:
        base, oracle = outcomes(BASELINE, t), outcomes(f"act_oracle_{t}", t)
        a, d = bootstrap(base, rng), bootstrap(oracle, rng)
        for name, run in METHODS.items():
            m, m_source = outcomes(run(t), t), outcomes(run(t), "source")
            gain = (m.mean() - base.mean()) / (oracle.mean() - base.mean())
            gains = (bootstrap(m, rng) - a) / (d - a)
            drop = 100 * (base_source.mean() - m_source.mean())
            drops = 100 * (bootstrap(base_source, rng) - bootstrap(m_source, rng))
            print(
                f"{t:<7} {name:<7} {gain:6.2f} {interval(gains, 2):>15} "
                f"{drop:7.1f} {interval(drops, 1):>17}"
            )


if __name__ == "__main__":
    main()
