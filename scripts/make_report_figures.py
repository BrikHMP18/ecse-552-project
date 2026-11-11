"""Draw the report figures from the downloaded runs (outputs/) and the collections (data/).

    uv run --with matplotlib python scripts/make_report_figures.py

Writes docs/report/figures/{results,selection,curves,wrist}.png. Runs that are not
downloaded and tested yet are left out, so the figures fill in as results arrive.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/report/figures")
BLUE, ORANGE, AQUA, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#1f1f1e",
    "#6b6a64",
    "#e4e3dd",
)
OBJECTS = ("source", "A", "B", "C", "D", "E")
ORACLES = ("A", "B", "C", "D", "E")
METHODS = (  # label, colour, run directory per test object
    ("(a) Source only", BLUE, lambda o: "act_deterministic"),
    ("(e) Augmentation", ORANGE, lambda o: "act_augment"),
    ("(d) Oracle for that target", AQUA, lambda o: f"act_oracle_{o}"),
)
plt.rcParams.update(
    {"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
     "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight"}
)  # fmt: skip


def seeds(run: str) -> list[Path]:
    return sorted(Path("outputs", run, "n300").glob("seed[0-9]"))


def test(run: str, obj: str) -> list[float]:
    """Test success (%) of each seed of a run on an object; empty until every seed is tested."""
    files = [d / f"test_{obj}" / "results.json" for d in seeds(run)]
    if len(files) < 3 or not all(f.exists() for f in files):
        return []
    return [100 * json.loads(f.read_text())["success_rate"] for f in files]


def results() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    width = 0.27
    for m, (label, colour, run) in enumerate(METHODS):
        for i, obj in enumerate(OBJECTS):
            v = test(run(obj), obj) if obj != "source" or m < 2 else []
            if not v:
                continue
            x = i + (m - 1) * width
            ax.bar(x, np.mean(v), width - 0.03, color=colour, label=label if i == 1 else None)
            ax.errorbar(x, np.mean(v), np.std(v, ddof=1), color=INK, lw=0.8, capsize=1.5)
            ax.scatter([x] * 3, v, s=5, color="white", edgecolor=INK, lw=0.5, zorder=3)
            top = max(max(v), np.mean(v) + np.std(v, ddof=1))
            ax.text(x, top + 2, f"{np.mean(v):.0f}", ha="center", va="bottom", fontsize=6.5)
    names = ["Source\n(red cube)", "A\n(blue cube)", "B\n(red 9x6x6)", "C\n(red 9x7x6)",
             "D\n(blue 9x7x6)", "E\n(red 6x9x6)"]  # fmt: skip
    ax.set_xticks(range(len(OBJECTS)), names)
    ax.set_ylim(0, 116)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Test success (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(OUT / "results.png")


def selection() -> None:
    runs = [("(a)", "act_deterministic"), ("(e)", "act_augment")]
    runs += [(f"(d) {t}", f"act_oracle_{t}") for t in ORACLES]
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    x, ticks = 0, []
    for name, run in runs:
        start = x
        for d in seeds(run):
            f = d / "selection.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())["validation_success"]
            best, last = 100 * s["best"], 100 * s["last"]
            ax.plot([x, x], [best, last], color=GRID, lw=2, zorder=1)
            ax.scatter(
                x,
                best,
                s=16,
                color=BLUE,
                zorder=2,
                label="Lowest validation loss" if x == 0 else None,
            )
            ax.scatter(
                x,
                last,
                s=16,
                color=ORANGE,
                marker="s",
                zorder=2,
                label="Last epoch" if x == 0 else None,
            )
            x += 1
        if x > start:
            ticks.append(((start + x - 1) / 2, name))
            x += 1
    ax.set_xticks([t for t, _ in ticks], [n for _, n in ticks])
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Source validation\nsuccess (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(OUT / "selection.png")


def curves() -> None:
    panels = [("(a) Source only", "act_deterministic"), ("(e) Augmentation", "act_augment")]
    panels += [(f"(d) Oracle {t}", f"act_oracle_{t}") for t in ORACLES]
    panels = [(n, r) for n, r in panels if any((d / "history.json").exists() for d in seeds(r))]
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 2.9), sharex=True, sharey=True)
    for ax, (name, run) in zip(axes.flat, panels):
        for d in seeds(run):
            if not (d / "history.json").exists():
                continue
            h = json.loads((d / "history.json").read_text())
            epoch = np.array([r["epoch"] for r in h])
            ax.plot(epoch, [r["train/l1"] for r in h], color=BLUE, lw=0.6)
            ax.plot(epoch, [r["val/l1"] for r in h], color=ORANGE, lw=0.6)
        ax.set_title(name, fontsize=7.5)
        ax.set_yscale("log")
        ax.set_yticks([0.03, 0.1, 0.3], ["0.03", "0.1", "0.3"])
        ax.minorticks_off()
        ax.grid(color=GRID, lw=0.5)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    axes[0, 0].plot([], [], color=BLUE, label="Training")
    axes[0, 0].plot([], [], color=ORANGE, label="Validation")
    fig.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.supxlabel("Epoch", fontsize=8)
    fig.supylabel("L1 action loss (log scale)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png")


def wrist() -> None:
    sets = [("Source, 300 demonstrations", "source-deterministic-n300-seed0", BLUE),
            ("Source with turned grasp, 100 demonstrations (IDM only)",
             "source-turned-n100-seed600000", ORANGE),
            ("Target E, 200 videos", "E-deterministic-n200-seed500000", AQUA)]  # fmt: skip
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 2.8), sharex=True)
    bins = np.arange(-6.4, 3.41, 0.2)
    for ax, (name, folder, colour) in zip(axes, sets):
        w = []
        for f in sorted(Path("data", folder).glob("episode_*.npz")):
            ep = np.load(f)
            w.append(ep["qpos"][: int(ep["length"]), 5])
        ax.hist(
            np.concatenate(w),
            bins=bins,
            color=colour,
            weights=np.full(sum(map(len, w)), 100 / sum(map(len, w))),
        )
        ax.set_title(name, fontsize=7.5, loc="left", pad=2)
        ax.axvline(-1.2, color=MUTED, lw=0.6, ls="--")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[1].set_ylabel("Frames (%)")
    axes[2].set_xlabel("Last wrist joint (rad)")
    fig.tight_layout()
    fig.savefig(OUT / "wrist.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (results, selection, curves, wrist):
        draw()
        plt.close("all")
