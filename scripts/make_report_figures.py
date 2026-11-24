"""Draw the report figures from the downloaded runs (outputs/) and the collections (data/).

    uv run --with matplotlib python scripts/make_report_figures.py

Writes docs/report/figures/{results,selection,curves,wrist,labels,idm_input,dann,idm_val,
act_val,act_selection,dann_curves}.png. Runs that are not downloaded and tested yet are left
out, so the figures fill in as results arrive.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path("docs/report/figures")
BLUE, ORANGE, PURPLE, MAGENTA, AQUA, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#7c4dbd",
    "#e87ba4",
    "#1baf7a",
    "#1f1f1e",
    "#6b6a64",
    "#e4e3dd",
)
OBJECTS = ("source", "A", "B", "C", "D", "E")
ORACLES = ("A", "B", "C", "D", "E")
IDM_TARGETS = ("A", "D", "E")  # (b) is not run on B and C: no usable gap
METHODS = (  # label, colour, run directory per test object (None: not run)
    ("(a) Source only", BLUE, lambda o: "act_deterministic"),
    ("(e) Augmentation", ORANGE, lambda o: "act_augment"),
    ("(b) Unadapted IDM for that target", PURPLE, lambda o: f"act_idm_{o}"),
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
    width = 0.21
    for m, (label, colour, run) in enumerate(METHODS):
        for i, obj in enumerate(OBJECTS):
            v = test(run(obj), obj) if obj != "source" or m < 2 else []
            if not v:
                continue
            x = i + (m - 1.5) * width
            ax.bar(x, np.mean(v), width - 0.03, color=colour, label=label if i == 1 else None)
            ax.errorbar(x, np.mean(v), np.std(v, ddof=1), color=INK, lw=0.8, capsize=1.5)
            ax.scatter([x] * 3, v, s=5, color="white", edgecolor=INK, lw=0.5, zorder=3)
            top = max(max(v), np.mean(v) + np.std(v, ddof=1))
            ax.text(x, top + 2, f"{np.mean(v):.0f}", ha="center", va="bottom", fontsize=6)
    names = ["Source\n(red cube)", "A\n(blue cube)", "B\n(red 9x6x6)", "C\n(red 9x7x6)",
             "D\n(blue 9x7x6)", "E\n(red 6x9x6)"]  # fmt: skip
    ax.set_xticks(range(len(OBJECTS)), names)
    ax.set_ylim(0, 116)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Test success (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(OUT / "results.png")


def selection() -> None:
    runs = [("(a)", "act_deterministic"), ("(e)", "act_augment")]
    runs += [(f"(d) {t}", f"act_oracle_{t}") for t in ORACLES]
    runs += [(f"(b) {t}", f"act_idm_{t}") for t in IDM_TARGETS]
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
    panels += [(f"(b) IDM {t}", f"act_idm_{t}") for t in IDM_TARGETS]
    panels = [(n, r) for n, r in panels if any((d / "history.json").exists() for d in seeds(r))]
    fig, axes = plt.subplots(3, 4, figsize=(6.6, 4.2), sharex=True, sharey=True)
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
    for ax in axes.flat[max(0, len(panels) - axes.shape[1]) : len(panels)]:
        ax.tick_params(labelbottom=True)  # last panel of each column
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


def labels() -> None:
    """IDM labels of the last wrist joint on E against the true targets (a diagnostic, R1):
    the first episode the label filter drops, and the largest step per episode."""
    from action_free.act_original import plausible
    from action_free.idm import absolute_targets, split_files

    lab = Path("data/labels/E-deterministic-idm_turned-seed0")
    files = split_files(Path("data/E-deterministic-n200-seed500000"))
    kept = set(plausible(files, lab, 2.0))
    f = next(f for f in files if f not in kept)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.1), width_ratios=(1.4, 1))
    ax.plot(absolute_targets(f)[:, 5], color=INK, lw=1, label="True target")
    ax.plot(np.load(lab / f"{f.stem}.npy")[:, 5], color=PURPLE, lw=1, label="IDM label")
    ax.set_xlabel(f"Step (E, {f.stem.replace('_', ' ')})")
    ax.set_ylabel("Last wrist joint (rad)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(color=GRID, lw=0.5)
    step = lambda a: np.abs(np.diff(a[:, :6], axis=0)).max()  # noqa: E731
    src = [step(absolute_targets(f)) for f in split_files(Path("data", SOURCE))]
    tgt = [step(np.load(lab / f"{f.stem}.npy")) for f in files]
    bins = np.arange(0, 7.6, 0.25)
    bx.hist(src, bins=bins, color=BLUE, alpha=0.8, label="Source demos, true targets")
    bx.hist(tgt, bins=bins, color=PURPLE, alpha=0.6, label="E, IDM labels")
    bx.axvline(2.0, color=MUTED, lw=0.8, ls="--")
    bx.set_yscale("log")
    bx.set_xlabel("Largest step per episode (rad)")
    bx.set_ylabel("Episodes")
    bx.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "labels.png")


def idm_input() -> None:
    """What the IDM sees: two consecutive frames of a source demo and their difference."""
    ep = np.load(Path("data", SOURCE, "episode_00000.npz"))
    t, camera = 20, 1  # approach, side camera
    a, b = ep["observations"][t, camera], ep["observations"][t + 1, camera]
    diff = np.abs(b.astype(int) - a.astype(int)).mean(2)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    for ax, image, title in zip(axes, (a, b, diff), ("$o_t$", "$o_{t+1}$", "$|o_{t+1} - o_t|$")):
        ax.imshow(image, cmap="Greys", vmin=0, vmax=30) if image.ndim == 2 else ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.savefig(OUT / "idm_input.png")


def dann() -> None:
    """Label error on each target of the unadapted IDM (b), DANN (c) and its control, seed 0,
    and the videos the 2 rad label filter drops (true actions for evaluation only, R1)."""
    from action_free.act_original import plausible
    from action_free.idm import split_files

    folders = {"A": "A-deterministic-n200-seed100000", "D": "D-deterministic-n200-seed400000",
               "E": "E-deterministic-n200-seed500000"}  # fmt: skip
    idms = (("(b) Unadapted IDM", PURPLE, "idm_turned"),
            ("Control, $\\lambda = 0$", MUTED, "idm_dann_{}_lambda0"),
            ("(c) DANN", MAGENTA, "idm_dann_{}"))  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    width = 0.27
    for i, t in enumerate(IDM_TARGETS):
        files = split_files(Path("data", folders[t]))
        for m, (label, colour, idm) in enumerate(idms):
            run = Path("outputs", idm.format(t), "seed0", "label_error.json")
            if not run.exists():
                continue
            error = json.loads(run.read_text())[folders[t]]["smae"]
            labels_dir = Path("data/labels", f"{t}-deterministic-{idm.format(t)}-seed0")
            dropped = len(files) - len(plausible(files, labels_dir, 2.0))
            x = i + (m - 1) * width
            for ax, v, text in ((axes[0], error, f"{error:.2f}"), (axes[1], dropped, dropped)):
                ax.bar(x, v, width - 0.04, color=colour)
                ax.text(x, v, text, ha="center", va="bottom", fontsize=6)
    for ax, ylabel in zip(axes, ("Label error (sMAE)", "Videos dropped (of 200)")):
        ax.set_xticks(range(len(IDM_TARGETS)), IDM_TARGETS)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    handles = [Patch(color=colour, label=label) for label, colour, _ in idms]
    fig.legend(handles=handles, frameon=False, ncols=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.08))  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "dann.png")


def idm_history(run: str) -> list[dict]:
    f = Path("outputs", run, "history.json")
    return json.loads(f.read_text()) if f.exists() else []


def idm_val() -> None:
    """Source validation error of the unadapted IDM per epoch, its three seeds."""
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    for seed in range(3):
        h = idm_history(f"idm_turned/seed{seed}")
        ax.plot([r["epoch"] for r in h], [r["val"]["smae"] for r in h], color=PURPLE, lw=1)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation sMAE\n(source)")
    ax.set_ylim(0, None)
    ax.grid(color=GRID, lw=0.5)
    fig.savefig(OUT / "idm_val.png")


def act_val() -> None:
    """ACT validation L1 loss per epoch, seed 0 of each method (on E for the target methods):
    each epoch faint, the 50-epoch moving average solid."""
    runs = (("(a) Source only", BLUE, "act_deterministic"),
            ("(e) Augmentation", ORANGE, "act_augment"),
            ("(b) IDM, E", PURPLE, "act_idm_E"),
            ("(d) Oracle, E", AQUA, "act_oracle_E"))  # fmt: skip
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    for label, colour, run in runs:
        h = json.loads(Path("outputs", run, "n300/seed0/history.json").read_text())
        epoch, val = np.array([r["epoch"] for r in h]), np.array([r["val/l1"] for r in h])
        mean = np.convolve(val, np.ones(50) / 50, mode="valid")  # 50-epoch moving average
        ax.plot(epoch, val, color=colour, lw=0.4, alpha=0.15)
        ax.plot(epoch[49:], mean, color=colour, lw=1.2, label=label)
    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2], ["0.05", "0.1", "0.2"])
    ax.minorticks_off()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation L1 loss")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(OUT / "act_val.png")


def act_selection() -> None:
    """Validation L1 loss against source validation success of the two checkpoints of every
    run: the lowest validation loss (best) and the last epoch."""
    families = [("(a)", BLUE, ["act_deterministic"]), ("(e)", ORANGE, ["act_augment"]),
                ("(b)", PURPLE, [f"act_idm_{t}" for t in IDM_TARGETS]),
                ("(d)", AQUA, [f"act_oracle_{t}" for t in ORACLES])]  # fmt: skip
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for name, colour, runs in families:
        for d in (d for run in runs for d in seeds(run)):
            h, sel = d / "history.json", d / "selection.json"
            if not (h.exists() and sel.exists()):
                continue
            val = [r["val/l1"] for r in json.loads(h.read_text())]
            s = json.loads(sel.read_text())
            x = [val[s["epochs"]["best"]], val[s["epochs"]["last"]]]
            y = [100 * s["validation_success"]["best"], 100 * s["validation_success"]["last"]]
            ax.plot(x, y, color=GRID, lw=1, zorder=1)
            ax.scatter(x[0], y[0], s=14, color=colour, zorder=2)
            ax.scatter(x[1], y[1], s=14, color=colour, marker="s", zorder=2)
        ax.scatter([], [], s=14, color=colour, label=name)
    ax.scatter([], [], s=14, color=MUTED, label="lowest validation loss")
    ax.scatter([], [], s=14, color=MUTED, marker="s", label="last epoch")
    ax.set_xlabel("Validation L1 loss of the checkpoint")
    ax.set_ylabel("Source validation\nsuccess (%)")
    ax.set_ylim(0, 105)
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=6, ncols=2, loc="lower left")
    fig.savefig(OUT / "act_selection.png")


def dann_curves() -> None:
    """DANN training, seed 0: source validation error per epoch against the unadapted IDM, and
    the discriminator's loss against the control without reversed gradient (ln 2: chance)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for seed in range(3):
        h = idm_history(f"idm_turned/seed{seed}")
        ax.plot([r["epoch"] for r in h], [r["val"]["smae"] for r in h], color=PURPLE, lw=1,
                label="(b) Unadapted IDM" if seed == 0 else None)  # fmt: skip
    for i, t in enumerate(IDM_TARGETS):
        h = idm_history(f"idm_dann_{t}/seed0")
        epochs = [r["epoch"] for r in h]
        ax.plot(epochs, [r["val"]["smae"] for r in h], color=MAGENTA, lw=1,
                label="(c) DANN, A, D, E" if i == 0 else None)  # fmt: skip
        bx.plot(epochs, [r["domain_loss"] for r in h], color=MAGENTA, lw=1,
                label="(c) DANN, A, D, E" if i == 0 else None)  # fmt: skip
    h = idm_history("idm_dann_D_lambda0/seed0")
    for axis, key in ((ax, lambda r: r["val"]["smae"]), (bx, lambda r: r["domain_loss"])):
        axis.plot([r["epoch"] for r in h], [key(r) for r in h], color=MUTED, lw=1,
                  label="Control, $\\lambda = 0$, D")  # fmt: skip
    bx.axhline(np.log(2), color=MUTED, lw=0.6, ls="--")
    bx.text(20, np.log(2), "chance (ln 2)", ha="right", va="bottom", fontsize=6, color=MUTED)
    ax.set_ylabel("Validation sMAE (source)")
    bx.set_ylabel("Discriminator loss")
    for axis in (ax, bx):
        axis.set_xlabel("Epoch")
        axis.set_ylim(0, None)
        axis.grid(color=GRID, lw=0.5)
        axis.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "dann_curves.png")


SOURCE = "source-deterministic-n300-seed0"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (
        results,
        selection,
        curves,
        wrist,
        labels,
        idm_input,
        dann,
        idm_val,
        act_val,
        act_selection,
        dann_curves,
    ):
        draw()
        plt.close("all")
