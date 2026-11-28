"""Trace one shared target sample through both labelers (verification,
docs/experiments/lapa.md, notes/plan-lapa.md).

    uv run python scripts/trace_labelers.py \
        data/A-deterministic-n200-seed100000/episode_00100.npz \
        labels/A-deterministic-idm_turned-seed0 \
        outputs/laq/seed0/best.pt outputs/lapa_g/seed0/best.pt --step 42

Shows the episode id and the step, what each labeler sees, what it predicts, and exactly
where the prediction lands in an ACT training chunk. The IDM labels the pair
(o_t, o_{t+1}) directly: a_idm. The LAPA labeler labels the same pair with the LAQ's
4-token code z_t and decodes it: a_lapa = g(o_t, z_t). Both are episode_XXXXX.npy[t],
which act_original.export_dataset writes into the episode's /action dataset; ACT samples a
chunk [t, t + chunk) from there, so the printed chunk row is exactly the training target
at this step. Diagnostics only: reads no target state (R3); true actions are not read.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from action_free.act_original import ACTION_DIM, CAMERAS
from action_free.idm import ACTION_NAMES, load_frames, select_device
from action_free.lapa import load_g, load_laq, load_pairs, z_codes

CHUNK = 20  # ACT's chunk size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("episode", type=Path)
    p.add_argument("idm_labels", type=Path, help="A published IDM pseudo-label folder")
    p.add_argument("laq_checkpoint", type=Path)
    p.add_argument("g_checkpoint", type=Path)
    p.add_argument("--step", type=int, default=None, help="Transition index; default: middle")
    p.add_argument("--size", type=int, default=128, help="Both labelers' image size")
    a = p.parse_args()
    device = select_device()
    frames = load_frames(a.episode, a.size)  # (T + 1, cams, 3, S, S) uint8
    t = a.step if a.step is not None else (len(frames) - 1) // 2
    if not 0 <= t < len(frames) - 1:
        raise SystemExit(f"--step must be in [0, {len(frames) - 2}]")

    # The IDM label of the transition (published folder, read as-is).
    idm = np.load(a.idm_labels / f"{a.episode.stem}.npy")  # (T, 7)
    a_idm = idm[t]

    # The LAPA label of the same transition: z_t = LAQ(o_t, o_{t+1}), a_t = g(o_t, z_t),
    # computed for the whole episode exactly as label_collection does.
    laq = load_laq(a.laq_checkpoint, device)
    decoder = load_g(a.g_checkpoint, device)
    pairs = load_pairs([a.episode], a.size, 1, labeled=False)
    z = z_codes(laq, pairs, device)
    with torch.no_grad():
        lapa = (
            decoder.predict(pairs.frames[pairs.index].to(device), z.to(device)).cpu().numpy()
        )  # (T, 7)
    a_lapa = lapa[t]
    z_t = z[t]

    print(
        f"episode {a.episode.stem} (seed {np.load(a.episode)['seed']}), "
        f"transition t = {t} of {len(frames) - 1} (10 Hz: 0.1 s per step)"
    )
    print(
        f"both labelers see the same pair: o_t and o_{{t+1}}, {len(CAMERAS)} cameras "
        f"x 3 colours at {a.size} px"
    )
    print(
        f"LAQ sees (x_t, x_{{t+H}}) with H = {laq.config['stride']} at training; this label "
        f"uses H = 1, the paper's labeling procedure"
    )
    print(f"z_t (4 tokens, each < {laq.config['codebook']}): {z_t.tolist()}")
    print(f"{'component':<8} {'IDM a_t':>10} {'LAPA a_t':>10} {'|IDM - LAPA|':>12}")
    for name, i, u in zip(ACTION_NAMES, a_idm, a_lapa):
        print(f"{name:<8} {i:10.4f} {u:10.4f} {abs(i - u):12.4f}")
    print("units: arm joints in rad (absolute targets), gripper opening in [0, 1]")

    # Where the label lands in ACT training: episode_XXXXX.npy[t] is row t of the
    # exported episode's /action dataset, and a chunk starting at t is rows [t, t + 20).
    assert idm.shape == lapa.shape == (len(frames) - 1, ACTION_DIM)
    assert np.allclose(lapa[t], a_lapa)
    print(f"npy rows: IDM {idm.shape}, LAPA {lapa.shape}; both write {a.episode.stem}.npy[t]")
    chunk = np.pad(lapa[t : t + CHUNK], ((0, max(0, CHUNK - len(lapa) + t)), (0, 0)))
    print(
        f"ACT chunk starting at t (padded to {CHUNK}): rows {t}..{min(t + CHUNK, len(lapa)) - 1}"
        f" of the exported /action dataset; this transition is chunk position 0:"
    )
    print(f"chunk position 0 = a_t = {chunk[0].round(4)}")


if __name__ == "__main__":
    main()
