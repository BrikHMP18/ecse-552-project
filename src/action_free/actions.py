"""Action representations: OCBench joint deltas and absolute joint targets.

OCBench actions are seven values in [-1, 1]: six arm-joint increments and one gripper-opening
increment, each added to the *current* measured position (`ManipulationEnv._set_joint_delta_
control`): target = q_now + action * scale. ACT reports better results with absolute target
joint positions (Zhao et al., 2023), and with open-loop chunks deltas accumulate execution
errors. So policies can also be trained on absolute targets:

    arm target     = q_t[0:6] + action[0:6] * ARM_SCALE
    gripper target = clip(opening_t + action[6] * GRIPPER_SCALE, 0, 1),  opening = q_t[6] / 0.8

computed from the stored simulator state (labels only; the policy still sees only images).
At execution, the low-level controller turns each absolute target back into the delta the
environment expects, using the robot's current joint positions (`to_delta`).
"""

from __future__ import annotations

import numpy as np

# OCBench `_joint_action_delta` for the lite tasks (action_delta_scale = 5); checked in tests.
ARM_SCALE = np.array([0.9, 0.9, 0.9, 1.8, 1.8, 1.8], dtype=np.float64)
GRIPPER_SCALE = 0.6
ARM_QPOS = slice(0, 6)
GRIPPER_QPOS = 6  # ur5e/robotiq/right_driver_joint; opening = qpos / 0.8
GRIPPER_OPENING_DIVISOR = 0.8


def gripper_opening(qpos: np.ndarray) -> np.ndarray:
    return np.asarray(qpos)[..., GRIPPER_QPOS] / GRIPPER_OPENING_DIVISOR


def to_absolute(actions: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Absolute targets (T, 7) from delta actions (T, 7) and the state before each step (T, nq)."""
    actions = np.asarray(actions, dtype=np.float64)
    qpos = np.asarray(qpos, dtype=np.float64)
    arm = qpos[..., ARM_QPOS] + actions[..., :6] * ARM_SCALE
    grip = np.clip(gripper_opening(qpos) + actions[..., 6] * GRIPPER_SCALE, 0.0, 1.0)
    return np.concatenate([arm, grip[..., None]], axis=-1).astype(np.float32)


def to_delta(targets: np.ndarray, qpos_now: np.ndarray) -> np.ndarray:
    """Delta actions in [-1, 1] that move from the current state towards absolute targets."""
    targets = np.asarray(targets, dtype=np.float64)
    qpos_now = np.asarray(qpos_now, dtype=np.float64)
    arm = (targets[..., :6] - qpos_now[..., ARM_QPOS]) / ARM_SCALE
    grip = (targets[..., 6] - gripper_opening(qpos_now)) / GRIPPER_SCALE
    return np.clip(np.concatenate([arm, grip[..., None]], axis=-1), -1.0, 1.0).astype(np.float32)
