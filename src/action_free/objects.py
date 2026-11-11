"""Source and target objects from the proposal (Aims/Goals, Scope).

Each object maps to keyword arguments for the OCBench block environment, so collection,
training, and evaluation build the same scene from one name. Robots, cameras, lighting,
and the task stay fixed; only the manipulated object changes.

Colours follow the proposal: a red source cube and a blue cube for target A. Stock OCBench
renders the single cube blue, so the source sets `cube_colors=('red',)` explicitly
(supported by our OCBench fork, CPU environments only). Targets B, C and D are 9 cm blocks
(`object_size`, same fork); the expert grasps them across their second side (6 or 7 cm),
and the 7 cm ones 2 cm above the centre, where the gripper linkage clears the block.
Target E (11 November 2026) is B's block lying across the gripper: its 9 cm side is along y
and does not fit the 8.5 cm opening, so the expert turns the grasp by 90 degrees. It changes
the behaviour, not the appearance.

Every object shares `SCENE_KWARGS`: a dark-gray 12 cm (inner side) square marks the fixed
goal, and success requires the whole cube inside it, resting on the table, released and
still for 1 s. With 12 cm, 97% of resting expert placements fit fully inside; 8 cm fits
only 60% (measured over 200 episodes, 8 November 2026).

Initial states are narrower than stock OCBench (decided 8 November 2026): the cube starts in
a 20 x 20 cm area beside the goal instead of anywhere on a 42 x 110 cm area, and the arm
always starts from one pose instead of a random one, as in ACT's simulated tasks. The
project data (10 November 2026) also fixes the cube yaw at 0 (scene `fixed_yaw`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Shared by every object so that only the manipulated object changes between domains.
SCENE_KWARGS: dict = {
    "goal_marker_size": 0.12,
    "success_hold_time": 1.0,
    # Cubes start in a 20 x 20 cm area beside the goal (position and yaw uniform inside).
    "object_sampling_bounds": ((0.32, 0.15), (0.52, 0.35)),
    # Every episode starts from the same arm pose: effector (x, y, z) in m and yaw, pointing down.
    "arm_init_pose": (0.35, 0.0, 0.30, 0.0),
}
# Scenes. The project uses "fixed_yaw" (every cube at yaw 0, like ACT's simulated tasks);
# "random_yaw" is stock OCBench's uniform yaw, kept for the stochastic reference data.
SCENES: dict[str, dict] = {
    "fixed_yaw": {**SCENE_KWARGS, "object_yaw_range": (0.0, 0.0)},
    "random_yaw": SCENE_KWARGS,
}


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    description: str
    env_kwargs: dict = field(default_factory=dict)


OBJECTS: dict[str, ObjectSpec] = {
    spec.name: spec
    for spec in (
        ObjectSpec("source", "red 6x6x6 cm cube", {"cube_colors": ("red",)}),
        ObjectSpec("A", "blue 6x6x6 cm cube (appearance)", {"cube_colors": ("blue",)}),
        # At yaw 0 the 9 cm side lies along x.
        ObjectSpec(
            "B",
            "red 9x6x6 cm block (geometry, same grasp width)",
            {"cube_colors": ("red",), "object_size": (0.09, 0.06, 0.06)},
        ),
        ObjectSpec(
            "C",
            "red 9x7x6 cm block (geometry and grasp width)",
            {"cube_colors": ("red",), "object_size": (0.09, 0.07, 0.06)},
        ),
        ObjectSpec(
            "D",
            "blue 9x7x6 cm block (appearance and geometry)",
            {"cube_colors": ("blue",), "object_size": (0.09, 0.07, 0.06)},
        ),
        ObjectSpec(
            "E",
            "red 6x9x6 cm block (behaviour: grasp turned by 90 degrees)",
            {"cube_colors": ("red",), "object_size": (0.06, 0.09, 0.06)},
        ),
    )
}


def env_kwargs(name: str, scene: str = "fixed_yaw") -> dict:
    """OCBench environment arguments for an object: a named scene plus the object."""
    if scene not in SCENES:
        raise KeyError(f"Unknown scene {scene!r}; choose from {sorted(SCENES)}.")
    return {**SCENES[scene], **get_object(name).env_kwargs}


def get_object(name: str) -> ObjectSpec:
    if name not in OBJECTS:
        raise KeyError(f"Unknown object {name!r}; choose from {sorted(OBJECTS)}.")
    return OBJECTS[name]
