"""Source and target objects from the proposal (Aims/Goals, Scope).

Each object maps to keyword arguments for the OCBench block environment, so collection,
training, and evaluation build the same scene from one name. Robots, cameras, lighting,
and the task stay fixed; only the manipulated object changes.

Colours follow the proposal: a red source cube and a blue cube for target A. Stock OCBench
renders the single cube blue, so the source sets `cube_colors=('red',)` explicitly
(supported by our OCBench fork, CPU environments only).

Every object shares `SCENE_KWARGS`: a dark-gray 12 cm (inner side) square marks the fixed
goal, and success requires the whole cube inside it, resting on the table, released and
still for 1 s. With 12 cm, 97% of resting expert placements fit fully inside; 8 cm fits
only 60% (measured over 200 episodes, 8 November 2026).

Initial states are narrower than stock OCBench (decided 8 November 2026): the cube starts in
a 20 x 20 cm area beside the goal instead of anywhere on a 42 x 110 cm area, and the arm
always starts from one pose instead of a random one, as in ACT's simulated tasks.
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


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    description: str
    env_kwargs: dict = field(default_factory=dict)
    implemented: bool = True


OBJECTS: dict[str, ObjectSpec] = {
    spec.name: spec
    for spec in (
        ObjectSpec("source", "red 6x6x6 cm cube", {"cube_colors": ("red",)}),
        ObjectSpec("A", "blue 6x6x6 cm cube (appearance)", {"cube_colors": ("blue",)}),
        # Geometry targets need a new block shape and adapted expert grasps in the fork.
        ObjectSpec("B", "red 9x6x6 cm block (geometry, same grasp width)", implemented=False),
        ObjectSpec("C", "red 9x7x6 cm block (geometry and grasp width)", implemented=False),
        ObjectSpec("D", "blue 9x7x6 cm block (appearance and geometry)", implemented=False),
    )
}


def env_kwargs(name: str) -> dict:
    """OCBench environment arguments for an object: the shared scene plus the object."""
    return {**SCENE_KWARGS, **get_object(name).env_kwargs}


def get_object(name: str) -> ObjectSpec:
    if name not in OBJECTS:
        raise KeyError(f"Unknown object {name!r}; choose from {sorted(OBJECTS)}.")
    spec = OBJECTS[name]
    if not spec.implemented:
        raise NotImplementedError(f"Object {name} ({spec.description}) is not implemented yet.")
    return spec
