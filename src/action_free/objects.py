"""Source and target objects from the proposal (Aims/Goals, Scope).

Each object maps to keyword arguments for the OCBench block environment, so collection,
training, and evaluation build the same scene from one name. Robots, cameras, lighting,
and the task stay fixed; only the manipulated object changes.

Colours follow the proposal: a red source cube and a blue cube for target A. Stock OCBench
renders the single cube blue, so the source sets `cube_colors=('red',)` explicitly
(supported by our OCBench fork, CPU environments only).
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


def get_object(name: str) -> ObjectSpec:
    if name not in OBJECTS:
        raise KeyError(f"Unknown object {name!r}; choose from {sorted(OBJECTS)}.")
    spec = OBJECTS[name]
    if not spec.implemented:
        raise NotImplementedError(f"Object {name} ({spec.description}) is not implemented yet.")
    return spec
