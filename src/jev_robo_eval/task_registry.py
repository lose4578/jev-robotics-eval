"""Tasks exposed by the CLI and their implemented oracle guide capability.

Registration describes the primitive-action environment interface. It is not
evidence that a controller succeeds, a visual action is calibrated, or an atomic
executor supports every task. New tasks are intended for L0/L1/L2 evaluation;
``oracle_supported`` is true only when a task-specific L3 guide exists.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    environment: str
    family: str
    oracle_supported: bool = False


TASK_REGISTRY = {
    "reach-v3": TaskSpec("metaworld", "reach"),
    "push-v3": TaskSpec("metaworld", "push", True),
    "door-open-v3": TaskSpec("metaworld", "door", True),
    "drawer-open-v3": TaskSpec("metaworld", "drawer"),
    "pick-place-v3": TaskSpec("metaworld", "pick_place", True),
    "peg-insert-side-v3": TaskSpec("metaworld", "insert", True),
    "shelf-place-v3": TaskSpec("metaworld", "pick_place", True),
    "bin-picking-v3": TaskSpec("metaworld", "pick_place", True),
    "assembly-v3": TaskSpec("metaworld", "assembly", True),
    "button-press-v3": TaskSpec("metaworld", "press"),
    "button-press-topdown-v3": TaskSpec("metaworld", "press"),
    "drawer-close-v3": TaskSpec("metaworld", "drawer"),
    "window-open-v3": TaskSpec("metaworld", "window"),
    "window-close-v3": TaskSpec("metaworld", "window"),
    "plate-slide-v3": TaskSpec("metaworld", "push"),
    "plate-slide-side-v3": TaskSpec("metaworld", "push"),
    "click_bell": TaskSpec("robotwin", "press", True),
    "press_stapler": TaskSpec("robotwin", "press", True),
    "move_pillbottle_pad": TaskSpec("robotwin", "pick_place", True),
    "click_alarmclock": TaskSpec("robotwin", "press"),
    "place_container_plate": TaskSpec("robotwin", "pick_place"),
    "place_object_scale": TaskSpec("robotwin", "pick_place"),
}


def tasks_for_environment(environment: str) -> tuple[str, ...]:
    if environment not in {"metaworld", "robotwin"}:
        raise ValueError(f"Unknown environment: {environment}")
    return tuple(name for name, spec in TASK_REGISTRY.items()
                 if spec.environment == environment)


METAWORLD_TASKS = tasks_for_environment("metaworld")
ROBOTWIN_TASKS = tasks_for_environment("robotwin")
