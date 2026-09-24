"""Static task semantics and bounded choices for the visual JEV hierarchy.

Candidate construction reads only task identity, the model's prior phase, and
the robot's gripper command. Scene geometry and success/contact facts never
choose a phase, a direction, or a motion scale here.
"""

from dataclasses import dataclass

from .core import ACTION_DESCRIPTIONS, Action


HIERARCHY_PROTOCOL = "jev.phase_scaled_primitive.v1"
PHASE_FIXED_PROTOCOL = "jev.phase_fixed_primitive.v1"
MOTION_SCALES = {"fine": 0.25, "normal": 0.5, "coarse": 1.0}
_XYZ = (Action.X_POS, Action.X_NEG, Action.Y_POS, Action.Y_NEG, Action.Z_POS, Action.Z_NEG)


@dataclass(frozen=True)
class PhaseSchema:
    family: str
    description: str
    phases: dict[str, str]
    neighbors: dict[str, tuple[str, ...]]


_CONTACT_PHASES = {
    "approach": "Move toward the visible contact surface, leaving room to align the tool.",
    "contact": "Align the fingers with the visible contact surface and establish gentle contact.",
    "recover": "Reposition after a missed contact, obstruction, or motion without visible progress.",
}


def _contact_schema(family: str, operation: str, description: str, operation_text: str) -> PhaseSchema:
    return PhaseSchema(family, description, {**_CONTACT_PHASES, operation: operation_text}, {
        "approach": ("approach", "contact", "recover"),
        "contact": ("contact", "approach", operation, "recover"),
        operation: (operation, "contact", "recover"),
        "recover": ("recover", "approach", "contact"),
    })


SCHEMAS = {
    "pressing": _contact_schema(
        "pressing", "press", "Press the visible actuator along its travel direction; no object transfer is required.",
        "Maintain contact and move the actuator along its visible travel direction to press it."),
    "horizontal_push": _contact_schema(
        "horizontal_push", "push", "Push the object or panel toward its destination while maintaining contact.",
        "Push horizontally toward the visible destination, adjusting contact when necessary."),
    "pull_slide": _contact_schema(
        "pull_slide", "move", "Engage the handle and pull or slide it along the mechanism's visible travel.",
        "Preserve handle contact and move along the intended slide or hinge motion."),
    "reach": PhaseSchema("reach", "Move the robot TCP to the visible target point.", {
        "approach": "Reduce the visible distance between the TCP and target.",
        "refine": "Make small corrections near the visible target without changing the gripper.",
        "recover": "Reposition when a prior motion was blocked or increased the target error.",
    }, {
        "approach": ("approach", "refine", "recover"),
        "refine": ("refine", "approach", "recover"),
        "recover": ("recover", "approach", "refine"),
    }),
    "pick_place": PhaseSchema("pick_place", "Acquire the object, lift it, then carry and place or insert it.", {
        "approach": "Align the open fingers above the visible object before descending.",
        "lower": "Descend toward the object while correcting the fingers' alignment.",
        "grasp": "Close the fingers around the object and check that it is held.",
        "lift": "Raise the held object clear of nearby surfaces and obstacles.",
        "transfer": "Carry the raised object toward the visible destination.",
        "place": "Align and lower or insert the held object; release when appropriate.",
        "recover": "Reacquire a missed or dropped object, or reposition after blocked motion.",
    }, {
        "approach": ("approach", "lower", "recover"),
        "lower": ("lower", "approach", "grasp", "recover"),
        "grasp": ("grasp", "lift", "lower", "recover"),
        "lift": ("lift", "transfer", "recover"),
        "transfer": ("transfer", "place", "recover"),
        "place": ("place", "transfer", "recover"),
        "recover": ("recover", "approach", "lower"),
    }),
}

TASK_FAMILIES = {
    **dict.fromkeys(("button-press-v3", "button-press-topdown-v3", "click_bell", "press_stapler", "click_alarmclock"), "pressing"),
    **dict.fromkeys(("push-v3", "drawer-close-v3", "door-close-v3", "plate-slide-v3", "plate-slide-side-v3"), "horizontal_push"),
    **dict.fromkeys(("door-open-v3", "drawer-open-v3", "window-open-v3", "window-close-v3"), "pull_slide"),
    "reach-v3": "reach",
    **dict.fromkeys(("pick-place-v3", "peg-insert-side-v3", "shelf-place-v3", "bin-picking-v3",
                    "assembly-v3", "move_pillbottle_pad", "place_container_plate", "place_object_scale"), "pick_place"),
}


def schema_for_task(task_name: str) -> PhaseSchema:
    try:
        return SCHEMAS[TASK_FAMILIES[task_name]]
    except KeyError as exc:
        raise ValueError(f"Adaptive hierarchy has no task-family schema for {task_name!r}") from exc


def eligible_phases(schema: PhaseSchema, previous_phase: str | None,
                    gripper_command: str) -> tuple[dict[str, str], str]:
    prior = previous_phase if previous_phase in schema.neighbors else "approach"
    candidates = schema.neighbors[prior]
    reason = (
        f"Use the {prior} neighborhood as a temporal prior based on model history, not proof of progress. "
        "Choose recovery if current evidence contradicts the previous phase."
    )
    if schema.family == "pick_place" and gripper_command == "open":
        candidates = tuple(phase for phase in candidates if phase not in {"lift", "transfer", "place"})
        reason += " The robot command is OPEN, so carrying a held object is ineligible."
    return {phase: schema.phases[phase] for phase in candidates}, reason


@dataclass(frozen=True)
class MotionCandidate:
    id: str
    action: Action
    scale: float
    description: str

    def to_dict(self) -> dict:
        return {"id": self.id, "action": self.action.value, "action_scale": self.scale,
                "description": self.description}


def phase_action_candidates(schema: PhaseSchema, phase: str,
                            gripper_command: str, granularity: str = "adaptive") -> tuple[MotionCandidate, ...]:
    """Offer compact signed movements; the model chooses both direction and scale."""
    if granularity not in {"adaptive", "phase-fixed"}:
        raise ValueError("Hierarchy candidate granularity must be adaptive or phase-fixed")
    if phase not in schema.phases:
        raise ValueError(f"Unknown {schema.family} phase: {phase}")
    candidates = []
    for action in _XYZ:
        if granularity == "phase-fixed":
            candidates.append(MotionCandidate(
                f"{action.value}_fixed", action, 1.0,
                f"{ACTION_DESCRIPTIONS[action]} Fixed motion: 1 times the configured movement amplitude; preserve gripper."))
            continue
        for size, scale in MOTION_SCALES.items():
            candidates.append(MotionCandidate(
                f"{action.value}_{size}", action, scale,
                f"{ACTION_DESCRIPTIONS[action]} {size.capitalize()} motion: {scale:g} times the configured movement amplitude; preserve gripper."))

    settings = ()
    if schema.family == "pick_place":
        if phase in {"approach", "lower", "recover", "place"}:
            settings = (Action.GRIP_OPEN,)
        elif phase == "grasp":
            settings = (Action.GRIP_CLOSE,)
    elif schema.family in {"pressing", "horizontal_push"} and phase in {"approach", "contact", "recover"}:
        settings = (Action.GRIP_CLOSE,)
    elif schema.family == "pull_slide" and phase in {"contact", "recover"}:
        settings = (Action.GRIP_OPEN, Action.GRIP_CLOSE)
    for action in settings:
        desired = "open" if action == Action.GRIP_OPEN else "closed"
        if gripper_command != desired:
            candidates.append(MotionCandidate(action.value, action, 1.0, ACTION_DESCRIPTIONS[action]))
    candidates.append(MotionCandidate("hold", Action.HOLD, 1.0,
                                      "Hold the current position and gripper setting while settling or if complete."))
    return tuple(candidates)
