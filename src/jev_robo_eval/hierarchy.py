"""Static task semantics and bounded choices for the visual JEV hierarchy.

Candidate construction reads only task identity, the model's prior phase, and
the robot's gripper command. Scene geometry and success/contact facts never
choose a phase, a direction, or a motion scale here.
"""

from dataclasses import dataclass

from .core import ACTION_DESCRIPTIONS, Action


HIERARCHY_PROTOCOL = "jev.phase_scaled_primitive.v3"
PHASE_FIXED_PROTOCOL = "jev.phase_fixed_primitive.v3"
MOTION_SCALES = {"fine": 0.25, "normal": 0.5, "coarse": 1.0}
_XYZ = (Action.X_POS, Action.X_NEG, Action.Y_POS, Action.Y_NEG, Action.Z_POS, Action.Z_NEG)


@dataclass(frozen=True)
class PhaseSchema:
    family: str
    description: str
    phases: dict[str, str]


_CONTACT_PHASES = {
    "approach": "The TCP is still far from or misaligned with the contact surface; contact has not been reached.",
    "contact": "The TCP is adjacent to and aligned with the contact surface, but stable engagement is not yet established.",
    "recover": "Previously established contact was lost, or motion is visibly obstructed; ordinary incomplete approach is not recovery.",
}


def _contact_schema(family: str, operation: str, description: str, operation_text: str) -> PhaseSchema:
    return PhaseSchema(family, description, {**_CONTACT_PHASES, operation: operation_text})


SCHEMAS = {
    "pressing": _contact_schema(
        "pressing", "press", "Press the visible actuator along its travel direction; no object transfer is required.",
        "The TCP is currently touching the aligned actuator surface, and the actuator stroke is incomplete."),
    "horizontal_push": _contact_schema(
        "horizontal_push", "push", "Push the object or panel toward its destination while maintaining contact.",
        "The TCP has established pushing contact, and the object or panel is not yet at its destination."),
    "pull_slide": _contact_schema(
        "pull_slide", "move", "Engage the handle and pull or slide it along the mechanism's visible travel.",
        "The handle is currently engaged, and its slide or hinge travel is incomplete."),
    "reach": PhaseSchema("reach", "Move the robot TCP to the visible target point.", {
        "approach": "The TCP remains clearly separated from the target.",
        "refine": "The TCP is already near the target and only final alignment remains.",
        "recover": "Motion toward the target is visibly obstructed or has lost its alignment.",
    }),
    "pick_place": PhaseSchema("pick_place", "Acquire the object, lift it, then carry and place or insert it.", {
        "approach": "The fingers are not yet horizontally aligned above the object.",
        "lower": "The fingers are horizontally aligned above the object but remain above grasping height.",
        "grasp": "The fingers surround the object at grasping height, but a stable hold is not yet established.",
        "lift": "The object is visibly held but has not yet cleared nearby surfaces.",
        "transfer": "The object is visibly held and clear of surfaces, but remains away from its destination.",
        "place": "The held object is aligned at its destination; final seating or release remains.",
        "recover": "A grasp failed, a held object was dropped, or motion is visibly obstructed.",
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
    candidates = tuple(schema.phases)
    reason = (
        "Classify the current evidence, not the requested task verb. Every phase can be reconsidered now; "
        "the previous inferred phase is only history."
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
    elif schema.family in {"pressing", "horizontal_push"}:
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
