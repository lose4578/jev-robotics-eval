"""Target roles and directional evidence derived from permitted pose fields."""

import math
from numbers import Real

from .core import Action


CONTACT_SURFACES = {
    "button-press-topdown-v3": "red button top", "button-press-v3": "front button surface",
    "click_bell": "bell top", "press_stapler": "stapler top", "click_alarmclock": "alarm clock top button",
    "window-open-v3": "window handle", "window-close-v3": "window handle",
    "drawer-open-v3": "drawer handle", "drawer-close-v3": "drawer handle",
    "door-open-v3": "door handle", "door-close-v3": "door handle",
    "plate-slide-v3": "round plate", "plate-slide-side-v3": "round plate",
    "assembly-v3": "ring-shaped nut", "peg-insert-side-v3": "loose peg",
    "move_pillbottle_pad": "pill bottle", "place_container_plate": "cup or bowl",
    "place_object_scale": "small object beside the scale", "reach-v3": "target point",
}
_PHASE_FACTS = {"both_fingers_touch_object", "object_lift_m", "door_angle_rad",
                "assembly_geometry_success", "nut_center_goal_xy_m", "nut_center_below_peg_top_m",
                "window_slide_m", "button_remaining_travel_m"}
_OPERATIONS = {
    "approach": "Approach the current contact point.", "contact": "Establish contact at the current contact point.",
    "lower": "Align the fingers at the object's grasping height.", "grasp": "Establish a stable hold around the object.",
    "lift": "Raise the held object clear of nearby surfaces.", "transfer": "Carry the held object to its destination.",
    "place": "Seat the held object at its destination and release when appropriate.",
    "press": "Complete the actuator stroke while keeping contact.", "push": "Move the contacted object toward its destination.",
    "move": "Move the engaged handle toward its destination.", "refine": "Align with the current target point.",
    "recover": "Reestablish alignment with the current contact point.",
}


def compact_phase_state(state: dict, task_name: str, family: str, targets: dict) -> dict:
    """Present current geometry without task verbs, intended phases, or bulky context."""
    result = {"current_contact_surface": CONTACT_SURFACES.get(task_name, "movable object"),
              "robot_tcp_xyz": state["control_xyz"], "gripper_command": state["gripper_command"]}
    if state.get("privilege_level") in {1, 2}:
        if "contact" in targets:
            result["current_contact_xyz"] = targets["contact"]["target_xyz"]
        if family in {"pick_place", "reach"} and "destination" in targets:
            result["destination_xyz"] = targets["destination"]["target_xyz"]
    if state.get("privilege_level") == 2:
        result.update({key: state[key] for key in sorted(_PHASE_FACTS) if key in state})
    return result


def compact_action_state(state: dict, task_name: str, family: str, phase: str, targets: dict) -> dict:
    """Expose only the model-selected phase's current target, calibration, and short history."""
    role = phase_target_role(family, phase, state.get("environment"))
    result = {"robot_tcp_xyz": state["control_xyz"], "inferred_phase": phase,
              "gripper_command": state["gripper_command"], "task": _OPERATIONS[phase]}
    if family == "reach":
        result["task"] = "Align with the current target point."
    if state.get("privilege_level") in {1, 2} and role in targets:
        result[f"current_{role}_xyz"] = targets[role]["target_xyz"]
    else:
        result["current_target_surface"] = (CONTACT_SURFACES.get(task_name, "movable object") if role == "contact"
                                             else "visible destination" if role == "destination"
                                             else "clear space above the held object")
    for key in ("nominal_motion_step_m", "previous_action", "last_tcp_motion_xyz", "consecutive_same_action",
                "consecutive_axis_reversals", "tcp_pixel"):
        if key in state:
            result[key] = state[key]
    if "action_screen_directions" in state:
        result["action_screen_directions"] = {
            key: value.get("delta_px") if isinstance(value, dict) else value
            for key, value in state["action_screen_directions"].items()
        }
    if state.get("privilege_level") == 2:
        result.update({key: state[key] for key in sorted(_PHASE_FACTS) if key in state})
    return result


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in value)):
        return None
    return [float(v) for v in value]


def target_evidence(state: dict, family: str) -> dict:
    """Compute no geometry at L0, and never consult unlisted state fields."""
    if state.get("privilege_level") not in {1, 2}:
        return {}
    tcp = _point(state.get("control_xyz"))
    if tcp is None:
        return {}
    robotwin_press = family == "pressing" and state.get("environment") == "robotwin"
    sources = {} if robotwin_press else {"destination": "goal_xyz"}
    if family != "reach":
        sources["contact"] = "goal_xyz" if robotwin_press else "object_xyz"
    result = {}
    for role, source in sources.items():
        point = _point(state.get(source))
        if point is None:
            continue
        delta = [round(target - now, 6) for target, now in zip(point, tcp)]
        result[role] = {
            "source": source, "provenance": "derived_from_permitted_pose",
            "target_xyz": point, "delta_xyz": delta,
            "axis_abs_error_m": [abs(value) for value in delta],
            "distance_m": round(math.sqrt(sum(value * value for value in delta)), 6),
            "xy_distance_m": round(math.hypot(*delta[:2]), 6),
            "meaning": ("Current contact location; acquire contact here before manipulating."
                        if role == "contact" else
                        "Final task destination or actuator endpoint; not the initial contact location."),
        }
    return result


def phase_target_role(family: str, phase: str, environment: str | None = None) -> str | None:
    if family == "reach":
        return "destination"
    if family == "pressing" and environment == "robotwin":
        return "contact"
    if phase in {"approach", "contact", "lower", "grasp", "recover"}:
        return "contact"
    if phase == "lift":
        return None
    return "destination"


def candidate_grounding(action: Action, family: str, phase: str, evidence: dict,
                        environment: str | None = None, *, scale: float = 1.0,
                        nominal_motion_step_m: float | None = None) -> str:
    """Describe both signs; do not choose or remove any action based on geometry."""
    role = phase_target_role(family, phase, environment)
    if action == Action.GRIP_CLOSE:
        return "The fingers already surround the current contact point and closing is needed."
    if action == Action.GRIP_OPEN:
        return "The fingers need to open for acquisition or release at the current target."
    if action == Action.HOLD:
        return "The TCP is already at the current target, or a gripper change is settling."
    nominal = (float(nominal_motion_step_m) * scale
               if isinstance(nominal_motion_step_m, Real) and not isinstance(nominal_motion_step_m, bool)
               and math.isfinite(nominal_motion_step_m) and nominal_motion_step_m > 0 else None)
    calibration = f" Nominal step {nominal:.4f} m." if nominal is not None else f" Movement scale {scale:g}."
    if role is None:
        return "Lift clearance: upward motion raises the held object; image evidence must establish that it is held." + calibration
    axis = "xyz".index(action.value[0])
    sign = 1 if action.value.endswith("pos") else -1
    axis_name = action.value[0].upper()
    condition = f"Appropriate when target {axis_name} is {'greater' if sign > 0 else 'smaller'} than TCP {axis_name}."
    point = evidence.get(role)
    if point is None:
        return (f"Appropriate when the visible {'contact surface' if role == 'contact' else 'destination'} lies in "
                f"{'positive' if sign > 0 else 'negative'} world {axis_name} from the TCP; use the image motion calibration." + calibration)
    delta = point["delta_xyz"][axis]
    relation = "toward" if sign * delta > 0 else "away from"
    description = condition + f" This moves {relation} the {role} coordinate; current signed error {delta:+.4f} m." + calibration
    if nominal is not None and sign * delta > 0 and nominal > abs(delta):
        description += " This nominal step may cross the target coordinate."
    return description
