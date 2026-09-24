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
                "assembly_geometry_success", "nut_center_goal_xy_m", "nut_center_below_peg_top_m"}
_ACTION_KEYS = {
    "task", "task_name", "privilege_level", "environment", "active_arm", "coordinate_frame",
    "control_xyz", "gripper_command", "gripper_opening", "object_xyz", "goal_xyz",
    "peg_head_xyz", "nut_center_xyz", "target_evidence", "action_screen_directions", "tcp_pixel",
    "nominal_motion_step_m", "previous_action", "previous_candidate", "previous_action_scale",
    "consecutive_same_action", "consecutive_axis_reversals", "last_tcp_motion_xyz", "motion_history",
    "previous_inferred_phase", "decisions_in_previous_phase",
} | _PHASE_FACTS


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


def compact_action_state(state: dict) -> dict:
    """Retain action evidence and calibration while excluding joints and the full scene."""
    return {key: value for key, value in state.items() if key in _ACTION_KEYS}


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
    if action in {Action.GRIP_OPEN, Action.GRIP_CLOSE, Action.HOLD}:
        return "No target alignment motion."
    nominal = (float(nominal_motion_step_m) * scale
               if isinstance(nominal_motion_step_m, Real) and not isinstance(nominal_motion_step_m, bool)
               and math.isfinite(nominal_motion_step_m) and nominal_motion_step_m > 0 else None)
    calibration = f" Nominal motion {nominal:.4g} m; actual motion may differ." if nominal is not None else ""
    if role is None:
        return "Lift clearance: upward motion raises the held object; image evidence must establish that it is held." + calibration
    label = "Contact alignment" if role == "contact" else "Destination alignment after engagement"
    point = evidence.get(role)
    if point is None:
        return f"{label}: judge direction from the visible {'contact surface' if role == 'contact' else 'destination'} and TCP." + calibration
    axis = "xyz".index(action.value[0])
    sign = 1 if action.value.endswith("pos") else -1
    delta = point["delta_xyz"][axis]
    relation = ("initially reduces" if sign * delta > 0 else
                "increases" if sign * delta < 0 else "moves away from zero")
    description = f"{label}: {action.value} {relation} axis error (target minus TCP {delta:+.4f} m)." + calibration
    if nominal is not None and sign * delta > 0 and nominal > abs(delta):
        description += " This nominal step may cross the target coordinate."
    if role == "destination" and family != "reach" and "contact" in evidence:
        description += f" Current contact point remains {evidence['contact']['distance_m']:.3f} m from TCP; destination travel requires engagement."
    return description
