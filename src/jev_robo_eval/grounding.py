"""Target roles and directional evidence derived from permitted pose fields."""

import math
from numbers import Real

from .core import Action


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
