"""Explicit observation budgets, shared by the simulator and policy boundary."""

from copy import deepcopy


SENSOR_KEYS = {
    "task_name", "coordinate_frame", "control_point", "control_xyz", "robot",
    "gripper_opening", "gripper_command", "simulator_steps", "information",
    "privilege_level", "camera_name", "action_screen_directions",
    "environment", "active_arm", "control_steps", "tcp_pixel",
}
POSE_KEYS = {"object_xyz", "goal_xyz", "scene", "peg_head_xyz", "nut_center_xyz",
             "atomic_target_xyz"}
INTERACTION_KEYS = {
    "goal_minus_control_xyz", "control_to_goal_distance", "target_relative_to_gripper",
    "object_relative_to_gripper", "goal_relative_to_object", "object_minus_control_xyz",
    "goal_minus_object_xyz", "tcp_to_object_m", "object_lift_m",
    "both_fingers_touch_object", "door_angle_rad", "goal_relative_to_peg_head",
    "goal_relative_to_nut_center", "nut_center_goal_xy_m", "nut_center_below_peg_top_m",
    "assembly_geometry_success", "success_distance_m",
}
ROBOT_KEYS = {
    "hand_body_xyz", "hand_quat_wxyz", "arm_joint_order",
    "arm_joint_position_rad", "arm_joint_velocity_rad_s",
}
WAYPOINT_KEYS = {
    "phase", "target_xyz", "delta_xyz", "relative_direction", "instruction",
    "desired_gripper", "gripper_change_needed",
}


def validate_plan_only(level, plan_only):
    if plan_only and level != 3:
        raise ValueError("plan_only requires privilege_level=3 (oracle waypoint source)")


def resolve_access(information="privileged", guidance=None, privilege_level=None):
    if information not in {"privileged", "nonprivileged"}:
        raise ValueError("information must be privileged or nonprivileged")
    if privilege_level is not None:
        if privilege_level not in {0, 1, 2, 3}:
            raise ValueError("privilege_level must be 0, 1, 2, or 3")
        information = "nonprivileged" if privilege_level == 0 else "privileged"
        guidance = guidance or ("waypoints" if privilege_level == 3 else "direct")
    else:
        guidance = guidance or "direct"
        privilege_level = 0 if information == "nonprivileged" else 3 if guidance == "waypoints" else 2
    if guidance not in {"direct", "waypoints"}:
        raise ValueError("guidance must be direct or waypoints")
    if guidance == "waypoints" and privilege_level != 3:
        raise ValueError("Oracle waypoint guidance requires privilege_level=3")
    if privilege_level == 3 and guidance != "waypoints":
        raise ValueError("privilege_level=3 includes waypoint guidance; use level 2 for direct control")
    return information, guidance, privilege_level


def filter_policy_state(state, level, *, plan_only=False):
    """Never pass a new evaluation field merely because it was added upstream."""
    plan_only = plan_only or state.get("plan_only", False)
    validate_plan_only(level, plan_only)
    allowed = set(SENSOR_KEYS)
    if level >= 1 and not plan_only:
        allowed |= POSE_KEYS
    if level >= 2 and not plan_only:
        allowed |= INTERACTION_KEYS
    if level == 3:
        allowed |= {"active_waypoint", "image_annotations"}
    if level not in {0, 1, 2, 3}:
        raise ValueError("Unknown privilege level")
    result = {key: deepcopy(value) for key, value in state.items() if key in allowed}
    if "robot" in result:
        result["robot"] = {key: value for key, value in result["robot"].items() if key in ROBOT_KEYS}
    if plan_only:
        result["plan_only"] = True
        result["plan_provenance"] = "oracle_rules_using_environment_truth"
        if "active_waypoint" in result:
            result["active_waypoint"] = {
                key: value for key, value in result["active_waypoint"].items()
                if key in WAYPOINT_KEYS
            }
    return result
