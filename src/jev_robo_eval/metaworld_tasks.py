"""Task-specific geometric facts. These fields do not choose robot actions."""

import numpy as np


def relative_text(delta, subject="Target"):
    directions = (("right (+X)", "left (-X)"),
                  ("forward (+Y)", "backward (-Y)"),
                  ("up (+Z)", "down (-Z)"))
    return [f"{subject} is {abs(float(d)) * 100:.1f} cm {pos if d >= 0 else neg}"
            for d, (pos, neg) in zip(delta, directions)]


def _both_fingers_touch_body(env, body_name):
    """Check contacts with collidable geoms in a task body and its children.

    Some MetaWorld tasks use unnamed collision geoms; their named visual geom
    cannot be passed to ``touching_object`` to detect a physical grasp.
    """
    model, data = env.model, env.data
    root_id = model.body(body_name).id

    def belongs_to_object(body_id):
        while body_id:
            if body_id == root_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    object_geoms = {
        geom_id for geom_id in range(model.ngeom)
        if belongs_to_object(int(model.geom_bodyid[geom_id]))
        and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
    }
    pad_ids = (model.geom("leftpad_geom").id, model.geom("rightpad_geom").id)
    forces = [0.0, 0.0]
    for contact in data.contact:
        if contact.efc_address < 0:
            continue
        for index, pad_id in enumerate(pad_ids):
            if ((contact.geom1 == pad_id and contact.geom2 in object_geoms)
                    or (contact.geom2 == pad_id and contact.geom1 in object_geoms)):
                forces[index] += float(data.efc_force[contact.efc_address])
    return all(force > 0 for force in forces)


def task_facts(env, task_name, obs, control, initial_object):
    goal, obj = obs[-3:], obs[4:7]
    if task_name == "reach-v3":
        return {"target_relative_to_gripper": relative_text(goal - control)}
    facts = {
        "object_xyz": np.round(obj, 4).tolist(),
        "object_relative_to_gripper": relative_text(obj - control, "Object"),
        "goal_relative_to_object": relative_text(goal - obj, "Goal"),
        "object_minus_control_xyz": np.round(obj - control, 4).tolist(),
        "goal_minus_object_xyz": np.round(goal - obj, 4).tolist(),
        "tcp_to_object_m": round(float(np.linalg.norm(obj - control)), 4),
        "object_lift_m": round(float(obj[2] - initial_object[2]), 4),
    }
    distance = float(np.linalg.norm(obj - goal))
    if task_name in {"push-v3", "pick-place-v3", "peg-insert-side-v3"}:
        facts["both_fingers_touch_object"] = bool(
            env.touching_object(env.data.geom("peg").id)
            if task_name == "peg-insert-side-v3" else env.touching_main_object
        )
    elif task_name in {"shelf-place-v3", "bin-picking-v3", "assembly-v3"}:
        body_name = "RoundNut" if task_name == "assembly-v3" else "obj"
        facts["both_fingers_touch_object"] = _both_fingers_touch_body(env, body_name)
    if task_name == "door-open-v3":
        distance = float(abs(obj[0] - goal[0]))
        facts["door_angle_rad"] = round(float(env.data.joint("doorjoint").qpos[0]), 4)
    elif task_name == "peg-insert-side-v3":
        head = env.data.site("pegHead").xpos.copy()
        facts["peg_head_xyz"] = np.round(head, 4).tolist()
        facts["goal_relative_to_peg_head"] = relative_text(goal - head, "Goal")
        distance = float(np.linalg.norm((head - goal) * [1, 2, 2]))
    elif task_name == "assembly-v3":
        center = env.data.site("RoundNut").xpos.copy()
        xy_error = float(np.linalg.norm(center[:2] - goal[:2]))
        below_top = float(goal[2] - center[2])
        facts["nut_center_xyz"] = np.round(center, 4).tolist()
        facts["goal_relative_to_nut_center"] = relative_text(goal - center, "Goal")
        facts["nut_center_goal_xy_m"] = round(xy_error, 5)
        facts["nut_center_below_peg_top_m"] = round(below_top, 5)
        facts["assembly_geometry_success"] = xy_error < 0.02 and below_top > 0
        del facts["goal_relative_to_object"]
        del facts["goal_minus_object_xyz"]
        distance = xy_error
    facts["success_distance_m"] = round(distance, 5)
    return facts
