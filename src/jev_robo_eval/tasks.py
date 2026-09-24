"""Human-readable task goals; simulator mechanics stay in environment adapters."""

TASK_GOALS = {
    "reach-v3": "Move the gripper center to the target point.",
    "push-v3": "Push the object across the table to goal_xyz.",
    "door-open-v3": "Pull the door handle to open the hinged door.",
    "drawer-open-v3": "Pull the drawer handle to open the drawer.",
    "pick-place-v3": "Grasp the object, lift it off the table, and carry it to goal_xyz.",
    "peg-insert-side-v3": "Grasp the peg and insert its tip into the side-facing hole in the box.",
    "shelf-place-v3": "Pick up the object and place it on the shelf at goal_xyz.",
    "bin-picking-v3": "Pick up the object in the source bin, lift it over the rim, and place it in the target bin at goal_xyz.",
    "assembly-v3": "Pick up the ring-shaped nut and lower its hole over the vertical peg at goal_xyz.",
    "button-press-v3": "Press the front-facing button inward into its box.",
    "button-press-topdown-v3": "Press the upward-facing button down into its box.",
    "drawer-close-v3": "Push the open drawer inward until it is closed.",
    "window-open-v3": "Move the window handle along its horizontal track to open the window.",
    "window-close-v3": "Move the window handle along its horizontal track to close the window.",
    "plate-slide-v3": "Slide the round plate across the table into the visible goal slot.",
    "plate-slide-side-v3": "Slide the round plate sideways across the table into the visible goal slot.",
    "click_bell": "Press down on the top of the visible bell using the active robot arm.",
    "press_stapler": "Press the top of the visible stapler down using the active robot arm.",
    "move_pillbottle_pad": "Pick up the pill bottle and place it upright on the visible pad.",
    "click_alarmclock": "Press down on the top button of the visible alarm clock using the active robot arm.",
    "place_container_plate": "Pick up the visible cup or bowl, place it on the plate, and release it.",
    "place_object_scale": "Pick up the small object beside the electronic scale, place it on top of the scale, and release it.",
}

# These descriptions identify visible objects and intended outcomes without
# supplying simulator object positions, target coordinates, or contact state.
SENSOR_TASK_GOALS = {
    "reach-v3": (
        "Move the gripper center to the task target. This task's target is an invisible simulator "
        "point when goal markers are hidden; an RGB-only observation cannot identify it."
    ),
    "push-v3": (
        "Push the small object across the table to the task target. This task's target is an "
        "invisible simulator point when goal markers are hidden; an RGB-only observation cannot identify it."
    ),
    "door-open-v3": "From the camera image, find the door handle and pull it to open the hinged door.",
    "drawer-open-v3": "From the camera image, find the drawer handle and pull the drawer out to open it.",
    "pick-place-v3": (
        "Grasp and lift the small object, then carry it to the task target. This task's target is "
        "an invisible simulator point when goal markers are hidden; an RGB-only observation cannot identify it."
    ),
    "peg-insert-side-v3": (
        "From the camera image, find the loose peg and the side-facing hole in the box. "
        "Grasp the peg and insert its tip into the hole."
    ),
    **{name: TASK_GOALS[name] for name in (
        "button-press-v3", "button-press-topdown-v3", "drawer-close-v3",
        "window-open-v3", "window-close-v3", "plate-slide-v3", "plate-slide-side-v3",
        "click_alarmclock", "place_container_plate", "place_object_scale",
    )},
    "click_bell": TASK_GOALS["click_bell"],
    "press_stapler": TASK_GOALS["press_stapler"],
    "move_pillbottle_pad": TASK_GOALS["move_pillbottle_pad"],
    "shelf-place-v3": (
        "From the camera image, find the small block on the table and the wooden shelf. "
        "Grasp the block, lift it above the shelf opening, then place it inside the shelf."
    ),
    "bin-picking-v3": (
        "From the camera image, find the object in the source bin and the separate target bin. "
        "Grasp the object, lift it clear of the source bin rim, move it over the target bin, "
        "then lower and release it inside the target bin."
    ),
    "assembly-v3": (
        "From the camera image, find the ring-shaped nut on the table and the upright peg. "
        "Grasp the nut, lift it, align the nut's central hole above the peg, then lower "
        "the nut onto the peg."
    ),
}

TASK_GUIDES = {
    "push-v3": (
        "First approach the OBJECT with the gripper open, align over its XY position, and lower "
        "the TCP near the object. Then close the gripper and push the object horizontally toward "
        "goal_xyz. The OBJECT must reach the goal; reaching the goal with an empty hand does not help."
    ),
    "door-open-v3": (
        "Close the gripper. First approach the HANDLE object_xyz from above and slightly toward "
        "its +X/+Y side, then descend to contact. Keep contact and move the handle through its hinge "
        "arc toward negative X; Y decreases early in this motion. Success depends on handle X."
    ),
    "pick-place-v3": (
        "First open the gripper and approach above the OBJECT at object_xyz. Align XY, then lower "
        "the TCP to object height, close the gripper, and wait for gripping. Lift the object off the "
        "table before carrying it toward goal_xyz. Keep the gripper CLOSED during lifting and carrying. "
        "The object, not the empty gripper, must reach the goal."
    ),
    "peg-insert-side-v3": (
        "First open the gripper, align above pegGrasp (object_xyz), lower onto it, and close. Lift "
        "the peg. While holding it, align pegHead to the goal Y and Z while staying on the positive "
        "X side of the hole; then move negative X to insert. pegHead is the tip, object_xyz the "
        "grasp point; judge completion by the tip position. Keep the gripper closed while carrying."
    ),
}

ROBOTWIN_TASK_GUIDES = {
    "click_bell": (
        "Use the active arm to press the visible bell. Close the gripper to make a pressing tip. "
        "Move above the bell's top, align horizontally, then move downward to press it. "
        "The task requires pressing, so lifting or carrying the bell is unnecessary."
    ),
    "press_stapler": (
        "Use the active arm to press the visible stapler. Close the gripper to make a pressing tip. "
        "Move above the stapler's top, align horizontally, then move downward to press it closed. "
        "The task requires pressing, so lifting or carrying the stapler is unnecessary."
    ),
    "move_pillbottle_pad": (
        "Use the active arm to move the pill bottle to the visible pad. Approach above the bottle "
        "with open fingers, lower to surround it, close, then lift before carrying over the pad. "
        "Lower the bottle onto the pad and release. A close command alone does not prove a grasp."
    ),
}
TASK_GUIDES.update(ROBOTWIN_TASK_GUIDES)


def task_goal(task_name: str, information: str = "privileged") -> str:
    if information == "nonprivileged":
        return SENSOR_TASK_GOALS.get(task_name, f"Complete the visible {task_name} task using the camera image.")
    if information != "privileged":
        raise ValueError("information must be privileged or nonprivileged")
    return TASK_GOALS.get(task_name, f"Complete the {task_name} task.")
