"""Hand-authored high-level targets for three harder MT1 tasks.

JEV still chooses one of the nine primitive actions at every decision. Targets
are gripper TCP positions in MetaWorld world coordinates.
"""

import numpy as np

from .metaworld_tasks import relative_text


class HardWaypointGuide:
    TASKS = {"shelf-place-v3", "bin-picking-v3", "assembly-v3"}

    def __init__(self, task_name):
        if task_name not in self.TASKS:
            raise ValueError(f"No hard waypoint guide for {task_name}")
        self.task_name = task_name
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.phase_started = 0
        self.initial_z = None
        self._lost_grip_since = None

    def _phase(self, name, steps):
        self.phase = name
        self.phase_started = steps
        if name == "approach":
            self._lost_grip_since = None

    def _held_target(self, tcp, current_point, desired_point):
        return tcp + desired_point - current_point

    def update(self, state):
        tcp = np.asarray(state["control_xyz"], dtype=float)
        obj = np.asarray(state["object_xyz"], dtype=float)
        goal = np.asarray(state["goal_xyz"], dtype=float)
        steps = state["simulator_steps"]
        if self.initial_z is None:
            self.initial_z = float(obj[2])

        if self.phase in {"align_x", "advance", "carry", "place", "align_center", "mount"}:
            distance = float(np.linalg.norm(tcp - obj))
            lift = float(obj[2] - self.initial_z)
            contact = bool(state.get("both_fingers_touch_object", False))
            suspected_loss = not contact and (
                distance > 0.10 or (lift < 0.025 and distance > 0.05)
            )
            if suspected_loss:
                if self._lost_grip_since is None:
                    self._lost_grip_since = steps
                elif steps - self._lost_grip_since >= 6:
                    self._phase("approach", steps)
                    return self.update(state)
            else:
                self._lost_grip_since = None

        if self.task_name == "assembly-v3":
            grasp_offset = np.array([-0.015, 0.0, 0.0])
            approach_height = 0.10
        elif self.task_name == "bin-picking-v3":
            # Keep the fingers within the starting bin when the cube is at an edge.
            grasp_offset = np.array([0.0, np.clip(obj[1], 0.675, 0.725) - obj[1], 0.0])
            approach_height = 0.14
        else:
            grasp_offset = np.zeros(3)
            approach_height = 0.12

        if self.phase == "approach":
            target = obj + grasp_offset + [0.0, 0.0, approach_height]
            if np.linalg.norm(target - tcp) < 0.03 and state["gripper_opening"] > 0.8:
                self._phase("lower", steps)
        if self.phase == "lower":
            target = obj + grasp_offset
            target[2] = max(0.02, target[2])
            if np.linalg.norm(target - tcp) < 0.025:
                self._phase("grasp", steps)
        if self.phase == "grasp":
            target = obj + grasp_offset
            target[2] = max(0.02, target[2])
            if (state["gripper_command"] == "closed"
                    and state["gripper_opening"] < 0.75
                    and steps - self.phase_started >= 6):
                self._phase("lift", steps)
        if self.phase == "lift":
            if self.task_name == "assembly-v3":
                center = np.asarray(state["nut_center_xyz"], dtype=float)
                desired = np.array([center[0], center[1], goal[2] + 0.10])
                target = self._held_target(tcp, center, desired)
                lifted = center[2] >= goal[2] + 0.08
            else:
                lift_z = max(0.14, goal[2]) if self.task_name == "shelf-place-v3" else 0.14
                desired = np.array([obj[0], obj[1], lift_z])
                target = self._held_target(tcp, obj, desired)
                lifted = obj[2] >= (goal[2] - 0.03 if self.task_name == "shelf-place-v3" else 0.11)
            if lifted:
                next_phase = {
                    "shelf-place-v3": "align_x",
                    "bin-picking-v3": "carry",
                    "assembly-v3": "align_center",
                }[self.task_name]
                self._phase(next_phase, steps)
            elif np.linalg.norm(tcp - obj) > 0.10 and steps - self.phase_started > 18:
                self._phase("approach", steps)
                return self.update(state)

        if self.phase == "align_x":
            desired = np.array([goal[0], obj[1], goal[2]])
            target = self._held_target(tcp, obj, desired)
            if abs(obj[0] - goal[0]) < 0.025:
                self._phase("advance", steps)
        if self.phase == "advance":
            desired = goal.copy()
            target = self._held_target(tcp, obj, desired)

        if self.phase == "carry":
            desired = np.array([goal[0], goal[1], 0.14])
            target = self._held_target(tcp, obj, desired)
            if np.linalg.norm(obj[:2] - goal[:2]) < 0.025:
                self._phase("place", steps)
        if self.phase == "place":
            desired = np.array([goal[0], goal[1], 0.035])
            target = self._held_target(tcp, obj, desired)

        if self.phase == "align_center":
            center = np.asarray(state["nut_center_xyz"], dtype=float)
            desired = np.array([goal[0], goal[1], goal[2] + 0.10])
            target = self._held_target(tcp, center, desired)
            if np.linalg.norm(center[:2] - goal[:2]) < 0.015:
                self._phase("mount", steps)
        if self.phase == "mount":
            center = np.asarray(state["nut_center_xyz"], dtype=float)
            if np.linalg.norm(center[:2] - goal[:2]) > 0.025:
                self._phase("align_center", steps)
                return self.update(state)
            desired = np.array([goal[0], goal[1], goal[2] - 0.01])
            target = self._held_target(tcp, center, desired)

        gripper = "open" if self.phase in {"approach", "lower"} else "closed"
        target[2] = np.clip(target[2], 0.015, 0.40)
        delta = target - tcp
        waypoint = {
            "source": "hand_authored_task_waypoint",
            "phase": self.phase,
            "target_xyz": np.round(target, 4).tolist(),
            "delta_xyz": np.round(delta, 4).tolist(),
            "relative_direction": relative_text(delta, "Current waypoint"),
            "desired_gripper": gripper,
            "gripper_change_needed": state["gripper_command"] != gripper,
        }
        # Clarify lifting after grasp; other phases use the generic local goal.
        # Mentioning "keep open" during approach made JEV repeat grip_open.
        if self.phase == "lift":
            grasp_state = ("The object is already grasped." if state.get("both_fingers_touch_object")
                           else "The gripper is closed.")
            waypoint["instruction"] = (
                grasp_state + " Lift the object by moving the gripper UP to the current elevated "
                "waypoint while keeping the gripper closed."
            )
        return waypoint
