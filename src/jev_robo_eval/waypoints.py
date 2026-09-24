"""Optional hand-authored task decomposition; JEV still selects primitive actions.

This is an assisted control mode, not evidence of autonomous task planning.
All targets are gripper TCP positions in world coordinates.
"""

import numpy as np

from .metaworld_tasks import relative_text
from .hard_waypoints import HardWaypointGuide


class WaypointGuide:
    TASKS = {"push-v3", "door-open-v3", "pick-place-v3", "peg-insert-side-v3"} | HardWaypointGuide.TASKS

    def __init__(self, task_name):
        if task_name not in self.TASKS:
            raise ValueError(f"No waypoint guide for {task_name}")
        self.task_name = task_name
        self._delegate = HardWaypointGuide(task_name) if task_name in HardWaypointGuide.TASKS else None
        self.reset()

    def reset(self):
        if self._delegate is not None:
            self._delegate.reset()
        self.phase = "approach"
        self.phase_started = 0
        self.initial_z = None
        self._push_best_distance = None
        self._push_progress_step = 0
        self._lost_grip_since = None

    def _phase(self, name, steps):
        self.phase = name
        self.phase_started = steps
        if name == "push":
            self._push_best_distance = None
            self._push_progress_step = steps
        if name == "approach":
            self._lost_grip_since = None

    def update(self, state):
        if self._delegate is not None:
            waypoint = self._delegate.update(state)
            self.phase = waypoint["phase"]
            return waypoint
        tcp = np.asarray(state["control_xyz"])
        obj = np.asarray(state["object_xyz"])
        goal = np.asarray(state["goal_xyz"])
        steps = state["simulator_steps"]
        if self.initial_z is None:
            self.initial_z = obj[2]
        if self.task_name == "push-v3":
            direction = goal - obj
            direction[2] = 0
            direction /= max(np.linalg.norm(direction), 1e-6)
            behind = obj - direction * 0.045
            if self.phase == "approach":
                target = behind + [0, 0, 0.10]
                if np.linalg.norm(target - tcp) < 0.03:
                    self._phase("lower", steps)
            if self.phase == "lower":
                target = behind + [0, 0, 0.005]
                if np.linalg.norm(target - tcp) < 0.025:
                    self._phase("push", steps)
            if self.phase == "push":
                distance = float(np.linalg.norm(goal - obj))
                if self._push_best_distance is None or distance < self._push_best_distance - 0.003:
                    self._push_best_distance = distance
                    self._push_progress_step = steps
                ahead = float(np.dot(tcp[:2] - obj[:2], direction[:2]))
                lateral = float(np.linalg.norm(tcp[:2] - obj[:2] - ahead * direction[:2]))
                # Reacquire from behind if the TCP slips past the puck or the puck stalls.
                slipped = distance > 0.09 and (ahead > 0.02 or lateral > 0.055)
                if slipped or steps - self._push_progress_step >= 30:
                    self._phase("approach", steps)
                    return self.update(state)
                # Use a short lookahead while pushing, then aim near the goal for precision.
                target = (goal - direction * 0.02) if distance <= 0.09 else (obj + direction * 0.08)
                target[2] = obj[2] + 0.005
            gripper = "closed"
        elif self.task_name == "door-open-v3":
            hand_offset = tcp - np.asarray(state["robot"]["hand_body_xyz"])
            if self.phase == "approach":
                target = obj + [0.01, 0.02, 0.2] + hand_offset
                if np.linalg.norm(target - tcp) < 0.04:
                    self._phase("lower", steps)
            if self.phase == "lower":
                target = obj + [0.01, 0.02, 0] + hand_offset
                if np.linalg.norm(target - tcp) < 0.035:
                    self._phase("open_door", steps)
            if self.phase == "open_door":
                target = obj + [-0.05, 0, 0] + hand_offset
            gripper = "closed"
        else:
            if self.phase in {"carry", "align_tip", "insert"}:
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
            gripper = "open" if self.phase in {"approach", "lower"} else "closed"
            if self.phase == "approach":
                target = obj + [0, 0, 0.10]
                if np.linalg.norm(target - tcp) < 0.03 and state["gripper_opening"] > 0.8:
                    self._phase("lower", steps)
            if self.phase == "lower":
                target = obj.copy()
                target[2] = max(0.02, obj[2])
                if np.linalg.norm(target - tcp) < 0.025:
                    self._phase("grasp", steps)
                    gripper = "closed"
            if self.phase == "grasp":
                target = obj.copy()
                target[2] = max(0.02, obj[2])
                gripper = "closed"
                if state["gripper_command"] == "closed" and state["gripper_opening"] < 0.75 and steps - self.phase_started >= 6:
                    self._phase("lift", steps)
            if self.phase == "lift":
                target = np.array([obj[0], obj[1], max(goal[2], self.initial_z + 0.14)])
                if obj[2] - self.initial_z > 0.075:
                    self._phase("carry" if self.task_name == "pick-place-v3" else "align_tip", steps)
                elif np.linalg.norm(tcp - obj) > 0.10 and steps - self.phase_started > 18:
                    self._phase("approach", steps)
                    return self.update(state)
            if self.phase == "carry":
                target = goal + tcp - obj
            if self.phase in {"align_tip", "insert"}:
                head = np.asarray(state["peg_head_xyz"])
                hole = np.asarray(state["scene"]["sites_xyz"]["hole"])
                before_hole = np.array([hole[0] + 0.07, goal[1], goal[2]])
                if self.phase == "align_tip" and np.linalg.norm(head - before_hole) < 0.025:
                    self._phase("insert", steps)
                target = tcp + (before_hole if self.phase == "align_tip" else goal) - head
        # These four MT1 tasks use a practical TCP range below the hand's upper limit.
        target[2] = np.clip(target[2], 0.015, 0.40)
        delta = target - tcp
        return {
            "source": "hand_authored_task_waypoint",
            "phase": self.phase,
            "target_xyz": np.round(target, 4).tolist(),
            "delta_xyz": np.round(delta, 4).tolist(),
            "relative_direction": relative_text(delta, "Current waypoint"),
            "desired_gripper": gripper,
            "gripper_change_needed": state["gripper_command"] != gripper,
        }
