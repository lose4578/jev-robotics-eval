"""MetaWorld MT1 adapter. Importing this module does not load MetaWorld."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image

from .core import Action, Observation, Transition
from .metaworld_tasks import task_facts
from .waypoints import WaypointGuide
from .vision_annotations import annotate_frame
from .observation_access import filter_policy_state, resolve_access, validate_plan_only


_DIRECTION = {
    Action.X_POS: (1, 0, 0),
    Action.X_NEG: (-1, 0, 0),
    Action.Y_POS: (0, 1, 0),
    Action.Y_NEG: (0, -1, 0),
    Action.Z_POS: (0, 0, 1),
    Action.Z_NEG: (0, 0, -1),
}
_PROJECTION_STEP_M = 0.05


def _vec(values: Any) -> list[float]:
    return [round(float(x), 4) for x in values]


def _screen_direction(dx: float, dy: float) -> str:
    """Name the dominant displacement in image coordinates (Y points down)."""
    largest = max(abs(dx), abs(dy))
    if largest < 0.5:
        return "negligible"
    horizontal = "right" if dx > 0 else "left" if abs(dx) >= 0.4 * largest else ""
    vertical = "down" if dy > 0 else "up" if abs(dy) >= 0.4 * largest else ""
    if abs(dx) < 0.4 * largest:
        horizontal = ""
    if abs(dy) < 0.4 * largest:
        vertical = ""
    return "-".join(part for part in (vertical, horizontal) if part)


class MetaWorldMT1:
    """One MT1 task instance; action repeat trades decision cost for control precision."""

    def __init__(
        self,
        task_name: str,
        *,
        mode: str = "text",
        seed: int = 0,
        task_index: int = 0,
        action_repeat: int = 3,
        image_size: int = 256,
        camera_name: str = "corner2",
        move_scale: float = 1.0,
        guidance: str | None = None,
        annotate_vision: bool = False,
        information: str = "privileged",
        proprio_projection: bool = False,
        privilege_level: int | None = None,
        plan_only: bool = False,
    ) -> None:
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if mode not in {"text", "vision"}:
            raise ValueError("mode must be text or vision")
        if image_size < 32:
            raise ValueError("image_size must be >= 32")
        if not 0 < move_scale <= 1:
            raise ValueError("move_scale must be in (0, 1]")
        information, guidance, privilege_level = resolve_access(information, guidance, privilege_level)
        validate_plan_only(privilege_level, plan_only)
        self.plan_only = plan_only
        if information == "nonprivileged" and mode != "vision":
            raise ValueError("nonprivileged observation requires mode='vision' to perceive task objects")
        if information == "nonprivileged" and guidance == "waypoints":
            raise ValueError("nonprivileged observation cannot use waypoint guidance")
        if privilege_level < 3 and annotate_vision:
            raise ValueError("oracle vision annotation requires privilege_level=3")
        import metaworld

        self.mode = mode
        self.information = information
        self.privilege_level = privilege_level
        self.proprio_projection = proprio_projection
        self.task_name = task_name
        self.action_repeat = action_repeat
        self.move_scale = move_scale
        self.annotate_vision = annotate_vision
        self.guide = WaypointGuide(task_name) if guidance == "waypoints" else None
        self.benchmark = metaworld.MT1(task_name, seed=seed)
        if task_name not in self.benchmark.train_classes:
            raise ValueError(f"Unknown MT1 task: {task_name}")
        if not 0 <= task_index < len(self.benchmark.train_tasks):
            raise ValueError(f"task_index must be 0..{len(self.benchmark.train_tasks) - 1}")
        env_class = self.benchmark.train_classes[task_name]
        kwargs: dict[str, Any] = {}
        if mode == "vision":
            kwargs = dict(render_mode="rgb_array", width=image_size, height=image_size, camera_name=camera_name)
        self.env = env_class(**kwargs)
        self.env.set_task(self.benchmark.train_tasks[task_index])
        self.camera_name = camera_name
        self.image_size = image_size
        if proprio_projection:
            camera_id = self.env.model.camera(camera_name).id
            if int(self.env.model.cam_bodyid[camera_id]) != 0 or int(self.env.model.cam_mode[camera_id]) != 0:
                self.env.close()
                raise ValueError("proprio_projection requires a fixed world camera")
            self._projection_camera_id = camera_id
        self._arm_joint_names = [f"right_j{i}" for i in range(7)]
        robot_root = self.env.model.body("base").id

        def robot_body(body_id: int) -> bool:
            while body_id != 0:
                if body_id == robot_root:
                    return True
                body_id = int(self.env.model.body_parentid[body_id])
            return False

        # Keep task objects and fixtures, excluding Sawyer's links and its
        # mocap controller. Body origins and interaction points are distinct.
        self._scene_body_ids = [
            i for i in range(1, self.env.model.nbody)
            if not robot_body(i) and self.env.model.body(i).name != "mocap"
        ]
        self._scene_site_ids = [
            i for i in range(self.env.model.nsite)
            if self.env.model.site(i).name
            and self.env.model.site(i).name != "goal"
            and not robot_body(int(self.env.model.site_bodyid[i]))
            and self.env.model.body(int(self.env.model.site_bodyid[i])).name != "mocap"
        ]
        self._gripper = -1.0  # MetaWorld: -1 opens, +1 closes.
        self._raw_obs: np.ndarray | None = None
        self._step_count = 0
        self._last_info: dict[str, Any] = {}
        self._initial_object = np.zeros(3)

    def _render_nonprivileged(self) -> Image.Image:
        """Temporarily remove simulator site markers from the RGB observation."""
        model = self.env.model
        # Sites are nonphysical MuJoCo markers, including goal, pegTop and
        # pegHead; their names vary by task and may reveal oracle positions.
        alpha = model.site_rgba[:, 3].copy()
        try:
            model.site_rgba[:, 3] = 0.0
            return Image.fromarray(self.env.render()).convert("RGB")
        finally:
            model.site_rgba[:, 3] = alpha

    def _action_screen_directions(self, control_xyz: np.ndarray) -> dict[str, dict[str, Any]]:
        """Project a local TCP motion through a fixed camera, without task truth."""
        camera_id = self._projection_camera_id
        rotation = self.env.data.cam_xmat[camera_id].reshape(3, 3)
        camera_xyz = self.env.data.cam_xpos[camera_id]
        focal = self.image_size / (2 * np.tan(np.deg2rad(self.env.model.cam_fovy[camera_id]) / 2))

        def project(point: np.ndarray) -> np.ndarray | None:
            local = rotation.T @ (point - camera_xyz)
            if local[2] >= -1e-6 or not np.all(np.isfinite(local)):
                return None
            pixel = np.array([
                self.image_size / 2 + focal * local[0] / -local[2],
                self.image_size / 2 - focal * local[1] / -local[2],
            ])
            if self.camera_name == "corner2":
                pixel = self.image_size - 1 - pixel
            return pixel

        origin = project(control_xyz)
        directions: dict[str, dict[str, Any]] = {}
        for action, axis in _DIRECTION.items():
            endpoint = project(control_xyz + _PROJECTION_STEP_M * np.asarray(axis))
            if origin is None or endpoint is None:
                directions[action.value] = {
                    "sample_distance_m": _PROJECTION_STEP_M,
                    "delta_px": None,
                    "screen_direction": "unavailable",
                }
                continue
            dx, dy = (endpoint - origin).tolist()
            directions[action.value] = {
                "sample_distance_m": _PROJECTION_STEP_M,
                "delta_px": [round(dx, 2), round(dy, 2)],
                "screen_direction": _screen_direction(dx, dy),
            }
        return directions

    def _observation(self) -> Observation:
        assert self._raw_obs is not None
        obs = self._raw_obs
        # Reach success is measured at the finger midpoint, not obs[:3]
        # (the hand body origin). These points have different Z coordinates.
        control_xyz = self.env.tcp_center.copy()
        robot = {
            "hand_body_xyz": _vec(obs[:3]),
            "hand_quat_wxyz": _vec(self.env.data.body("hand").xquat),
            "arm_joint_order": list(self._arm_joint_names),
            "arm_joint_position_rad": _vec([self.env.data.joint(n).qpos[0] for n in self._arm_joint_names]),
            "arm_joint_velocity_rad_s": _vec([self.env.data.joint(n).qvel[0] for n in self._arm_joint_names]),
        }
        evaluation_state: dict[str, Any] = {
            "task_name": self.task_name,
            "coordinate_frame": "MetaWorld world XYZ; actions move end effector along one named axis",
            "control_point": "center between the two gripper fingers",
            "control_xyz": _vec(control_xyz),
            "nominal_motion_step_m": float(self.env.action_scale * self.move_scale * self.action_repeat),
            "gripper_opening": round(float(obs[3]), 4),
            "goal_xyz": _vec(obs[-3:]),
            "goal_minus_control_xyz": _vec(obs[-3:] - control_xyz),
            "control_to_goal_distance": round(float(np.linalg.norm(obs[-3:] - control_xyz)), 5),
            "gripper_command": "closed" if self._gripper > 0 else "open",
            "simulator_steps": self._step_count,
            "robot": robot,
            "information": self.information,
            "privilege_level": self.privilege_level,
            "scene": {
                "interaction_points_xyz": [_vec(p) for p in np.asarray(self.env._get_pos_objects()).reshape(-1, 3)],
                "body_origins_xyz": {self.env.model.body(i).name or f"body_{i}": _vec(self.env.data.xpos[i]) for i in self._scene_body_ids},
                "sites_xyz": {self.env.model.site(i).name: _vec(self.env.data.site_xpos[i]) for i in self._scene_site_ids},
            },
        }
        evaluation_state.update(task_facts(self.env, self.task_name, obs, control_xyz, self._initial_object))
        # L2 may use the interaction-point offset for assembly transfer: the
        # desired TCP is the peg goal translated by the current gripper-to-nut
        # center offset.  It is exposed only after the privileged contact
        # signal is available; L1 must solve assembly without this helper.
        if (self.task_name == "assembly-v3" and self.privilege_level >= 2
                and evaluation_state.get("both_fingers_touch_object")
                and "nut_center_xyz" in evaluation_state):
            goal = np.asarray(evaluation_state["goal_xyz"], dtype=float)
            nut = np.asarray(evaluation_state["nut_center_xyz"], dtype=float)
            evaluation_state["atomic_target_xyz"] = _vec(goal + control_xyz - nut)
        if self.task_name != "reach-v3":
            del evaluation_state["goal_minus_control_xyz"]
        if self.guide is not None:
            evaluation_state["active_waypoint"] = self.guide.update(evaluation_state)
        if self.information == "nonprivileged":
            # Construct a strict allowlist independently of the evaluation
            # dictionary so future task facts cannot enter a policy request.
            state = {
                "task_name": self.task_name,
                "coordinate_frame": "MetaWorld world XYZ; actions move end effector along one named axis",
                "control_point": "center between the two gripper fingers",
                "control_xyz": _vec(control_xyz),
                "nominal_motion_step_m": evaluation_state["nominal_motion_step_m"],
                "robot": {
                    "hand_body_xyz": list(robot["hand_body_xyz"]),
                    "hand_quat_wxyz": list(robot["hand_quat_wxyz"]),
                    "arm_joint_order": list(robot["arm_joint_order"]),
                    "arm_joint_position_rad": list(robot["arm_joint_position_rad"]),
                    "arm_joint_velocity_rad_s": list(robot["arm_joint_velocity_rad_s"]),
                },
                "gripper_opening": round(float(obs[3]), 4),
                "gripper_command": "closed" if self._gripper > 0 else "open",
                "simulator_steps": self._step_count,
                "information": self.information,
                "privilege_level": self.privilege_level,
            }
        else:
            state = filter_policy_state(evaluation_state, self.privilege_level, plan_only=self.plan_only)
        if self.proprio_projection:
            state["camera_name"] = self.camera_name
            state["action_screen_directions"] = self._action_screen_directions(control_xyz)
            evaluation_state["camera_name"] = self.camera_name
            evaluation_state["action_screen_directions"] = deepcopy(state["action_screen_directions"])
            evaluation_state["proprio_projection"] = True
            evaluation_state["projection_step_m"] = _PROJECTION_STEP_M
        frame = None
        if self.mode == "vision":
            frame = self._render_nonprivileged() if self.information == "nonprivileged" else Image.fromarray(self.env.render()).convert("RGB")
            if self.camera_name == "corner2":
                frame = frame.transpose(Image.Transpose.ROTATE_180)
            if self.annotate_vision:
                evaluation_state["image_annotations"] = "Cyan TCP marks the gripper center; orange WAYPOINT marks the current target, projected in world coordinates."
                state["image_annotations"] = evaluation_state["image_annotations"]
                frame = annotate_frame(frame, self.env, self.camera_name, evaluation_state)
        return Observation(state=state, image=frame, evaluation_state=evaluation_state)

    def reset(self, *, seed: int | None = None) -> Observation:
        self._gripper = -1.0
        self._step_count = 0
        self._raw_obs, self._last_info = self.env.reset(seed=seed)
        self._initial_object = self._raw_obs[4:7].copy()
        if self.guide is not None:
            self.guide.reset()
        return self._observation()

    def step(self, action: Action, *, scale: float = 1.0) -> Transition:
        if isinstance(scale, bool) or not np.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("scale must be finite and in (0, 1]")
        if self._raw_obs is None:
            raise RuntimeError("Call reset before step")
        if action == Action.GRIP_OPEN:
            self._gripper = -1.0
        elif action == Action.GRIP_CLOSE:
            self._gripper = 1.0
        xyz = _DIRECTION.get(action, (0, 0, 0))
        command = np.asarray((*xyz, self._gripper), dtype=np.float32)
        command[:3] *= self.move_scale * scale
        reward_total = 0.0
        terminated = truncated = success = False
        info: dict[str, Any] = {}
        for _ in range(self.action_repeat):
            self._raw_obs, reward, terminated, truncated, info = self.env.step(command)
            self._step_count += 1
            reward_total += float(reward)
            success = success or bool(info.get("success", False))
            if terminated or truncated or success:
                break
        self._last_info = info
        return Transition(
            observation=self._observation(),
            reward=reward_total,
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=success,
            info=info,
        )

    def close(self) -> None:
        self.env.close()
