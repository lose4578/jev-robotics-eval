"""RoboTwin adapter using the shared nine-action robot environment contract.

RoboTwin is imported only on reset so MetaWorld-only installations can import
the rest of this package. All coordinates are RoboTwin world XYZ, in meters.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import importlib
import os
import sys
from typing import Any

import numpy as np
from PIL import Image

from .core import Action, Observation, Transition
from .observation_access import filter_policy_state, validate_plan_only
from .robotwin_tasks import TASKS
from .task_registry import TASK_REGISTRY


_AXES = {
    Action.X_POS: (1, 0, 0), Action.X_NEG: (-1, 0, 0),
    Action.Y_POS: (0, 1, 0), Action.Y_NEG: (0, -1, 0),
    Action.Z_POS: (0, 0, 1), Action.Z_NEG: (0, 0, -1),
}
_PROJECTION_STEP_M = 0.05
_TOPDOWN_QUAT_WXYZ = np.array([-0.5, 0.5, -0.5, -0.5])
_TOPDOWN_TCP_LEFT_XYZ = np.array([-0.30, -0.19, 0.98])


def _vec(x: Any) -> list[float]:
    return [round(float(v), 4) for v in x]


def _quat_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=float) / np.linalg.norm(quat_wxyz)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _project_cv(point_xyz: np.ndarray, intrinsic: np.ndarray,
                extrinsic: np.ndarray, image_size: tuple[int, int],
                raw_size: tuple[int, int]) -> np.ndarray | None:
    """Project world XYZ through RoboTwin's actual OpenCV camera matrices."""
    point = np.asarray([*point_xyz, 1.0], dtype=float)
    camera = np.asarray(extrinsic, dtype=float) @ point
    if not np.all(np.isfinite(camera)) or camera[2] <= 1e-6:
        return None
    pixel = np.asarray(intrinsic, dtype=float) @ camera[:3]
    pixel = pixel[:2] / pixel[2]
    return pixel * np.asarray(image_size, dtype=float) / np.asarray(raw_size, dtype=float)


def _screen_direction(dx: float, dy: float) -> str:
    length = max(abs(dx), abs(dy))
    if length < 0.5:
        return "negligible"
    parts = []
    if abs(dy) >= 0.4 * length:
        parts.append("down" if dy > 0 else "up")
    if abs(dx) >= 0.4 * length:
        parts.append("right" if dx > 0 else "left")
    return "-".join(parts)


@contextmanager
def _robotwin_context(root: Path):
    """Upstream object assets use paths relative to the RoboTwin checkout."""
    old_cwd = Path.cwd()
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            sys.path.remove(str(root))


class RoboTwinEnvironment:
    """Fixed-arm RoboTwin control; no object truth is used to select the arm."""

    def __init__(
        self, task_name: str, *, root: str | Path | None = None,
        mode: str = "vision", seed: int = 2, privilege_level: int = 0,
        image_size: int = 384, action_repeat: int = 3,
        move_scale: float = 0.5, task_config: str = "demo_clean",
        active_arm: str = "left", max_steps: int = 500,
        proprio_projection: bool = False, initial_pose: str = "home",
        plan_only: bool = False,
    ) -> None:
        if task_name not in TASKS:
            raise ValueError(f"Unsupported RoboTwin task: {task_name}")
        if mode not in {"text", "vision"}:
            raise ValueError("mode must be text or vision")
        if privilege_level not in {0, 1, 2, 3}:
            raise ValueError("privilege_level must be 0, 1, 2, or 3")
        if privilege_level == 3 and not TASK_REGISTRY[task_name].oracle_supported:
            raise ValueError(f"No L3 oracle guide for {task_name}; use L0/L1/L2")
        if active_arm not in {"left", "right"}:
            raise ValueError("active_arm must be left or right")
        if not task_config.isidentifier():
            raise ValueError("task_config must be a simple config name")
        if image_size < 32 or action_repeat < 1 or max_steps < 1:
            raise ValueError("image_size >= 32, action_repeat >= 1 and max_steps >= 1 are required")
        if not 0 < move_scale <= 1:
            raise ValueError("move_scale must be in (0, 1]")
        if initial_pose not in {"home", "topdown"}:
            raise ValueError("initial_pose must be home or topdown")
        validate_plan_only(privilege_level, plan_only)
        self.plan_only = plan_only
        root = root if root is not None else os.environ.get("ROBOTWIN_ROOT", str(Path.home() / "code" / "RoboTwin"))
        self.task_name, self.root, self.mode = task_name, Path(root).expanduser().resolve(), mode
        self.seed, self.privilege_level = seed, privilege_level
        self.image_size, self.action_repeat, self.move_scale = image_size, action_repeat, move_scale
        self.task_config, self.active_arm, self.max_steps = task_config, active_arm, max_steps
        self.proprio_projection = proprio_projection
        self.initial_pose = initial_pose
        self._initialization: dict[str, Any] = {"mode": initial_pose, "uses_object_truth": False}
        self.task: Any = None
        self._arm_tag: Any = None
        self._simulator_steps = 0
        self._control_steps = 0
        self._gripper_command = "open"
        self._success = False
        self._initial_object_z: float | None = None
        self._guide_phase: str | None = None
        self._guide_phase_enter = 0
        self._guide_lift_target: np.ndarray | None = None

    def _config(self) -> dict[str, Any]:
        import yaml

        config_file = self.root / "env_cfg" / "task_config" / f"{self.task_config}.yml"
        cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        embodiment_map = yaml.safe_load((self.root / "env_cfg/task_config/_embodiment_config.yml").read_text(encoding="utf-8"))
        embodiments = cfg["embodiment"]
        if len(embodiments) != 1:
            raise ValueError("RoboTwin adapter currently requires one shared dual-arm embodiment")
        robot_file = (self.root / embodiment_map[embodiments[0]]["file_path"]).resolve()
        robot_cfg = yaml.safe_load((robot_file / "config.yml").read_text(encoding="utf-8"))
        cfg.update(
            left_robot_file=str(robot_file), right_robot_file=str(robot_file),
            left_embodiment_config=robot_cfg, right_embodiment_config=robot_cfg,
            dual_arm_embodied=True, embodiment_name=embodiments[0], need_plan=True,
            save_data=False, save_freq=None, render_freq=0,
        )
        return cfg

    def _count_physics(self) -> None:
        original_step = self.task.scene.step

        def counted_step(*args: Any, **kwargs: Any) -> Any:
            result = original_step(*args, **kwargs)
            self._simulator_steps += 1
            # Brief contact can disappear before a planned motion finishes.
            # Keep the simulator's own success verdict outside policy state.
            if not self._success:
                self._success = bool(self.task.check_success())
            return result

        self.task.scene.step = counted_step

    def _initialize_topdown(self) -> None:
        """Move to a fixed safe robot pose; the target never uses scene objects."""
        ee = np.asarray(self.task.get_arm_pose(self._arm_tag), dtype=float)
        tcp = self._tcp()
        local_ee_to_tcp = _quat_rotation(ee[3:]).T @ (tcp - ee[:3])
        target_tcp = _TOPDOWN_TCP_LEFT_XYZ.copy()
        if self.active_arm == "right":
            target_tcp[0] *= -1
        target_ee = target_tcp - _quat_rotation(_TOPDOWN_QUAT_WXYZ) @ local_ee_to_tcp
        target_pose = np.r_[target_ee, _TOPDOWN_QUAT_WXYZ]
        passive = (self.task.robot.get_right_tcp_pose if self.active_arm == "left"
                   else self.task.robot.get_left_tcp_pose)
        passive_before = np.asarray(passive()[:3], dtype=float)
        self.task.plan_success = True
        succeeded = bool(self.task.move(self.task.move_to_pose(self._arm_tag, target_pose)))
        actual_tcp = self._tcp()
        passive_drift = float(np.linalg.norm(np.asarray(passive()[:3]) - passive_before))
        self._initialization = {
            "mode": "topdown", "uses_object_truth": False,
            "target_tcp_xyz": _vec(target_tcp), "actual_tcp_xyz": _vec(actual_tcp),
            "target_ee_xyz": _vec(target_ee), "target_ee_quat_wxyz": _vec(_TOPDOWN_QUAT_WXYZ),
            "passive_tcp_drift_m": round(passive_drift, 7),
            "physics_steps": self._simulator_steps,
            "plan_success": succeeded,
        }
        if not succeeded or np.linalg.norm(actual_tcp - target_tcp) > 0.03:
            raise RuntimeError(f"Failed to initialize fixed topdown pose: {self._initialization}")
        if self._success:
            raise RuntimeError("Fixed topdown initialization unexpectedly completed the task")

    def _tcp(self) -> np.ndarray:
        get_pose = (self.task.robot.get_left_tcp_pose if self.active_arm == "left"
                    else self.task.robot.get_right_tcp_pose)
        return np.asarray(get_pose()[:3], dtype=float)

    def _projection(self, camera_obs: dict[str, Any], tcp: np.ndarray,
                    image_size: tuple[int, int], raw_size: tuple[int, int]
                    ) -> tuple[list[float] | None, dict[str, dict[str, Any]]]:
        intrinsic = np.asarray(camera_obs["intrinsic_cv"], dtype=float)
        extrinsic = np.asarray(camera_obs["extrinsic_cv"], dtype=float)
        if intrinsic.shape != (3, 3) or extrinsic.shape not in {(3, 4), (4, 4)}:
            raise ValueError("RoboTwin head camera has invalid OpenCV calibration matrices")

        def project(point: np.ndarray) -> np.ndarray | None:
            return _project_cv(point, intrinsic, extrinsic, image_size, raw_size)

        origin = project(tcp)
        actions: dict[str, dict[str, Any]] = {}
        for action, axis in _AXES.items():
            end = project(tcp + _PROJECTION_STEP_M * np.asarray(axis, dtype=float))
            if origin is None or end is None:
                actions[action.value] = {
                    "sample_distance_m": _PROJECTION_STEP_M,
                    "delta_px": None, "screen_direction": "unavailable",
                }
                continue
            dx, dy = (end - origin).tolist()
            actions[action.value] = {
                "sample_distance_m": _PROJECTION_STEP_M,
                "delta_px": [round(dx, 2), round(dy, 2)],
                "screen_direction": _screen_direction(dx, dy),
            }
        return (None if origin is None else [round(float(v), 2) for v in origin], actions)

    def _task_points(self) -> tuple[np.ndarray, np.ndarray]:
        spec = TASKS[self.task_name]
        actor = getattr(self.task, spec.actor)
        object_xyz = np.asarray(actor.get_pose().p, dtype=float)
        if spec.goal_actor:
            goal_actor = getattr(self.task, spec.goal_actor)
            goal_xyz = np.asarray(goal_actor.get_functional_point(spec.goal_functional_point)[:3]
                                  if spec.goal_functional_point is not None else goal_actor.get_pose().p,
                                  dtype=float)
        else:
            goal_xyz = np.asarray(actor.get_contact_point(spec.contact_point)[:3], dtype=float)
        return object_xyz, goal_xyz

    def _robot_state(self) -> dict[str, Any]:
        robot = self.task.robot
        entity = robot.left_entity if self.active_arm == "left" else robot.right_entity
        joints = robot.left_arm_joints if self.active_arm == "left" else robot.right_arm_joints
        joint_indexes = [entity.get_active_joints().index(joint) for joint in joints]
        pose = np.asarray(robot.get_left_tcp_pose() if self.active_arm == "left"
                          else robot.get_right_tcp_pose(), dtype=float)
        return {
            "hand_body_xyz": _vec(pose[:3]),
            "hand_quat_wxyz": _vec(pose[3:]),
            "arm_joint_order": [joint.get_name() for joint in joints],
            "arm_joint_position_rad": _vec(entity.get_qpos()[joint_indexes]),
            "arm_joint_velocity_rad_s": _vec(entity.get_qvel()[joint_indexes]),
        }

    def _both_fingers_touch_object(self) -> bool:
        if self.task_name != "move_pillbottle_pad":
            return False
        robot = self.task.robot
        fingers = robot.left_gripper if self.active_arm == "left" else robot.right_gripper
        names = {joint.child_link.get_name() for joint, _, _ in fingers}
        if len(names) < 2:
            return False
        object_name = getattr(self.task, TASKS[self.task_name].actor).get_name()
        touching: set[str] = set()
        for contact in self.task.scene.get_contacts():
            a, b = (body.entity.name for body in contact.bodies)
            if a == object_name and b in names and contact.points:
                touching.add(b)
            elif b == object_name and a in names and contact.points:
                touching.add(a)
        return names <= touching

    def _phase(self, phase: str, tcp: np.ndarray) -> None:
        if phase != self._guide_phase:
            self._guide_phase = phase
            self._guide_phase_enter = self._control_steps
            if phase == "lift":
                self._guide_lift_target = tcp + [0, 0, 0.10]

    def _waypoint(self, tcp: np.ndarray, obj: np.ndarray, goal: np.ndarray, opening: float) -> dict[str, Any]:
        """Simple hand-authored guidance, exposed only at privilege level 3."""
        if self.task_name in {"click_bell", "press_stapler"}:
            if opening > 0.2:
                self._phase("close", tcp)
            elif self._guide_phase in {None, "close"}:
                self._phase("approach", tcp)
            if self._guide_phase == "approach" and (np.linalg.norm(tcp[:2] - goal[:2]) < 0.025
                                                   and tcp[2] >= goal[2] + 0.055):
                self._phase("press", tcp)
            phase = self._guide_phase
            if phase == "close":
                target, grip = tcp, "closed"
            elif phase == "approach":
                target, grip = goal + [0, 0, 0.08], "closed"
            else:
                target, grip = goal + [0, 0, 0.005], "closed"
        else:
            if self._guide_phase is None:
                self._phase("approach", tcp)
            if self._guide_phase == "approach" and np.linalg.norm((tcp - obj)[:2]) < 0.025:
                self._phase("lower", tcp)
            if self._guide_phase == "lower" and np.linalg.norm(tcp - (obj + [0, 0, 0.02])) < 0.03:
                self._phase("grasp", tcp)
            if self._guide_phase == "grasp" and self._gripper_command == "closed" \
                    and self._control_steps > self._guide_phase_enter:
                self._phase("lift", tcp)
            if self._guide_phase == "lift" and (obj[2] - self._initial_object_z > 0.04):
                self._phase("carry", tcp)
            if self._guide_phase == "lift" and self._guide_lift_target is not None \
                    and tcp[2] >= self._guide_lift_target[2] - 0.015 \
                    and obj[2] - self._initial_object_z < 0.02:
                self._phase("retry", tcp)
            if self._guide_phase == "retry" and opening > 0.8:
                self._phase("approach", tcp)
            if self._guide_phase == "carry" and np.linalg.norm((goal - obj)[:2]) < 0.03:
                self._phase("place", tcp)
            if self._guide_phase == "place" and obj[2] < self._initial_object_z + 0.015:
                self._phase("release", tcp)
            phase = self._guide_phase
            if phase in {"approach", "retry"}:
                target, grip = obj + [0, 0, 0.10], "open"
            elif phase == "lower":
                target, grip = obj + [0, 0, 0.02], "open"
            elif phase == "grasp":
                target, grip = tcp, "closed"
            elif phase == "lift":
                target, grip = self._guide_lift_target, "closed"
            elif phase == "carry":
                desired = np.array([goal[0], goal[1], max(obj[2], self._initial_object_z + 0.10)])
                target, grip = tcp + desired - obj, "closed"
            elif phase == "place":
                desired = np.array([goal[0], goal[1], self._initial_object_z])
                target, grip = tcp + desired - obj, "closed"
            else:
                target, grip = tcp, "open"
        delta = target - tcp
        directions = [f"{abs(float(v))*100:.1f} cm {'+' if v >= 0 else '-'}{axis}"
                      for v, axis in zip(delta, "XYZ")]
        return {
            "phase": phase, "target_xyz": _vec(target), "delta_xyz": _vec(delta),
            "relative_direction": ", ".join(directions),
            "instruction": f"{phase}: {TASKS[self.task_name].instruction}",
            "desired_gripper": grip,
            "gripper_change_needed": self._gripper_command != grip,
        }

    def _observation(self) -> Observation:
        raw = self.task.get_obs()
        camera_obs = raw["observation"]["head_camera"]
        rgb = np.asarray(camera_obs["rgb"])
        image = Image.fromarray(rgb).convert("RGB")
        scale = self.image_size / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.LANCZOS)
        tcp = self._tcp()
        obj, goal = self._task_points()
        robot = self._robot_state()
        opening = float(self.task.robot.get_left_gripper_val() if self.active_arm == "left"
                        else self.task.robot.get_right_gripper_val())
        info = "nonprivileged" if self.privilege_level == 0 else "privileged"
        full: dict[str, Any] = {
            "environment": "robotwin", "task_name": self.task_name,
            "coordinate_frame": "RoboTwin world XYZ, meters; +Z is upward",
            "control_point": f"{self.active_arm} gripper TCP",
            "active_arm": self.active_arm, "control_xyz": _vec(tcp),
            "robot": robot, "gripper_opening": round(opening, 4),
            "gripper_command": self._gripper_command,
            "simulator_steps": self._simulator_steps, "control_steps": self._control_steps,
            "information": info, "privilege_level": self.privilege_level,
            "object_xyz": _vec(obj), "goal_xyz": _vec(goal),
            "scene": {"object": TASKS[self.task_name].actor,
                      "goal": TASKS[self.task_name].goal_actor or "object contact point"},
            "object_minus_control_xyz": _vec(obj - tcp),
            "goal_minus_control_xyz": _vec(goal - tcp),
            "control_to_goal_distance": round(float(np.linalg.norm(goal - tcp)), 4),
            "target_relative_to_gripper": ", ".join(
                f"{abs(float(v))*100:.1f} cm {'+' if v >= 0 else '-'}{axis}"
                for v, axis in zip(goal - tcp, "XYZ")),
            "goal_minus_object_xyz": _vec(goal - obj),
            "tcp_to_object_m": round(float(np.linalg.norm(obj - tcp)), 4),
            "object_lift_m": round(float(obj[2] - (self._initial_object_z or obj[2])), 4),
            "success": self._success,
            "initialization": self._initialization,
        }
        if self.task_name == "move_pillbottle_pad":
            full["both_fingers_touch_object"] = self._both_fingers_touch_object()
        if self.privilege_level == 3:
            full["active_waypoint"] = self._waypoint(tcp, obj, goal, opening)
        if self.proprio_projection:
            tcp_pixel, directions = self._projection(camera_obs, tcp, image.size,
                                                       (rgb.shape[1], rgb.shape[0]))
            full["camera_name"] = "head_camera"
            full["tcp_pixel"] = tcp_pixel
            full["action_screen_directions"] = directions
            full["proprio_projection"] = True
            full["projection_step_m"] = _PROJECTION_STEP_M
        state = filter_policy_state(full, self.privilege_level, plan_only=self.plan_only)
        # Every mode exposes a real frame for recording. Text policies ignore it.
        return Observation(state=state, image=image, evaluation_state=full)

    def reset(self, *, seed: int | None = None) -> Observation:
        self.close()
        self._simulator_steps = self._control_steps = 0
        self._gripper_command, self._success = "open", False
        self._initial_object_z = None
        self._guide_phase = None
        self._guide_lift_target = None
        self._initialization = {"mode": self.initial_pose, "uses_object_truth": False}
        episode_seed = self.seed if seed is None else seed
        if not self.root.is_dir():
            raise FileNotFoundError(f"RoboTwin checkout not found: {self.root}")
        cfg = self._config()
        with _robotwin_context(self.root):
            task_module = importlib.import_module(f"envs.{self.task_name}")
            from envs.utils import ArmTag
            task = getattr(task_module, self.task_name)()
            try:
                task.setup_demo(task_name=self.task_name, seed=episode_seed, **cfg)
            except Exception:
                task.close_env()
                raise
        self.task, self._arm_tag = task, ArmTag(self.active_arm)
        self._count_physics()
        if self.initial_pose == "topdown":
            try:
                self._initialize_topdown()
            except Exception:
                self.close()
                raise
        self._initial_object_z = float(self._task_points()[0][2])
        self._success = bool(self.task.check_success())
        return self._observation()

    def step(self, action: Action, *, scale: float = 1.0) -> Transition:
        if isinstance(scale, bool) or not np.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("scale must be finite and in (0, 1]")
        if self.task is None:
            raise RuntimeError("Call reset before step")
        action = Action(action)
        if self._control_steps >= self.max_steps or self._success:
            raise RuntimeError("Episode has ended; call reset")
        self._control_steps += 1
        task, arm = self.task, self._arm_tag
        command_ok = True
        attempts = self.action_repeat if action in _AXES else 1
        for _ in range(attempts):
            if action in _AXES:
                pose = np.asarray(task.get_arm_pose(arm), dtype=float)
                pose[:3] += np.asarray(_AXES[action]) * (0.02 * self.move_scale * scale)
                task.plan_success = True
                command_ok = bool(task.move(task.move_to_pose(arm, pose)))
            elif action == Action.GRIP_OPEN:
                task.plan_success = True
                command_ok = bool(task.move(task.open_gripper(arm)))
                self._gripper_command = "open"
            elif action == Action.GRIP_CLOSE:
                task.plan_success = True
                command_ok = bool(task.move(task.close_gripper(arm)))
                self._gripper_command = "closed"
            else:
                task.scene.step()
            self._success = bool(task.check_success()) or self._success
            if not command_ok or self._success:
                break
        truncated = self._control_steps >= self.max_steps and not self._success
        info = {"plan_success": command_ok, "active_arm": self.active_arm,
                "physics_steps": self._simulator_steps}
        return Transition(self._observation(), float(self._success), self._success,
                          truncated, self._success, info)

    def close(self) -> None:
        if self.task is not None:
            self.task.close_env()
            self.task = None
