"""Fast contract checks that do not load the large RoboTwin simulator."""

from types import SimpleNamespace

import numpy as np
import pytest

from jev_robo_eval.robotwin_env import (
    RoboTwinEnvironment, _project_cv, _quat_rotation, _TOPDOWN_QUAT_WXYZ,
)
from jev_robo_eval.robotwin_tasks import TASKS


def test_robotwin_import_and_invalid_options_do_not_load_simulator():
    with pytest.raises(ValueError, match="Unsupported RoboTwin task"):
        RoboTwinEnvironment("not_a_task")
    assert RoboTwinEnvironment("click_bell", proprio_projection=True).proprio_projection
    with pytest.raises(ValueError, match="initial_pose"):
        RoboTwinEnvironment("click_bell", initial_pose="automatic")


def test_fixed_topdown_tool_axis_points_with_gravity():
    rotation = _quat_rotation(_TOPDOWN_QUAT_WXYZ)
    assert np.allclose(rotation @ [1, 0, 0], [0, 0, -1])


def test_projection_uses_cv_depth_and_preserved_aspect_scaling():
    intrinsic = np.array([[100, 0, 160], [0, 100, 120], [0, 0, 1]])
    extrinsic = np.eye(4)
    center = _project_cv(np.array([0, 0, 1]), intrinsic, extrinsic, (384, 288), (320, 240))
    right = _project_cv(np.array([0.05, 0, 1]), intrinsic, extrinsic, (384, 288), (320, 240))
    assert center.tolist() == [192, 144]
    assert np.allclose(right - center, [6, 0])
    assert _project_cv(np.array([0, 0, -1]), intrinsic, extrinsic, (384, 288), (320, 240)) is None


def test_axis_directions_depend_only_on_tcp_and_camera():
    env = RoboTwinEnvironment("click_bell", proprio_projection=True)
    camera = {"intrinsic_cv": np.array([[100, 0, 160], [0, 100, 120], [0, 0, 1]]),
              "extrinsic_cv": np.eye(4)}
    pixel, directions = env._projection(camera, np.array([0, 0, 1]), (384, 288), (320, 240))
    assert pixel == [192, 144]
    assert directions["x_pos"]["delta_px"] == [6, 0]
    assert directions["y_neg"]["screen_direction"] == "up"
    assert directions["z_pos"]["screen_direction"] == "negligible"


def test_press_waypoint_stays_in_press_phase_during_descent():
    env = RoboTwinEnvironment("click_bell", privilege_level=3)
    goal = np.array([-0.1, 0.0, 0.8])
    env._waypoint(np.array([-0.3, 0, 0.95]), goal, goal, 0.0)
    aligned = env._waypoint(np.array([-0.1, 0, 0.88]), goal, goal, 0.0)
    descended = env._waypoint(np.array([-0.1, 0, 0.83]), goal, goal, 0.0)
    assert aligned["phase"] == descended["phase"] == "press"
    assert descended["delta_xyz"][2] < 0


def test_bottle_waypoint_lifts_after_one_close_decision():
    env = RoboTwinEnvironment("move_pillbottle_pad", privilege_level=3)
    env._initial_object_z = 0.78
    obj = np.array([-0.1, 0.0, 0.78])
    goal = np.array([-0.2, 0.1, 0.74])
    env._waypoint(np.array([-0.2, 0, 0.90]), obj, goal, 1.0)
    env._waypoint(np.array([-0.1, 0, 0.80]), obj, goal, 1.0)
    env._gripper_command = "closed"
    env._control_steps = 1
    after_close = env._waypoint(np.array([-0.1, 0, 0.80]), obj, goal, 0.42)
    assert after_close["phase"] == "lift"
    assert after_close["target_xyz"][2] > 0.80
    assert after_close["gripper_change_needed"] is False


def _contact_fixture(task_name, active_arm="left"):
    env = RoboTwinEnvironment(task_name, active_arm=active_arm, privilege_level=2)

    def fingers(arm):
        return [(SimpleNamespace(child_link=SimpleNamespace(get_name=lambda name=name: name)), 1, 0)
                for name in (f"{arm}_finger_a", f"{arm}_finger_b")]

    env.task = SimpleNamespace(
        robot=SimpleNamespace(left_gripper=fingers("left"), right_gripper=fingers("right"),
                              get_left_gripper_val=lambda: 0.4, get_right_gripper_val=lambda: 0.4),
        scene=SimpleNamespace(get_contacts=lambda: contacts),
        get_obs=lambda: {"observation": {"head_camera": {"rgb": np.zeros((32, 32, 3), dtype=np.uint8)}}},
    )
    setattr(env.task, TASKS[task_name].actor, SimpleNamespace(get_name=lambda: "task_object"))
    contacts = []
    return env, contacts


def _contact(a, b, *, points=True):
    return SimpleNamespace(
        bodies=[SimpleNamespace(entity=SimpleNamespace(name=name)) for name in (a, b)],
        points=[object()] if points else [],
    )


@pytest.mark.parametrize("task_name", ["move_pillbottle_pad", "place_container_plate", "place_object_scale"])
@pytest.mark.parametrize("arm", ["left", "right"])
def test_placement_contact_uses_registered_actor_and_both_active_fingers(task_name, arm):
    env, contacts = _contact_fixture(task_name, arm)
    passive = "right" if arm == "left" else "left"
    contacts.extend([_contact("task_object", f"{passive}_finger_a"),
                     _contact("task_object", f"{passive}_finger_b"),
                     _contact("unrelated_object", f"{arm}_finger_b")])
    assert env._both_fingers_touch_object() is False
    contacts.append(_contact("task_object", f"{arm}_finger_a"))
    assert env._both_fingers_touch_object() is False
    contacts.append(_contact(f"{arm}_finger_b", "task_object", points=False))
    assert env._both_fingers_touch_object() is False
    contacts.append(_contact(f"{arm}_finger_b", "task_object"))
    assert env._both_fingers_touch_object() is True


@pytest.mark.parametrize("task_name", ["place_container_plate", "place_object_scale"])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_new_placement_observation_exposes_measured_contact_only_at_l2(monkeypatch, task_name, level):
    env, contacts = _contact_fixture(task_name)
    contacts.extend([_contact("task_object", "left_finger_a"),
                     _contact("left_finger_b", "task_object")])
    env.privilege_level = level
    monkeypatch.setattr(env, "_tcp", lambda: np.array([0, 0, 0.8]))
    monkeypatch.setattr(env, "_task_points", lambda: (np.array([0, 0, 0.8]), np.array([0.1, 0, 0.8])))
    monkeypatch.setattr(env, "_robot_state", lambda: {})
    observation = env._observation()
    assert observation.evaluation_state["both_fingers_touch_object"] is True
    assert ("both_fingers_touch_object" in observation.state) == (level >= 2)

    env.task.robot.left_gripper = env.task.robot.left_gripper[:1]
    assert env._both_fingers_touch_object() is None
    unavailable = env._observation()
    assert "both_fingers_touch_object" not in unavailable.evaluation_state
    assert "both_fingers_touch_object" not in unavailable.state
