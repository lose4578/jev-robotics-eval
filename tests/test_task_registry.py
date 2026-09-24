"""Task admission and geometric-fact contracts without loading a simulator."""

from types import SimpleNamespace

import numpy as np
import pytest

from jev_robo_eval.metaworld_tasks import task_facts
from jev_robo_eval.robotwin_tasks import TASKS as ROBOTWIN_SPECS
from jev_robo_eval.task_registry import (
    METAWORLD_TASKS, ROBOTWIN_TASKS, TASK_REGISTRY, tasks_for_environment,
)
from jev_robo_eval.tasks import SENSOR_TASK_GOALS, TASK_GOALS, task_goal
from jev_robo_eval.waypoints import WaypointGuide


def test_registered_tasks_have_explicit_goals_and_correct_environment():
    assert len(METAWORLD_TASKS) >= 15
    assert len(ROBOTWIN_TASKS) >= 6
    assert not set(METAWORLD_TASKS) & set(ROBOTWIN_TASKS)
    assert set(METAWORLD_TASKS) | set(ROBOTWIN_TASKS) == set(TASK_REGISTRY)
    assert set(ROBOTWIN_TASKS) == set(ROBOTWIN_SPECS)
    for name, spec in TASK_REGISTRY.items():
        assert name in tasks_for_environment(spec.environment)
        assert task_goal(name) == TASK_GOALS[name]
        assert task_goal(name, "nonprivileged") == SENSOR_TASK_GOALS[name]
        assert "goal_xyz" not in SENSOR_TASK_GOALS[name]
        assert "object_xyz" not in SENSOR_TASK_GOALS[name]
    with pytest.raises(ValueError, match="Unknown environment"):
        tasks_for_environment("unknown")


def test_registry_only_advertises_implemented_oracle_guides():
    assert {name for name in METAWORLD_TASKS if TASK_REGISTRY[name].oracle_supported} == WaypointGuide.TASKS
    assert {name for name in ROBOTWIN_TASKS if TASK_REGISTRY[name].oracle_supported} == {
        "click_bell", "press_stapler", "move_pillbottle_pad",
    }


def test_robotwin_goal_spec_identifies_contact_or_destination():
    for spec in ROBOTWIN_SPECS.values():
        assert (spec.contact_point is not None) != (spec.goal_actor is not None)
        if spec.goal_functional_point is not None:
            assert spec.goal_actor is not None
    assert ROBOTWIN_SPECS["click_alarmclock"].actor == "alarm"
    assert ROBOTWIN_SPECS["place_container_plate"].goal_actor == "plate"
    scale = ROBOTWIN_SPECS["place_object_scale"]
    assert scale.goal_actor == "scale" and scale.goal_functional_point == 0


@pytest.mark.parametrize(("task", "axis"), [
    ("button-press-v3", 1), ("button-press-topdown-v3", 2),
    ("window-open-v3", 0), ("window-close-v3", 0),
])
def test_fact_distance_uses_the_simulator_success_axis(task, axis):
    # Other coordinates deliberately differ: a full XYZ norm would report
    # failure even when the button/handle reaches its success coordinate.
    obs = np.zeros(10)
    obs[4:7] = [0.4, 0.5, 0.6]
    obs[-3:] = [0.1, 0.2, 0.3]
    obs[-3 + axis] = obs[4 + axis]
    joints = {"window_slide": SimpleNamespace(qpos=np.array([0.2]))}
    env = SimpleNamespace(data=SimpleNamespace(joint=joints.__getitem__))
    facts = task_facts(env, task, obs, np.zeros(3), obs[4:7].copy())
    assert facts["success_distance_m"] == 0.0
    if task.startswith("window"):
        assert facts["window_slide_m"] == 0.2
    else:
        assert facts["button_remaining_travel_m"] == 0.0


@pytest.mark.parametrize("task", ["drawer-close-v3", "plate-slide-v3", "plate-slide-side-v3"])
def test_free_object_and_drawer_distance_retains_xyz_norm(task):
    obs = np.zeros(10)
    obs[4:7] = [0.03, 0.04, 0]
    facts = task_facts(None, task, obs, np.zeros(3), obs[4:7].copy())
    assert facts["success_distance_m"] == 0.05
