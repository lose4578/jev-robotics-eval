"""The L3-P ablation preserves waypoints and removes extra environment truth."""

import json

from PIL import Image
import pytest

from jev_robo_eval.core import Action, Observation
from jev_robo_eval.jev import JevPolicy
from jev_robo_eval.observation_access import filter_policy_state
from jev_robo_eval.metaworld_env import MetaWorldMT1


def _state():
    return {
        "privilege_level": 3, "information": "privileged", "task_name": "assembly-v3",
        "control_xyz": [0, 0.5, 0.2], "robot": {"hand_body_xyz": [0, 0.5, 0.24]},
        "gripper_command": "open", "gripper_opening": 1.0,
        "object_xyz": [8.7654] * 3, "goal_xyz": [9.8765] * 3,
        "scene": {"secret": "scene-sentinel"}, "both_fingers_touch_object": True,
        "success": True, "object_lift_m": 0.4,
        "active_waypoint": {
            "phase": "approach", "target_xyz": [0.1, 0.6, 0.3],
            "delta_xyz": [0.1, 0.1, 0.1], "relative_direction": "up right forward",
            "instruction": "Approach the planned point.", "desired_gripper": "open",
            "gripper_change_needed": False, "future_truth": "nested-sentinel",
        },
    }


@pytest.mark.parametrize("mode", ["text", "vision"])
def test_plan_only_request_is_full_l3_minus_scene(monkeypatch, mode):
    policy = JevPolicy(mode=mode)
    calls = []
    def post(payload):
        calls.append(payload.get("request", payload))
        return {"answers": {"action": {"choice": "x_pos"}}}
    monkeypatch.setattr(policy, "_post", post)
    image = Image.new("RGB", (32, 32))
    policy.decide(Observation(_state(), image), "Assemble the nut.")
    policy.decide(Observation({**_state(), "plan_only": True}, image), "Assemble the nut.")
    full, planned = calls
    expected = {**full, "state": {k: v for k, v in full["state"].items() if k != "scene"}}
    assert planned == expected
    serialized = json.dumps(planned)
    for secret in ("scene-sentinel", "nested-sentinel", "8.7654", "9.8765",
                   "both_fingers_touch_object", "object_lift_m"):
        assert secret not in serialized
    assert planned["state"]["goal_xyz"] == [0.1, 0.6, 0.3]
    state = filter_policy_state(_state(), 3, plan_only=True)
    assert state["active_waypoint"]["phase"] == "approach"
    assert state["plan_provenance"] == "oracle_rules_using_environment_truth"
    assert not {"scene", "object_xyz", "goal_xyz", "success"} & state.keys()


@pytest.mark.parametrize("level", [0, 1, 2])
def test_plan_only_rejects_non_l3(level):
    with pytest.raises(ValueError, match="requires privilege_level=3"):
        filter_policy_state(_state(), level, plan_only=True)


def test_real_metaworld_same_waypoint_and_physical_state():
    observations = []
    for plan_only in (False, True):
        env = MetaWorldMT1("assembly-v3", mode="vision", image_size=64,
                          privilege_level=3, plan_only=plan_only)
        try:
            observations.append(env.reset(seed=2))
        finally:
            env.close()
    full, planned = observations
    assert planned.evaluation_state == full.evaluation_state
    assert planned.image.tobytes() == full.image.tobytes()
    assert planned.state["active_waypoint"] == {
        k: v for k, v in full.state["active_waypoint"].items()
        if k in planned.state["active_waypoint"]
    }
    assert "object_xyz" in full.state and "object_xyz" not in planned.state


def test_recovery_execution_feedback_updates_policy_history(monkeypatch):
    policy = JevPolicy(mode="vision")
    requests = []
    def post(payload):
        requests.append(payload["request"])
        return {"answers": {"action": {"choice": "x_pos"}}}
    monkeypatch.setattr(policy, "_post", post)
    state = {"privilege_level": 0, "information": "nonprivileged", "task_name": "click_bell",
             "control_xyz": [0, 0, 1], "gripper_command": "closed", "gripper_opening": 0,
             "control_steps": 0}
    policy.decide(Observation(state, Image.new("RGB", (32, 32))), "Press bell")
    policy.on_action_executed(Action.Z_POS)
    policy.decide(Observation({**state, "control_steps": 1}, Image.new("RGB", (32, 32))), "Press bell")
    assert requests[-1]["state"]["previous_action"] == "z_pos"
