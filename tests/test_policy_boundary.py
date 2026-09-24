"""The policy boundary may see camera/proprioception, never evaluation truth."""

from __future__ import annotations

import base64
import io
import json

from PIL import Image
import pytest

from jev_robo_eval.core import Action, Decision, Observation, Transition
from jev_robo_eval.jev import JevPolicy
from jev_robo_eval.runner import run_episode


FORBIDDEN = {
    "goal_xyz", "object_xyz", "scene", "both_fingers_touch_object",
    "success_distance_m", "active_waypoint", "nut_center_xyz",
}
ALL_ACTIONS = {action.value for action in Action}


def test_robotwin_direct_history_survives_a_failed_plan_without_physics_progress(monkeypatch):
    policy = JevPolicy(mode="vision", sensor_policy="direct")
    sent = []

    def fake_post(payload):
        sent.append(payload["request"])
        return {"answers": {"action": {"choice": "z_neg"}}}

    monkeypatch.setattr(policy, "_post", fake_post)
    state = {"environment": "robotwin", "task_name": "click_bell", "privilege_level": 0,
             "information": "nonprivileged", "active_arm": "left", "simulator_steps": 0,
             "control_xyz": [-0.2, 0.0, 0.9], "gripper_opening": 1.0,
             "gripper_command": "open", "tcp_pixel": {"u": -10, "v": 120}}
    for step in (0, 1):
        policy.decide(Observation({**state, "control_steps": step}, Image.new("RGB", (32, 32))),
                      "Press the bell.")
    assert "previous_action" not in sent[0]["state"]
    assert sent[1]["state"]["previous_action"] == "z_neg"
    assert sent[1]["state"]["last_tcp_motion_xyz"] == [0.0, 0.0, 0.0]
    assert sent[1]["state"]["tcp_pixel"] == {"u": -10, "v": 120}
    assert FORBIDDEN.isdisjoint(sent[1]["state"])


def test_press_phases_use_robot_command_and_model_history_without_oracle(monkeypatch):
    policy = JevPolicy(mode="vision", sensor_policy="staged")
    phase_choices = iter(("prepare", "align", "press"))
    candidates = []

    def fake_post(payload):
        request = _assert_safe_payload(payload)
        if "phase" in request["questions"]:
            choices = request["questions"]["phase"]["criteria"]
            candidates.append(set(choices))
            choice = next(phase_choices)
            assert choice in choices
            return {"answers": {"phase": {"choice": choice, "probabilities": {choice: 1}}}}
        assert set(request["questions"]["action"]["criteria"]) == ALL_ACTIONS
        assert request["state"]["task"].startswith("Current operation:")
        assert FORBIDDEN.isdisjoint(request["state"])
        return {"answers": {"action": {"choice": "grip_close"}}}

    monkeypatch.setattr(policy, "_post", fake_post)
    for step in range(3):
        observation = _observation(step=step, gripper="open" if step == 0 else "closed")
        observation.state.update(task_name="click_bell", environment="robotwin",
                                 privilege_level=0, control_steps=step)
        policy.decide(observation, "Press the bell.")
    assert "press" not in candidates[0]
    assert candidates[2] == {"align", "press", "prepare", "recover"}


def _observation(*, step=0, gripper="open", tcp=(0.0, 0.6, 0.15)) -> Observation:
    state = {
        "information": "nonprivileged",
        "task_name": "assembly-v3",
        "control_xyz": list(tcp),
        "gripper_opening": 1.0 if gripper == "open" else 0.4,
        "gripper_command": gripper,
        "simulator_steps": step,
        "robot": {"hand_body_xyz": [tcp[0], tcp[1], tcp[2] + 0.045]},
        "camera_name": "corner2",
        "action_screen_directions": {
            "x_pos": {"sample_distance_m": 0.05, "delta_px": [4.0, -2.0],
                      "screen_direction": "up-right"},
        },
        # A defensive policy allowlist should drop these even if another
        # adapter accidentally adds them to state.
        "goal_xyz": [9.8765, 9.8765, 9.8765],
        "object_xyz": [8.7654, 8.7654, 8.7654],
        "scene": {"secret": "oracle-sentinel"},
        "both_fingers_touch_object": True,
        "success_distance_m": 0.0001,
    }
    return Observation(
        state=state,
        image=Image.new("RGB", (32, 32), "white"),
        evaluation_state={"goal_xyz": [9.8765, 9.8765, 9.8765], "secret": "eval-only"},
    )


def _assert_safe_payload(payload: dict) -> dict:
    assert set(payload) == {"request", "image_base64"}
    with Image.open(io.BytesIO(base64.b64decode(payload["image_base64"]))) as image:
        assert image.size == (32, 32)
    request = payload["request"]
    state = request["state"]
    assert state["information"] == "nonprivileged"
    assert state["camera_name"] == "corner2"
    assert "action_screen_directions" in state
    assert FORBIDDEN.isdisjoint(state)
    serialized = json.dumps(request)
    assert "oracle-sentinel" not in serialized
    assert "eval-only" not in serialized
    assert "9.8765" not in serialized
    assert "8.7654" not in serialized
    return request


def test_runner_strips_evaluation_state_before_policy(tmp_path) -> None:
    def safe_observation(step):
        observation = _observation(step=step)
        safe_state = {key: value for key, value in observation.state.items()
                      if key not in FORBIDDEN}
        return Observation(state=safe_state, image=observation.image,
                           evaluation_state=observation.evaluation_state)

    class Environment:
        def reset(self, *, seed=None):
            return safe_observation(step=0)

        def step(self, action):
            assert action == Action.HOLD
            return Transition(safe_observation(step=3), reward=1.0,
                              terminated=False, truncated=False, success=True)

        def close(self):
            pass

    class Policy:
        def decide(self, observation, task):
            assert observation.evaluation_state is None
            assert observation.image is not None
            assert observation.state["information"] == "nonprivileged"
            assert FORBIDDEN.isdisjoint(observation.state)
            return Decision(Action.HOLD)

    trace = tmp_path / "episode.jsonl"
    result = run_episode(Environment(), Policy(), task="Visible task", seed=0,
                         max_decisions=1, trace_path=trace, verbose=False)
    assert result.success
    row = json.loads(trace.read_text().strip())
    assert row["state_before"]["secret"] == "eval-only"
    assert row["policy_state_before"]["information"] == "nonprivileged"


def test_direct_sends_only_safe_payload_and_keeps_custom_task(monkeypatch) -> None:
    policy = JevPolicy(mode="vision", sensor_policy="direct")
    requests = []

    def fake_post(payload):
        request = _assert_safe_payload(payload)
        requests.append(request)
        assert set(request["questions"]) == {"action"}
        assert set(request["questions"]["action"]["criteria"]) == ALL_ACTIONS
        return {"answers": {"action": {
            "choice": "x_pos", "probabilities": {"x_pos": 0.8, "hold": 0.2},
        }}}

    monkeypatch.setattr(policy, "_post", fake_post)
    custom_task = "Mount the visible ring on the upright peg."
    decision = policy.decide(_observation(), custom_task)
    assert decision.action == Action.X_POS
    assert requests[-1]["state"]["task"] == custom_task
    assert decision.request == requests[-1]

    policy.decide(_observation(), "Mount the ring at goal_xyz.")
    assert "goal_xyz" not in requests[-1]["state"]["task"]


@pytest.mark.parametrize("level", [1, 2])
def test_direct_level_budget_filters_unexpected_truth(monkeypatch, level: int) -> None:
    """A new adapter field must not silently widen the direct JEV request."""
    policy = JevPolicy(mode="vision", sensor_policy="direct")
    observation = _observation()
    observation.state.update({
        "information": "privileged",
        "privilege_level": level,
        "object_xyz": [0.1, 0.2, 0.3],
        "goal_xyz": [0.4, 0.5, 0.6],
        "scene": {"interaction_points_xyz": [[0.1, 0.2, 0.3]]},
        "both_fingers_touch_object": True,
        "success_distance_m": 0.001,
        "unexpected_truth": "future-oracle-field",
    })
    requests = []

    def fake_post(payload):
        request = payload["request"]
        requests.append(request)
        state = request["state"]
        assert state["privilege_level"] == level
        assert state["object_xyz"] == [0.1, 0.2, 0.3]
        assert state["goal_xyz"] == [0.4, 0.5, 0.6]
        assert "unexpected_truth" not in state
        assert "active_waypoint" not in state
        if level == 1:
            assert "both_fingers_touch_object" not in state
            assert "success_distance_m" not in state
        else:
            assert state["both_fingers_touch_object"] is True
            assert state["success_distance_m"] == 0.001
        assert set(request["questions"]["action"]["criteria"]) == ALL_ACTIONS
        return {"answers": {"action": {"choice": "hold"}}}

    monkeypatch.setattr(policy, "_post", fake_post)
    decision = policy.decide(observation, "Mount the visible ring on the peg.")
    assert decision.action == Action.HOLD
    assert len(requests) == 1


@pytest.mark.parametrize("level", [1, 2])
def test_lower_privilege_rejects_oracle_waypoint_before_model_call(monkeypatch, level: int) -> None:
    policy = JevPolicy(mode="vision", sensor_policy="direct")
    observation = _observation()
    observation.state.update({
        "information": "privileged",
        "privilege_level": level,
        "active_waypoint": {"instruction": "oracle-only"},
    })

    def forbidden_post(payload):
        pytest.fail("JEV must not receive an oracle waypoint at levels 1 or 2")

    monkeypatch.setattr(policy, "_post", forbidden_post)
    with pytest.raises(ValueError, match="Oracle waypoints require privilege level 3"):
        policy.decide(observation, "Mount the visible ring on the peg.")


def test_staged_candidates_use_only_own_phase_and_gripper_command(monkeypatch) -> None:
    policy = JevPolicy(mode="vision", sensor_policy="staged")
    chosen_phases = iter(("approach", "lower", "grasp", "lift", "recover"))
    expected_candidates = iter((
        {"approach", "lower", "recover"},
        {"approach", "lower", "recover"},
        {"lower", "approach", "grasp", "recover"},
        {"grasp", "lift", "approach", "recover"},
        {"recover"},
    ))
    sent_phase_requests = []
    sent_action_requests = []

    def fake_post(payload):
        request = _assert_safe_payload(payload)
        if "phase" in request["questions"]:
            assert set(request["questions"]["phase"]["criteria"]) == next(expected_candidates)
            assert "eligible" in request["questions"]["phase"]["instructions"]
            sent_phase_requests.append(request)
            phase = next(chosen_phases)
            return {"answers": {"phase": {
                "choice": phase, "probabilities": {phase: 0.9},
            }}}
        assert set(request["questions"]["action"]["criteria"]) == ALL_ACTIONS
        sent_action_requests.append(request)
        return {"answers": {"action": {
            "choice": "x_pos", "probabilities": {"x_pos": 0.9},
        }}}

    monkeypatch.setattr(policy, "_post", fake_post)
    inputs = (
        (0, "open"), (3, "open"), (6, "closed"),
        (9, "closed"), (12, "open"),
    )
    for index, (step, gripper) in enumerate(inputs):
        observation = _observation(step=step, gripper=gripper,
                                   tcp=(0.01 * index, 0.6, 0.15))
        decision = policy.decide(observation, "Mount the visible nut on the peg.")
        assert decision.action == Action.X_POS
        assert decision.request["phase_decision"]["request"] == sent_phase_requests[-1]
        assert decision.request["phase_decision"]["probabilities"] == {
            decision.request["phase_decision"]["choice"]: 0.9,
        }
        assert decision.request["questions"] == sent_action_requests[-1]["questions"]
        if index:
            assert sent_phase_requests[-1]["state"]["previous_action"] == "x_pos"
            assert sent_phase_requests[-1]["state"]["previous_inferred_phase"] == (
                "approach", "lower", "grasp", "lift")[index - 1]
            assert sent_phase_requests[-1]["state"]["last_tcp_motion_xyz"] == [0.01, 0.0, 0.0]
    assert len(sent_phase_requests) == len(sent_action_requests) == 5


def test_staged_rejects_model_phase_outside_eligible_candidates(monkeypatch) -> None:
    policy = JevPolicy(mode="vision", sensor_policy="staged")
    calls = []

    def fake_post(payload):
        request = _assert_safe_payload(payload)
        calls.append(request)
        return {"answers": {"phase": {
            "choice": "place", "probabilities": {"place": 0.9},
        }}}

    monkeypatch.setattr(policy, "_post", fake_post)
    with pytest.raises(RuntimeError, match="Invalid Jev phase response") as error:
        policy.decide(_observation(), "Mount the ring on the peg.")
    assert "Ineligible phase" in str(error.value.__cause__)
    assert len(calls) == 1
