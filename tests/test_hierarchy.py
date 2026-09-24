"""Adaptive visual choices must preserve the observation budget and model control."""

import json

from PIL import Image
import pytest

from jev_robo_eval.core import Action, Observation
from jev_robo_eval.hierarchy import (
    MOTION_SCALES, eligible_phases, phase_action_candidates, schema_for_task,
)
from jev_robo_eval.jev import JevPolicy
from jev_robo_eval.task_registry import TASK_REGISTRY


def observation(*, task="button-press-v3", step=0, level=0, gripper="open", **extra):
    return Observation({
        "task_name": task, "privilege_level": level,
        "information": "nonprivileged" if level == 0 else "privileged",
        "control_xyz": [0.0, 0.6, 0.15], "gripper_opening": 1.0,
        "gripper_command": gripper, "simulator_steps": step, **extra,
    }, Image.new("RGB", (32, 32), "white"), evaluation_state={"hidden_success": True})


def policy(**kwargs):
    return JevPolicy(**{"mode": "vision", "sensor_policy": "staged", "action_granularity": "adaptive", **kwargs})


@pytest.mark.parametrize("task_name", TASK_REGISTRY)
def test_every_registered_task_has_a_bounded_family_schema(task_name):
    schema = schema_for_task(task_name)
    for phase in schema.phases:
        for command in ("open", "closed"):
            phases, _ = eligible_phases(schema, phase, command)
            assert phases and set(phases) <= set(schema.phases)
            candidates = phase_action_candidates(schema, phase, command)
            assert len(candidates) <= 20
            assert len({candidate.id for candidate in candidates}) == len(candidates)
            assert all(0 < candidate.scale <= 1 for candidate in candidates)
            movements = [candidate for candidate in candidates
                         if candidate.action not in {Action.GRIP_OPEN, Action.GRIP_CLOSE, Action.HOLD}]
            assert len(movements) == 18
            for action in {candidate.action for candidate in movements}:
                assert {candidate.scale for candidate in movements if candidate.action == action} == {0.25, 0.5, 1.0}
            assert ("grip_open" if command == "open" else "grip_close") not in {
                candidate.id for candidate in candidates}


@pytest.mark.parametrize("task_name, family, operation", [
    ("button-press-v3", "pressing", "press"),
    ("button-press-topdown-v3", "pressing", "press"),
    ("click_alarmclock", "pressing", "press"),
    ("push-v3", "horizontal_push", "push"),
    ("drawer-close-v3", "horizontal_push", "push"),
    ("plate-slide-side-v3", "horizontal_push", "push"),
    ("window-close-v3", "pull_slide", "move"),
])
def test_contact_families_have_no_mandatory_grasp_lift_or_transfer(task_name, family, operation):
    schema = schema_for_task(task_name)
    assert schema.family == family
    assert operation in schema.phases
    assert {"grasp", "lift", "transfer", "place"}.isdisjoint(schema.phases)
    assert "contact" in eligible_phases(schema, None, "open")[0]


def test_motion_candidates_keep_both_signs_and_all_scales():
    schema = schema_for_task("push-v3")
    choices = {candidate.id: candidate for candidate in phase_action_candidates(schema, "push", "closed")}
    for direction in ("x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg"):
        for size, scale in MOTION_SCALES.items():
            assert choices[f"{direction}_{size}"].scale == scale
    assert "z_pos_fine" in choices and "z_neg_fine" in choices
    assert "z_pos_coarse" in choices


def test_phases_can_be_reconsidered_immediately_after_a_wrong_operation():
    pressing = schema_for_task("button-press-v3")
    for previous in (None, "press", "recover"):
        assert set(eligible_phases(pressing, previous, "open")[0]) == set(pressing.phases)
    picking = schema_for_task("pick-place-v3")
    assert set(eligible_phases(picking, "place", "closed")[0]) == set(picking.phases)
    assert set(eligible_phases(picking, "place", "open")[0]) == set(picking.phases) - {"lift", "transfer", "place"}
    for task, phase in (("button-press-v3", "press"), ("push-v3", "push")):
        assert "grip_close" in {c.id for c in phase_action_candidates(schema_for_task(task), phase, "open")}


@pytest.mark.parametrize("task_name", TASK_REGISTRY)
def test_phase_fixed_preserves_family_primitive_eligibility_at_unit_scale(task_name):
    schema = schema_for_task(task_name)
    for phase in schema.phases:
        for command in ("open", "closed"):
            adaptive = phase_action_candidates(schema, phase, command)
            fixed = phase_action_candidates(schema, phase, command, granularity="phase-fixed")
            assert {candidate.action for candidate in fixed} == {candidate.action for candidate in adaptive}
            assert len(fixed) == len({candidate.action for candidate in fixed})
            assert all(candidate.scale == 1.0 for candidate in fixed)
            assert all(candidate.id.endswith("_fixed") or candidate.id in {"grip_open", "grip_close", "hold"}
                       for candidate in fixed)


def test_phase_fixed_runs_the_same_two_stage_hierarchy_with_distinct_protocol(monkeypatch):
    requests = {}
    decisions = {}
    for granularity, selected in (("adaptive", "y_pos_fine"), ("phase-fixed", "y_pos_fixed")):
        controller = policy(action_granularity=granularity)
        sent = []

        def post(payload):
            request = payload["request"]
            sent.append(request)
            key = "phase" if "phase" in request["questions"] else "action"
            return {"answers": {key: {"choice": "contact" if key == "phase" else selected}}}

        monkeypatch.setattr(controller, "_post", post)
        decisions[granularity] = controller.decide(observation(), "Press the button.")
        assert len(sent) == 2
        requests[granularity] = sent
    assert requests["adaptive"][0] == requests["phase-fixed"][0]
    assert requests["adaptive"][1]["state"] == requests["phase-fixed"][1]["state"]
    assert decisions["adaptive"].action == decisions["phase-fixed"].action == Action.Y_POS
    assert decisions["adaptive"].action_scale == 0.25
    assert decisions["phase-fixed"].action_scale == 1.0
    fixed = decisions["phase-fixed"]
    assert fixed.selection["action_granularity"] == "phase-fixed"
    assert fixed.selection["selected_candidate"] == "y_pos_fixed"
    assert fixed.selection["candidate_protocol"] != decisions["adaptive"].selection["candidate_protocol"]
    assert fixed.request["phase_decision"]["choice"] == "contact"
    assert "Fine =" not in requests["phase-fixed"][1]["questions"]["action"]["instructions"]
    assert "fixed 1.0" in requests["phase-fixed"][1]["questions"]["action"]["instructions"]


@pytest.mark.parametrize("size, phase", [("fine", "refine"), ("normal", "approach"), ("coarse", "approach")])
def test_model_selects_direction_and_scale_and_preserves_both_call_records(monkeypatch, size, phase):
    controller = policy()
    sent = []

    def post(payload):
        request = payload["request"]
        sent.append(request)
        if "phase" in request["questions"]:
            return {"answers": {"phase": {"choice": phase, "probabilities": {phase: 1}}},
                    "usage": {"image_tokens": 17}}
        selected = f"x_neg_{size}"
        assert selected in request["questions"]["action"]["criteria"]
        return {"answers": {"action": {"choice": selected, "probabilities": {selected: 1}}},
                "usage": {"image_tokens": 19}}

    monkeypatch.setattr(controller, "_post", post)
    decision = controller.decide(observation(task="reach-v3"), "Reach the visible target.")
    assert decision.action == Action.X_NEG
    assert decision.action_scale == MOTION_SCALES[size]
    assert decision.selection["selected_candidate"] == f"x_neg_{size}"
    assert decision.selection["action_scale"] == decision.action_scale
    assert decision.selection["inferred_phase"] == phase
    assert decision.request["phase_decision"]["request"] == sent[0]
    assert decision.request["phase_decision"]["usage"] == {"image_tokens": 17}
    assert decision.usage == {"image_tokens": 19}
    assert len(sent) == 2


def test_l0_hidden_truth_cannot_change_requests_candidates_or_decision(monkeypatch):
    recordings = []
    for marker in ("first-hidden-truth", "second-hidden-truth"):
        controller = policy()
        sent = []

        def post(payload):
            sent.append(payload)
            key = "phase" if "phase" in payload["request"]["questions"] else "action"
            return {"answers": {key: {"choice": "contact" if key == "phase" else "y_pos_fine"}}}

        monkeypatch.setattr(controller, "_post", post)
        obs = observation(object_xyz=[marker] * 3, goal_xyz=[marker] * 3,
                          scene={"object": marker}, both_fingers_touch_object=True,
                          unexpected_truth=marker, robot={"oracle_secret": marker}, nominal_motion_step_m=0.02)
        decision = controller.decide(obs, "Press the visible button.")
        assert marker not in json.dumps(sent)
        assert "hidden_success" not in json.dumps(sent)
        assert sent[1]["request"]["state"]["nominal_motion_step_m"] == 0.02
        assert set(sent[0]["request"]["state"]) == {"current_contact_surface", "robot_tcp_xyz", "gripper_command"}
        recordings.append((sent, decision.action, decision.action_scale))
    assert recordings[0] == recordings[1]


@pytest.mark.parametrize("level", (1, 2))
def test_adaptive_uses_exact_level_budget_for_both_requests(monkeypatch, level):
    controller = policy()
    sent = []

    def post(payload):
        request = payload["request"]
        sent.append(request)
        key = "phase" if "phase" in request["questions"] else "action"
        return {"answers": {key: {"choice": "contact" if key == "phase" else "y_neg_fine"}}}

    monkeypatch.setattr(controller, "_post", post)
    decision = controller.decide(observation(level=level, object_xyz=[0.1, 0.7, 0.1],
                                            goal_xyz=[0.1, 0.8, 0.1], both_fingers_touch_object=True,
                                            hidden_success=True), "Press the button.")
    assert sent[0]["state"]["current_contact_xyz"] == [0.1, 0.7, 0.1]
    assert sent[1]["state"]["current_contact_xyz"] == [0.1, 0.7, 0.1]
    for request in sent:
        assert ("both_fingers_touch_object" in request["state"]) == (level == 2)
        assert "hidden_success" not in request["state"]
    assert "target_evidence" not in sent[1]["state"]
    assert decision.request["observation_audit"]["target_evidence"]["contact"]["source"] == "object_xyz"
    assert decision.request["observation_audit"]["target_evidence"]["contact"]["delta_xyz"] == [0.1, 0.1, -0.05]
    criteria = sent[-1]["questions"]["action"]["criteria"]
    assert "moves toward" in criteria["y_pos_fine"]
    assert "moves away from" in criteria["y_neg_fine"]
    # Coordinates do not choose an action in the executor: the model chose Y-.
    assert decision.action == Action.Y_NEG and decision.action_scale == 0.25


@pytest.mark.parametrize("level", [0, 1, 2])
def test_interaction_facts_reach_both_heads_only_at_l2(monkeypatch, level):
    controller = policy()
    sent = []

    def post(payload):
        request = payload["request"]
        sent.append(request)
        key = "phase" if "phase" in request["questions"] else "action"
        return {"answers": {key: {"choice": "approach" if key == "phase" else "y_pos_coarse"}}}

    monkeypatch.setattr(controller, "_post", post)
    controller.decide(observation(level=level, object_xyz=[0.0, 0.9, 0.2], goal_xyz=[0.0, 0.9, 0.1],
                                  window_slide_m=0.02, button_remaining_travel_m=0.03), "Press down.")
    for request in sent:
        assert ("window_slide_m" in request["state"]) == (level == 2)
        assert ("button_remaining_travel_m" in request["state"]) == (level == 2)
        assert "Press down." not in json.dumps(request)
    assert "target_evidence" not in json.dumps(sent)


def test_stall_history_is_evidence_and_does_not_override_repeated_model_choice(monkeypatch):
    controller = policy()
    sent = []

    def post(payload):
        request = payload["request"]
        sent.append(request)
        key = "phase" if "phase" in request["questions"] else "action"
        return {"answers": {key: {"choice": "contact" if key == "phase" else "y_pos_fine"}}}

    monkeypatch.setattr(controller, "_post", post)
    for step in range(3):
        decision = controller.decide(observation(step=step), "Press the button.")
        assert decision.action == Action.Y_POS and decision.action_scale == 0.25
    history = sent[-1]["state"]
    assert history["consecutive_same_action"] == 2
    assert history["last_tcp_motion_xyz"] == [0.0, 0.0, 0.0]
    assert decision.request["observation_audit"]["history"]["decisions_in_previous_phase"] == 2
    assert decision.request["observation_audit"]["history"]["motion_history"]["consecutive_stalled_movements"] == 2
    controller.decide(observation(step=0), "Press the button.")
    assert "previous_action" not in sent[-1]["state"]
    assert "motion_history" not in sent[-1]["state"]


def test_gripper_change_is_not_stall_and_reversal_is_recorded_without_forcing_an_action(monkeypatch):
    controller = policy()
    choices = iter(("grip_close", "x_pos_fine", "x_neg_fine", "x_pos_fine"))
    sent = []

    def post(payload):
        request = payload["request"]
        sent.append(request)
        key = "phase" if "phase" in request["questions"] else "action"
        return {"answers": {key: {"choice": "contact" if key == "phase" else next(choices)}}}

    monkeypatch.setattr(controller, "_post", post)
    for step in range(4):
        decision = controller.decide(observation(step=step, gripper="open" if step == 0 else "closed"), "Press the button.")
        if step == 1:
            assert decision.request["observation_audit"]["history"]["motion_history"]["consecutive_stalled_movements"] == 0
    assert sent[3]["state"]["previous_action"] == "grip_close"
    assert sent[-1]["state"]["consecutive_axis_reversals"] == 1
    assert controller._adaptive_axis_reversals == 2


def test_motion_callback_records_actual_scale_and_clears_only_mismatched_candidate():
    controller = policy()
    controller._sensor_last_action = "x_pos"
    controller._adaptive_last_candidate = "x_pos_fine"
    controller._adaptive_last_scale = 0.25
    controller.on_motion_executed(Action.X_POS, 0.25)
    assert controller._adaptive_last_candidate == "x_pos_fine"
    controller.on_motion_executed(Action.X_POS, 1.0)
    assert controller._adaptive_last_scale == 1.0
    assert controller._adaptive_last_candidate is None
    controller.on_motion_executed(Action.Y_NEG, 0.5)
    assert controller._sensor_last_action == "y_neg"
    assert controller._adaptive_last_scale == 0.5
    assert controller._adaptive_last_candidate is None


@pytest.mark.parametrize("kwargs", [{"mode": "text"}, {"sensor_policy": "direct"},
                                    {"action_space": "atomic"}, {"action_granularity": "invalid"}])
@pytest.mark.parametrize("granularity", ("adaptive", "phase-fixed"))
def test_adaptive_rejects_incompatible_configuration(kwargs, granularity):
    config = {"mode": "vision", "sensor_policy": "staged", "action_granularity": granularity, **kwargs}
    with pytest.raises(ValueError):
        JevPolicy(**config)


@pytest.mark.parametrize("extra", [{"privilege_level": 3}, {"active_waypoint": {}}, {"plan_only": True}])
@pytest.mark.parametrize("granularity", ("adaptive", "phase-fixed"))
def test_adaptive_rejects_oracle_before_post(monkeypatch, extra, granularity):
    controller = policy(action_granularity=granularity)
    monkeypatch.setattr(controller, "_post", lambda payload: pytest.fail("Oracle state must be rejected"))
    with pytest.raises(ValueError, match="without oracle waypoints"):
        controller.decide(observation(**extra), "Press the button.")


def test_unknown_task_fails_explicitly():
    with pytest.raises(ValueError, match="no task-family schema"):
        policy().decide(observation(task="unsupported-task"), "Do the task.")


@pytest.mark.parametrize("choice", ["y_pos", "y_pos_fixed", "unknown"])
def test_action_outside_offered_candidates_fails(monkeypatch, choice):
    controller = policy()

    def post(payload):
        key = "phase" if "phase" in payload["request"]["questions"] else "action"
        return {"answers": {key: {"choice": "contact" if key == "phase" else choice}}}

    monkeypatch.setattr(controller, "_post", post)
    with pytest.raises(RuntimeError, match="Invalid Jev adaptive choice"):
        controller.decide(observation(), "Press the button.")


def test_sampling_uses_candidate_distribution_and_replays_seed(monkeypatch):
    sequences = []
    for _ in range(2):
        controller = policy(action_selection="sample", policy_seed=7)

        def post(payload):
            questions = payload["request"]["questions"]
            if "phase" in questions:
                return {"answers": {"phase": {"choice": "contact"}}}
            choices = questions["action"]["criteria"]
            probabilities = {key: float(key in {"y_pos_fine", "y_pos_normal"}) for key in choices}
            return {"answers": {"action": {"choice": "y_pos_fine", "probabilities": probabilities}}}

        monkeypatch.setattr(controller, "_post", post)
        decisions = [controller.decide(observation(step=step), "Press the button.") for step in range(12)]
        sequences.append([decision.action_scale for decision in decisions])
        assert all(decision.action == Action.Y_POS for decision in decisions)
        assert all(decision.selection["service_choice"] == "y_pos_fine" for decision in decisions)
    assert sequences[0] == sequences[1]
    assert set(sequences[0]) == {0.25, 0.5}
