"""Bounded MetaWorld intent candidates and their legacy-action lowering."""

from __future__ import annotations

import json

import pytest

from jev_robo_eval.atomic_actions import (
    ATOMIC_PROTOCOL,
    AtomicActionId,
    AtomicActionUnavailable,
    MetaWorldAtomicExecutor,
    metaworld_atomic_candidates,
)
from jev_robo_eval.core import Action, Observation
from jev_robo_eval.jev import JevPolicy


def _candidate_map(state):
    return {candidate.id: candidate for candidate in metaworld_atomic_candidates(state)}


def test_metaworld_candidates_have_finite_one_step_protocol_and_no_coordinates():
    state = {
        "privilege_level": 2,
        "control_xyz": [0.1, 0.2, 0.3],
        "goal_xyz": [9.8765, 9.8765, 9.8765],
        "object_xyz": [8.7654, 8.7654, 8.7654],
        "secret": "evaluation-only",
    }
    candidates = metaworld_atomic_candidates(state)
    assert [candidate.id for candidate in candidates] == [item.value for item in AtomicActionId
                                                          if item not in {AtomicActionId.VISUAL_LEFT,
                                                                          AtomicActionId.VISUAL_RIGHT,
                                                                          AtomicActionId.VISUAL_UP,
                                                                          AtomicActionId.VISUAL_DOWN}]
    for candidate in candidates:
        descriptor = candidate.to_dict()
        assert set(("id", "intent", "target", "parameters", "termination", "provenance")) <= descriptor.keys()
        assert descriptor["parameters"]["max_executor_steps"] == 1
        assert descriptor["termination"]["return_after_step"] is True
    serialized = json.dumps([candidate.to_dict() for candidate in candidates])
    # Candidate descriptors carry a reference to a visible field, never the
    # continuous simulator values (or unrelated future truth fields).
    assert "9.8765" not in serialized
    assert "8.7654" not in serialized
    assert "evaluation-only" not in serialized


def test_candidate_generation_fails_closed_if_evaluation_state_is_passed():
    with pytest.raises(ValueError, match="evaluation_state"):
        metaworld_atomic_candidates({"control_xyz": [0, 0, 0], "evaluation_state": {"goal_xyz": [1, 2, 3]}})


def test_executor_lowers_intents_to_one_signed_legacy_action():
    state = {"control_xyz": [0.0, 0.0, 0.0], "atomic_target_xyz": [0.2, -0.4, 0.1]}
    candidates = _candidate_map(state)
    executor = MetaWorldAtomicExecutor()

    align = executor.lower(candidates[AtomicActionId.ALIGN_XY.value], state)
    assert align.action == Action.Y_NEG
    assert align.steps == 1
    assert align.termination == "one_environment_step"

    adjust = executor.lower(candidates[AtomicActionId.ADJUST_Z.value], state)
    assert adjust.action == Action.Z_POS
    track = executor.lower(candidates[AtomicActionId.TRACK_WAYPOINT.value], state)
    assert track.action == Action.Y_NEG
    assert executor.lower(candidates[AtomicActionId.GRIP_OPEN.value], state).action == Action.GRIP_OPEN
    assert executor.lower(candidates[AtomicActionId.GRIP_CLOSE.value], state).action == Action.GRIP_CLOSE
    assert executor.lower(candidates[AtomicActionId.HOLD.value], state).action == Action.HOLD


def test_target_dependent_l0_intent_does_not_invent_environment_coordinates():
    state = {"control_xyz": [0.0, 0.0, 0.0]}
    candidate = _candidate_map(state)[AtomicActionId.VISUAL_RIGHT.value]
    with pytest.raises(AtomicActionUnavailable, match="calibrated action_screen_directions"):
        MetaWorldAtomicExecutor().lower(candidate, state)


def test_l0_visual_direction_uses_only_calibrated_robot_motion():
    state = {
        "control_xyz": [0.0, 0.0, 0.0],
        "action_screen_directions": {
            "x_pos": {"delta_px": [5.0, 0.0]},
            "x_neg": {"delta_px": [-5.0, 0.0]},
            "z_pos": {"delta_px": [0.0, -4.0]},
            "z_neg": {"delta_px": [0.0, 4.0]},
        },
    }
    candidates = _candidate_map(state)
    executor = MetaWorldAtomicExecutor()
    assert executor.lower(candidates["visual_right"], state).action == Action.X_POS
    assert executor.lower(candidates["visual_left"], state).action == Action.X_NEG
    assert executor.lower(candidates["visual_up"], state).action == Action.Z_POS
    assert executor.lower(candidates["visual_down"], state).action == Action.Z_NEG


def test_l0_visual_direction_fails_closed_when_calibration_has_no_matching_motion():
    state = {
        "control_xyz": [0.0, 0.0, 0.0],
        "action_screen_directions": {
            "x_neg": {"delta_px": [-5.0, 0.0]},
        },
    }
    candidate = _candidate_map(state)["visual_right"]
    with pytest.raises(AtomicActionUnavailable, match="no calibrated"):
        MetaWorldAtomicExecutor().lower(candidate, state)


def test_transfer_candidates_withhold_close_and_track_until_visible_grasp_approach():
    far = {
        "task_name": "shelf-place-v3", "gripper_command": "open",
        "control_xyz": [0.0, 0.0, 0.2], "object_xyz": [0.1, 0.0, 0.02],
        "goal_xyz": [0.0, 0.3, 0.2],
    }
    assert "track_waypoint" not in {c.id for c in metaworld_atomic_candidates(far)}
    assert "grip_close" not in {c.id for c in metaworld_atomic_candidates(far)}
    near = dict(far, control_xyz=[0.1, 0.0, 0.03])
    assert "grip_close" in {c.id for c in metaworld_atomic_candidates(near)}


def test_contact_tasks_withhold_goal_track_until_closed():
    state = {
        "task_name": "push-v3", "gripper_command": "open",
        "control_xyz": [0.0, 0.0, 0.2], "object_xyz": [0.0, 0.0, 0.02],
        "goal_xyz": [0.2, 0.0, 0.02],
    }
    assert "track_waypoint" not in {c.id for c in metaworld_atomic_candidates(state)}
    closed = dict(state, gripper_command="closed")
    assert [c.id for c in metaworld_atomic_candidates(closed)] == ["track_waypoint"]


def test_jev_uses_only_finite_atomic_choice_and_records_lowering(monkeypatch):
    policy = JevPolicy(mode="text", action_space="metaworld_atomic")
    requests = []

    def fake_post(payload):
        requests.append(payload)
        return {"answers": {"action": {"choice": "align_xy"}}, "input_tokens": 11}

    monkeypatch.setattr(policy, "_post", fake_post)
    state = {
        "privilege_level": 1,
        "information": "privileged",
        "task_name": "pick-place-v3",
        "control_xyz": [0.0, 0.0, 0.0],
        "goal_xyz": [0.25, -0.01, 0.01],
        "object_xyz": [0.9, 0.9, 0.9],
        "evaluation_state": {"secret": "must-not-be-read"},
    }
    decision = policy.decide(Observation(state, image=None), "Place the object.")
    assert decision.action == Action.X_POS
    request = requests[0]
    assert request["candidate_protocol"] == ATOMIC_PROTOCOL
    assert set(request["questions"]["action"]["criteria"]) == {
        item.value for item in AtomicActionId
        if item not in {
            AtomicActionId.VISUAL_LEFT, AtomicActionId.VISUAL_RIGHT,
            AtomicActionId.VISUAL_UP, AtomicActionId.VISUAL_DOWN,
        }
    }
    assert decision.selection["selected_candidate"] == "align_xy"
    assert decision.selection["execution"]["steps"] == 1
    assert decision.usage == {"input_tokens": 11}
    serialized = json.dumps(request)
    assert "must-not-be-read" not in serialized


def test_jev_l0_targetless_atomic_choice_falls_back_audibly(monkeypatch):
    policy = JevPolicy(mode="text", action_space="atomic")

    monkeypatch.setattr(policy, "_post", lambda payload: {
        "answers": {"action": {"choice": "visual_right"}},
    })
    state = {
        "privilege_level": 0,
        "information": "nonprivileged",
        "task_name": "assembly-v3",
        "control_xyz": [0.0, 0.0, 0.0],
        "gripper_command": "open",
        "gripper_opening": 1.0,
    }
    decision = policy.decide(Observation(state), "Find and place the ring.")
    assert decision.action == Action.HOLD
    assert decision.selection["execution"]["termination"] == "unavailable_target_fallback"
    assert decision.selection["selected_candidate"] == "visual_right"
