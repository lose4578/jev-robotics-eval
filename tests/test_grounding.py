"""Geometry evidence must describe allowed targets without selecting an action."""

import pytest

from jev_robo_eval.core import Action
from jev_robo_eval.grounding import candidate_grounding, compact_action_state, compact_phase_state, phase_target_role, target_evidence


def state(level=1, **extra):
    return {"privilege_level": level, "control_xyz": [0.0, 0.0, 0.0],
            "object_xyz": [0.1, 0.3, 0.2], "goal_xyz": [0.1, 0.3, 0.1], **extra}


def test_l0_never_derives_pose_even_when_hidden_fields_are_present():
    assert target_evidence(state(level=0), "pressing") == {}
    assert target_evidence(state(level=0, goal_xyz=[100.0] * 3), "pressing") == {}


@pytest.mark.parametrize("level", [1, 2])
def test_contact_roles_follow_adapter_semantics_without_inventing_an_endpoint(level):
    mw = target_evidence(state(level=level, environment="metaworld"), "pressing")
    rt = target_evidence(state(level=level, environment="robotwin"), "pressing")
    assert mw["contact"]["source"] == "object_xyz"
    assert mw["destination"]["source"] == "goal_xyz"
    assert rt["contact"]["source"] == "goal_xyz"
    assert "destination" not in rt
    assert phase_target_role("pressing", "press", "robotwin") == "contact"
    assert mw["contact"]["delta_xyz"] == [0.1, 0.3, 0.2]
    assert mw["contact"]["provenance"] == "derived_from_permitted_pose"


def test_signed_effects_describe_both_choices_and_still_point_up_for_low_tcp():
    evidence = target_evidence(state(), "pressing")
    for phase in ("approach", "press"):
        toward = candidate_grounding(Action.Z_POS, "pressing", phase, evidence)
        away = candidate_grounding(Action.Z_NEG, "pressing", phase, evidence)
        assert "initially reduces" in toward
        assert "increases" in away
    assert "requires engagement" in candidate_grounding(Action.Y_POS, "pressing", "press", evidence)


def test_unknown_fields_cannot_change_grounding_and_bad_pose_is_not_replaced():
    expected = target_evidence(state(), "pick_place")
    assert target_evidence(state(future_truth=[10, 20, 30], object_minus_control_xyz=[9, 9, 9]), "pick_place") == expected
    assert "contact" not in target_evidence(state(object_xyz=[float("nan"), 0, 0]), "pick_place")
    assert "contact" not in target_evidence(state(object_xyz=None), "pick_place")


def test_empty_grasp_phases_use_contact_and_lift_has_no_fabricated_waypoint():
    for phase in ("approach", "lower", "grasp", "recover"):
        assert phase_target_role("pick_place", phase) == "contact"
    assert phase_target_role("pick_place", "transfer") == "destination"
    assert phase_target_role("pick_place", "lift") is None
    text = candidate_grounding(Action.Y_POS, "pick_place", "approach", {})
    assert "visible contact surface" in text
    assert " m" not in text


def test_nominal_calibration_marks_possible_crossing_without_removing_a_choice():
    evidence = target_evidence(state(object_xyz=[0.008, 0.3, 0.2]), "pressing")
    fine = candidate_grounding(Action.X_POS, "pressing", "approach", evidence,
                               scale=0.25, nominal_motion_step_m=0.02)
    coarse = candidate_grounding(Action.X_POS, "pressing", "approach", evidence,
                                 scale=1.0, nominal_motion_step_m=0.02)
    assert "0.005 m" in fine and "may cross" not in fine
    assert "0.02 m" in coarse and "may cross" in coarse
    visual = candidate_grounding(Action.X_POS, "pressing", "approach", {},
                                 scale=0.25, nominal_motion_step_m=0.02)
    assert "0.005 m" in visual and "may cross" not in visual


@pytest.mark.parametrize("level", [0, 1, 2])
def test_compact_phase_excludes_intention_history_and_bulk_context(level):
    visible = state(level=level, task="Press down immediately", task_semantics="press",
                    gripper_command="open", previous_inferred_phase="press", previous_action="z_neg",
                    scene={"geometry": "large scene"}, robot={"joint_vectors": [0.0] * 7},
                    action_screen_directions={"z_neg": "down"}, both_fingers_touch_object=True)
    compact = compact_phase_state(visible, "button-press-topdown-v3", "pressing", target_evidence(visible, "pressing"))
    allowed = {"current_contact_surface", "robot_tcp_xyz", "gripper_command"}
    if level >= 1:
        allowed.add("current_contact_xyz")
    if level == 2:
        allowed.add("both_fingers_touch_object")
    assert set(compact) == allowed
    assert compact["current_contact_surface"] == "red button top"
    action_state = compact_action_state(visible)
    assert "scene" not in action_state and "robot" not in action_state and "task_semantics" not in action_state
    assert action_state["task"] == "Press down immediately"
    assert action_state["action_screen_directions"] == {"z_neg": "down"}


@pytest.mark.parametrize("family", ["pick_place", "reach"])
def test_compact_phase_keeps_destination_only_for_required_families(family):
    visible = state(gripper_command="closed")
    compact = compact_phase_state(visible, "reach-v3" if family == "reach" else "pick-place-v3",
                                   family, target_evidence(visible, family))
    assert compact["destination_xyz"] == visible["goal_xyz"]
