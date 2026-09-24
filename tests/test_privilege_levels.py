import pytest

from jev_robo_eval.metaworld_env import MetaWorldMT1
from jev_robo_eval.observation_access import filter_policy_state, resolve_access


@pytest.mark.parametrize("level", range(4))
def test_levels_expose_only_their_budget(level):
    env = MetaWorldMT1("assembly-v3", mode="vision", image_size=64, privilege_level=level,
                       proprio_projection=True)
    try:
        obs = env.reset(seed=2)
        state = obs.state
        assert state["privilege_level"] == level
        assert ("object_xyz" in state) == (level >= 1)
        assert ("nut_center_xyz" in state) == (level >= 1)
        assert ("scene" in state) == (level >= 1)
        assert ("both_fingers_touch_object" in state) == (level >= 2)
        assert ("assembly_geometry_success" in state) == (level >= 2)
        assert ("object_lift_m" in state) == (level >= 2)
        assert ("active_waypoint" in state) == (level == 3)
        assert "object_xyz" in obs.evaluation_state
        assert "future_oracle_secret" not in filter_policy_state(
            {**state, "future_oracle_secret": True}, level)
    finally:
        env.close()


def test_grade_three_requires_waypoints_and_lower_grades_reject_them():
    assert resolve_access(privilege_level=3) == ("privileged", "waypoints", 3)
    with pytest.raises(ValueError):
        resolve_access(privilege_level=3, guidance="direct")
    for level in (0, 1, 2):
        with pytest.raises(ValueError):
            resolve_access(privilege_level=level, guidance="waypoints")


@pytest.mark.parametrize("level", range(4))
def test_mechanism_interaction_facts_require_l2(level):
    facts = {"window_slide_m": 0.12, "button_remaining_travel_m": 0.04,
             "unregistered_contact_fact": True}
    state = filter_policy_state(facts, level)
    for field in ("window_slide_m", "button_remaining_travel_m"):
        assert (field in state) == (level >= 2)
        if level >= 2:
            assert state[field] == facts[field]
    assert "unregistered_contact_fact" not in state
    assert not facts.keys() & filter_policy_state(facts, 3, plan_only=True).keys()
