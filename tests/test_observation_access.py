"""Checks that MetaWorld policy observations do not expose simulator truth."""

from __future__ import annotations

import numpy as np
import pytest

from jev_robo_eval.metaworld_env import MetaWorldMT1


POLICY_KEYS = {
    "task_name", "coordinate_frame", "control_point", "control_xyz",
    "robot", "gripper_opening", "gripper_command", "simulator_steps",
    "information", "privilege_level",
}
ROBOT_KEYS = {
    "hand_body_xyz", "hand_quat_wxyz", "arm_joint_order",
    "arm_joint_position_rad", "arm_joint_velocity_rad_s",
}


@pytest.mark.parametrize("kwargs", [
    {"information": "unknown"},
    {"information": "nonprivileged", "mode": "text"},
    {"information": "nonprivileged", "mode": "vision", "guidance": "waypoints"},
    {"information": "nonprivileged", "mode": "vision", "annotate_vision": True},
])
def test_invalid_observation_access_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MetaWorldMT1("reach-v3", **kwargs)


@pytest.mark.parametrize("task", [
    "reach-v3", "push-v3", "pick-place-v3", "peg-insert-side-v3",
    "shelf-place-v3", "bin-picking-v3", "assembly-v3",
])
def test_nonprivileged_observation_is_isolated_and_render_restored(task: str) -> None:
    env = MetaWorldMT1(task, mode="vision", information="nonprivileged", image_size=128)
    try:
        observation = env.reset(seed=0)
        assert set(observation.state) == POLICY_KEYS
        assert set(observation.state["robot"]) == ROBOT_KEYS
        assert observation.state["information"] == "nonprivileged"
        assert observation.evaluation_state is not None
        assert observation.evaluation_state["information"] == "nonprivileged"
        assert "goal_xyz" in observation.evaluation_state
        assert "scene" in observation.evaluation_state
        assert observation.state is not observation.evaluation_state
        assert observation.state["robot"] is not observation.evaluation_state["robot"]
        observation.state["robot"]["arm_joint_order"].append("policy_mutation")
        assert "policy_mutation" not in observation.evaluation_state["robot"]["arm_joint_order"]

        # Site markers affect the RGB image, but the mask is removed before
        # returning and neither simulation state nor geometry is changed.
        model = env.env.model
        site_rgba = model.site_rgba.copy()
        geom_rgba = model.geom_rgba.copy()
        qpos = env.env.data.qpos.copy()
        qvel = env.env.data.qvel.copy()
        assert np.array_equal(model.site_rgba, site_rgba)
        reference = np.asarray(env.env.render())
        if env.camera_name == "corner2":
            reference = np.rot90(reference, 2)
        assert np.any(np.asarray(observation.image) != reference)
        assert np.array_equal(model.site_rgba, site_rgba)
        assert np.array_equal(model.geom_rgba, geom_rgba)
        assert np.array_equal(env.env.data.qpos, qpos)
        assert np.array_equal(env.env.data.qvel, qvel)
    finally:
        env.close()


def test_site_mask_is_restored_when_render_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    env = MetaWorldMT1("reach-v3", mode="vision", information="nonprivileged")
    try:
        env.reset(seed=0)
        original = env.env.model.site_rgba.copy()

        def failing_render() -> None:
            assert np.all(env.env.model.site_rgba[:, 3] == 0)
            raise RuntimeError("render failed")

        monkeypatch.setattr(env.env, "render", failing_render)
        with pytest.raises(RuntimeError, match="render failed"):
            env._observation()
        assert np.array_equal(env.env.model.site_rgba, original)
    finally:
        env.close()


def test_render_mask_does_not_change_reward() -> None:
    env = MetaWorldMT1("reach-v3", mode="vision", information="nonprivileged")
    try:
        env.reset(seed=0)
        action = np.zeros(4, dtype=np.float32)
        reward_before, info_before = env.env.evaluate_state(env.env._get_obs(), action)
        env._observation()
        reward_after, info_after = env.env.evaluate_state(env.env._get_obs(), action)
        assert reward_after == reward_before
        assert info_after == info_before
    finally:
        env.close()


def test_projection_uses_only_fixed_camera_and_tcp() -> None:
    env = MetaWorldMT1(
        "push-v3", mode="vision", information="nonprivileged",
        proprio_projection=True, image_size=128,
    )
    try:
        before = env.reset(seed=0)
        projection = before.state["action_screen_directions"]
        assert before.state["camera_name"] == "corner2"
        assert set(projection) == {"x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg"}
        assert all(item["sample_distance_m"] == 0.05 for item in projection.values())
        assert all(len(item["delta_px"]) == 2 for item in projection.values())
        assert all(item["screen_direction"] not in {"", "unavailable"} for item in projection.values())
        assert before.evaluation_state["action_screen_directions"] == projection
        assert before.evaluation_state["action_screen_directions"] is not projection

        control_before = env.env.tcp_center.copy()
        object_before = env.env._get_pos_objects().copy()
        env.env._set_obj_xyz(object_before + np.array([0.02, 0.02, 0.0]))
        env.env._target_pos = env.env._target_pos + np.array([-0.02, 0.0, 0.0])
        env._raw_obs = env.env._get_obs()
        after = env._observation()

        assert np.array_equal(env.env.tcp_center, control_before)
        assert after.evaluation_state["goal_xyz"] != before.evaluation_state["goal_xyz"]
        assert after.evaluation_state["scene"]["interaction_points_xyz"] != before.evaluation_state["scene"]["interaction_points_xyz"]
        assert after.state["action_screen_directions"] == projection
        assert after.state["camera_name"] == before.state["camera_name"]

        # The same camera routine remains defined if the TCP leaves the view.
        camera_id = env._projection_camera_id
        behind = env.env.data.cam_xpos[camera_id] + env.env.data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
        unavailable = env._action_screen_directions(behind)
        assert all(item == {"sample_distance_m": 0.05, "delta_px": None, "screen_direction": "unavailable"} for item in unavailable.values())
    finally:
        env.close()


def test_privileged_text_keeps_original_state() -> None:
    env = MetaWorldMT1("reach-v3", mode="text")
    try:
        observation = env.reset(seed=0)
        assert observation.image is None
        assert observation.state["information"] == "privileged"
        assert "goal_xyz" in observation.state
        assert "scene" in observation.state
        assert observation.evaluation_state == observation.state
        assert observation.evaluation_state is not observation.state
        assert observation.evaluation_state["robot"] is not observation.state["robot"]
    finally:
        env.close()
