"""Motion amplitude survives execution, recording, and replay metadata."""

import json

import pytest
from PIL import Image

from jev_robo_eval.core import Action, Decision, Observation, Transition
from jev_robo_eval.recording import RecordingEnvironment
from jev_robo_eval.runner import run_episode


def test_scaled_action_reaches_recording_environment_and_trace(tmp_path):
    class Environment:
        def reset(self, *, seed=None):
            return Observation({"simulator_steps": 0}, Image.new("RGB", (64, 64)))

        def step(self, action, *, scale=1.0):
            assert action == Action.X_POS and scale == 0.25
            return Transition(Observation({"simulator_steps": 3}, Image.new("RGB", (64, 64))),
                              0.5, False, True, False)

        def close(self):
            pass

    class Policy:
        def decide(self, observation, task):
            return Decision(Action.X_POS, action_scale=0.25)

        def on_motion_executed(self, action, scale):
            self.executed = (action, scale)

    environment = RecordingEnvironment(Environment())
    trace = tmp_path / "trace.jsonl"
    policy = Policy()
    result = run_episode(environment, policy, task="move", trace_path=trace, verbose=False)
    row = json.loads(trace.read_text())
    assert result.simulator_steps == 3
    assert row["action_scale"] == row["proposed_action_scale"] == 0.25
    assert "x0.25" in environment.labels[-1]
    assert policy.executed == (Action.X_POS, 0.25)


def test_metaworld_fine_motion_reduces_displacement_with_same_physics_budget():
    pytest.importorskip("metaworld")
    from jev_robo_eval.metaworld_env import MetaWorldMT1
    displacements = []
    for scale in (0.25, 1.0):
        env = MetaWorldMT1("reach-v3", seed=2, privilege_level=1, action_repeat=3)
        try:
            before = env.reset(seed=2).state["control_xyz"][0]
            transition = env.step(Action.X_POS, scale=scale)
            displacements.append(transition.observation.state["control_xyz"][0] - before)
            assert transition.observation.state["simulator_steps"] == 3
            with pytest.raises(ValueError, match="scale"):
                env.step(Action.X_POS, scale=0)
        finally:
            env.close()
    assert 0 < displacements[0] < displacements[1]
