"""Recording survives runner cleanup and includes unsuccessful trajectories."""

import json

from PIL import Image
import pytest

from jev_robo_eval.core import Action, Decision, Observation, Transition
from jev_robo_eval.recording import RecordingEnvironment
from jev_robo_eval.runner import run_episode


@pytest.mark.parametrize("success", [True, False])
def test_actual_frames_export_after_environment_closed(tmp_path, success):
    class Environment:
        closed = False

        def reset(self, *, seed=None):
            return Observation({"simulator_steps": 0}, Image.new("RGB", (384, 32), "red"))

        def step(self, action):
            return Transition(Observation({"simulator_steps": 1}, Image.new("RGB", (384, 32), "blue")),
                              0.0, False, not success, success)

        def close(self):
            self.closed = True

    class Policy:
        def decide(self, observation, task):
            return Decision(Action.X_POS)

    env = RecordingEnvironment(Environment())
    result = run_episode(env, Policy(), task="move", verbose=False)
    assert result.success == success
    assert env.environment.closed
    output = tmp_path / ("success.gif" if success else "failed.gif")
    env.export(output, metadata={"task": "test", "seed": 2, "privilege_level": 0},
               outcome="success" if success else "failed")
    with Image.open(output) as gif:
        assert gif.n_frames == 2
        assert gif.convert("RGB").getpixel((0, 34)) == (255, 0, 0)
        gif.seek(1)
        assert gif.convert("RGB").getpixel((0, 34)) == (0, 0, 255)
    assert json.loads(output.with_suffix(".json").read_text())["recording"] == "actual_observations"
    with pytest.raises(FileExistsError):
        env.export(output, metadata={}, outcome="failed")


def test_missing_frames_do_not_create_a_fabricated_replay(tmp_path):
    env = RecordingEnvironment(None)
    with pytest.raises(ValueError, match="No camera frame"):
        env.export(tmp_path / "empty.gif", metadata={}, outcome="error")
    assert not list(tmp_path.iterdir())
