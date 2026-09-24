"""Recovery changes execution only when visible proprioception shows a stall."""

import json
import random

from jev_robo_eval.core import Action, Decision, Observation, Transition
from jev_robo_eval.recovery import StuckRecovery
from jev_robo_eval.runner import run_episode


MOVES = {Action.X_POS, Action.X_NEG, Action.Y_POS, Action.Y_NEG, Action.Z_POS, Action.Z_NEG}


def _advance(recovery, number, xyz, proposed, *, command="open"):
    state = {"control_xyz": list(xyz), "gripper_command": command,
             "scene": {"secret": "must not be inspected"}}
    choice = recovery.select(state, proposed, number)
    recovery.record_step(number, state, proposed, choice)
    return choice


def test_repeated_blocked_move_triggers_bounded_seeded_translation():
    def sequence(seed):
        recovery = StuckRecovery(seed=seed, window=6, perturb_steps=2)
        choices = [_advance(recovery, number, (0, 0, 0), Action.Z_NEG)
                   for number in range(1, 8)]
        return recovery, choices

    before = random.getstate()
    recovery, choices = sequence(19)
    assert random.getstate() == before
    assert [choice.action for choice in choices[:5]] == [Action.Z_NEG] * 5
    assert choices[5].action in MOVES - {Action.Z_NEG}
    assert choices[6].action in MOVES
    assert choices[6].action != {
        Action.X_POS: Action.X_NEG, Action.X_NEG: Action.X_POS,
        Action.Y_POS: Action.Y_NEG, Action.Y_NEG: Action.Y_POS,
        Action.Z_POS: Action.Z_NEG, Action.Z_NEG: Action.Z_POS,
    }[choices[5].action]
    assert choices[5].intervention["trigger"]["reason"] == "repeated_translation_stall"
    assert choices[5].intervention["trigger"]["tcp_positions_xyz"] == [[0, 0, 0]] * 6
    assert choices[5].intervention["rng_seed"] == 19
    assert recovery.intervention_count == 1
    assert recovery.intervention_steps == 2
    assert [choice.action for choice in sequence(19)[1]] == [choice.action for choice in choices]


def test_completed_nonzero_ab_cycle_triggers_after_six_moves():
    recovery = StuckRecovery(seed=3, window=6, perturb_steps=1)
    xyz = 0.0
    for number in range(1, 7):
        action = Action.X_POS if number % 2 else Action.X_NEG
        choice = _advance(recovery, number, (xyz, 0, 0), action)
        assert choice.intervention is None
        xyz += 0.01 if action == Action.X_POS else -0.01
    assert abs(xyz) < 1e-9
    choice = _advance(recovery, 7, (xyz, 0, 0), Action.X_POS)
    assert choice.intervention["trigger"]["reason"] == "opposite_action_cycle"
    assert choice.intervention["trigger"]["window_path_length_m"] == 0.06
    assert choice.intervention["trigger"]["window_displacement_m"] == 0.0
    assert choice.action in {Action.Y_POS, Action.Y_NEG, Action.Z_POS, Action.Z_NEG}


def test_repeated_hold_and_already_commanded_gripper_are_eligible():
    hold = StuckRecovery(seed=1, window=4, perturb_steps=1)
    for number in range(1, 4):
        assert _advance(hold, number, (0, 0, 0), Action.HOLD).intervention is None
    assert _advance(hold, 4, (0, 0, 0), Action.HOLD).intervention["trigger"]["reason"] == "repeated_hold_stall"

    gripper = StuckRecovery(seed=2, window=4, perturb_steps=1)
    _advance(gripper, 1, (0, 0, 0), Action.GRIP_CLOSE, command="open")
    for number in (2, 3, 4):
        assert _advance(gripper, number, (0, 0, 0), Action.GRIP_CLOSE,
                        command="closed").intervention is None
    assert _advance(gripper, 5, (0, 0, 0), Action.GRIP_CLOSE,
                    command="closed").intervention["trigger"]["reason"] == "repeated_gripper_stall"


def test_runner_keeps_proposed_and_actual_actions_and_calls_policy_hook(tmp_path):
    class Environment:
        def __init__(self):
            self.actions = []
            self.closed = False

        def reset(self, *, seed=None):
            return Observation({"control_xyz": [0, 0, 0], "gripper_command": "open"},
                               evaluation_state={"object_xyz": [9, 9, 9], "simulator_steps": 0})

        def step(self, action):
            self.actions.append(action)
            return Transition(
                Observation({"control_xyz": [0, 0, 0], "gripper_command": "open"},
                            evaluation_state={"object_xyz": [9, 9, 9],
                                              "simulator_steps": len(self.actions)}),
                reward=0.0, terminated=False, truncated=False, success=False,
            )

        def close(self):
            self.closed = True

    class Policy:
        def __init__(self):
            self.seen = []
            self.executed = []

        def decide(self, observation, task):
            assert observation.evaluation_state is None
            self.seen.append(observation.state)
            return Decision(Action.Z_NEG)

        def on_action_executed(self, action):
            self.executed.append(action)

    env, policy = Environment(), Policy()
    recovery = StuckRecovery(seed=11, window=4, cooldown=2,
                             max_interventions=1, perturb_steps=2)
    trace = tmp_path / "episode.jsonl"
    result = run_episode(env, policy, task="press", max_decisions=8,
                         recovery=recovery, trace_path=trace, verbose=False)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert result.intervention_count == 1
    assert result.intervention_steps == 2
    assert result.decisions == len(rows) == len(policy.seen) == 8
    assert env.closed
    assert policy.executed == env.actions
    assert all(row["proposed_action"] == "z_neg" for row in rows)
    assert [row["action"] for row in rows] == [action.value for action in env.actions]
    assert rows[3]["intervention"]["step_index"] == 1
    assert rows[4]["intervention"]["step_index"] == 2
    assert rows[3]["action"] != rows[3]["proposed_action"]
    assert rows[3]["state_before"]["object_xyz"] == [9, 9, 9]
    assert "object_xyz" not in rows[3]["intervention"]["trigger"]


def test_default_runner_has_no_intervention(tmp_path):
    class Environment:
        def reset(self, *, seed=None):
            return Observation({"control_xyz": [0, 0, 0]})

        def step(self, action):
            assert action == Action.HOLD
            return Transition(Observation({"control_xyz": [0, 0, 0]}), 0, True, False, False)

        def close(self):
            pass

    class Policy:
        def decide(self, observation, task):
            return Decision(Action.HOLD)

    trace = tmp_path / "plain.jsonl"
    result = run_episode(Environment(), Policy(), task="wait", max_decisions=2,
                         trace_path=trace, verbose=False)
    row = json.loads(trace.read_text())
    assert result.intervention_count == result.intervention_steps == 0
    assert row["action"] == row["proposed_action"] == "hold"
    assert row["intervention"] is None
