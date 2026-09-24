"""Sampling must be reproducible and must preserve the model's distribution."""

import math

import pytest

from jev_robo_eval.core import Action
from jev_robo_eval.jev import JevPolicy


def response(probabilities=None):
    if probabilities is None:
        probabilities = {a.value: 1 / 9 for a in Action}
    return {"answers": {"action": {"choice": "x_pos", "probabilities": probabilities}},
            "image_tokens": 144}


def sequence(seed):
    policy = JevPolicy(action_selection="sample", policy_seed=seed)
    return [policy._decision_from_result(response(), {}).action for _ in range(20)]


def test_sampler_replays_seed_and_explores_without_changing_service_choice():
    assert sequence(42) == sequence(42)
    assert sequence(42) != sequence(43)
    assert len(set(sequence(42))) > 1
    policy = JevPolicy(action_selection="sample", policy_seed=0)
    result = response()
    decision = policy._decision_from_result(result, {"state": "pixels and proprioception"})
    assert decision.action != Action.X_POS
    assert decision.selection["service_choice"] == "x_pos"
    assert decision.selection["selected_action"] == decision.action.value
    assert decision.selection["phase_selection"] is None
    assert decision.probabilities == result["answers"]["action"]["probabilities"]
    assert decision.selection["effective_probabilities"] == decision.probabilities
    assert decision.usage == {"image_tokens": 144}
    assert decision.request == {"state": "pixels and proprioception"}


def test_argmax_default_does_not_draw_random_actions():
    policy = JevPolicy()
    for _ in range(10):
        decision = policy._decision_from_result(response(), {})
        assert decision.action == Action.X_POS
        assert decision.selection["method"] == "argmax"
        assert "effective_probabilities" not in decision.selection


def test_temperature_preserves_zero_mass_and_records_effective_distribution():
    probabilities = {a.value: 0.0 for a in Action}
    probabilities.update(x_pos=0.8, x_neg=0.2)
    policy = JevPolicy(action_selection="sample", sampling_temperature=0.5)
    decision = policy._decision_from_result(response(probabilities), {})
    effective = decision.selection["effective_probabilities"]
    assert effective["x_pos"] == pytest.approx(16 / 17)
    assert effective["x_neg"] == pytest.approx(1 / 17)
    assert effective["hold"] == 0
    assert math.fsum(effective.values()) == pytest.approx(1)
    assert decision.probabilities == probabilities


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1])
def test_invalid_probabilities_fail_without_uniform_fallback(bad):
    probabilities = {a.value: 0.1 for a in Action}
    probabilities["hold"] = bad
    with pytest.raises(RuntimeError, match="Sampling requires"):
        JevPolicy(action_selection="sample")._decision_from_result(response(probabilities), {})


@pytest.mark.parametrize("probabilities", [
    {}, {"x_pos": 1.0}, {a.value: 0.0 for a in Action},
    {**{a.value: 0.1 for a in Action}, "unknown": 0.1},
])
def test_missing_zero_or_unknown_distribution_fails(probabilities):
    with pytest.raises(RuntimeError, match="Sampling requires"):
        JevPolicy(action_selection="sample")._decision_from_result(response(probabilities), {})


@pytest.mark.parametrize("kwargs", [
    {"action_selection": "random"}, {"policy_seed": -1}, {"policy_seed": 0.5},
    {"sampling_temperature": 0}, {"sampling_temperature": float("nan")},
    {"sampling_temperature": float("inf")},
])
def test_invalid_sampler_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        JevPolicy(**kwargs)
