"""Check repeated episodes without a simulator or decision service."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from jev_robo_eval.runner import EpisodeResult


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_tasks.py"
SPEC = importlib.util.spec_from_file_location("evaluate_tasks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_repeat_episodes_have_independent_traces_gifs_and_policy_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    constructed = []

    class FakeEnvironment:
        def __init__(self, task: str, **kwargs: object) -> None:
            self.task = task
            self.kwargs = kwargs
            constructed.append(self)

    class FakePolicy:
        def __init__(self, **kwargs: object) -> None:
            self.policy_seed = kwargs["policy_seed"]
            self.action_selection = kwargs["action_selection"]
            self.temperature = kwargs["sampling_temperature"]

    def fake_run(environment: FakeEnvironment, policy: FakePolicy, **kwargs: object) -> EpisodeResult:
        trace = kwargs["trace_path"]
        seed = kwargs["seed"]
        actions = ["x_pos", "x_neg", "y_pos", "y_neg", "z_pos"]
        action = actions[policy.policy_seed - 100]
        trace.write_text(json.dumps({
            "state_before": {"task_name": environment.task, "seed": seed, "control_xyz": [0, 0, 0]},
            "action": action,
        }) + "\n")
        return EpisodeResult(success=policy.policy_seed == 102, decisions=1,
                             simulator_steps=3, reward=1.0, stopped_by="decision_limit")

    def fake_export(trace: Path, gif: Path, **kwargs: object) -> Path:
        gif.parent.mkdir(parents=True, exist_ok=True)
        assert not gif.exists()
        gif.write_bytes(b"GIF89a")
        return gif

    monkeypatch.setattr(evaluator, "MetaWorldMT1", FakeEnvironment)
    monkeypatch.setattr(evaluator, "JevPolicy", FakePolicy)
    monkeypatch.setattr(evaluator, "run_episode", fake_run)
    monkeypatch.setattr(evaluator, "export_episode", fake_export)
    monkeypatch.setattr(evaluator, "rebuild_index", lambda root: None)
    output = tmp_path / "repeat"
    argv = ["evaluate_tasks.py", "--tasks", "reach-v3", "push-v3", "--seeds", "2",
            "--episodes", "5", "--action-selection", "sample", "--policy-seed", "100",
            "--sampling-temperature", "0.7", "--output-dir", str(output),
            "--gif-dir", str(tmp_path / "gifs")]
    monkeypatch.setattr(sys, "argv", argv)
    evaluator.main()

    summary = json.loads((output / "summary.json").read_text())
    results = summary["results"]
    assert len(results) == len(constructed) == 10
    assert [(r["episode_index"], r["policy_seed"]) for r in results] == [
        (i, 99 + i) for i in range(1, 6)] * 2
    assert len({r["trace"] for r in results}) == 10
    assert len({r["gif"] for r in results}) == 10
    assert all(f"-episode{r['episode_index']}" in r["trace"] for r in results)
    assert all((output / r["trace"]).is_file() and Path(r["gif"]).is_file() for r in results)
    assert all(r["action_selection"] == "sample" and r["sampling_temperature"] == 0.7 for r in results)
    assert all(environment.kwargs["seed"] == 2 for environment in constructed)
    for group in summary["task_seed_groups"]:
        assert group["attempted"] == group["completed"] == 5
        assert group["errors"] == 0 and group["successes"] == 1
        assert group["unique_initial_state_hashes"] == 1
        assert group["unique_action_sequence_hashes"] == 5
        assert group["at_least_one_success"] is True
    with pytest.raises(FileExistsError):
        evaluator.main()


def test_default_episode_retains_legacy_trace_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class FakePolicy:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["action_selection"] == "argmax"
            assert kwargs["policy_seed"] == 0
            assert kwargs["sampling_temperature"] == 1.0

    def fake_run(environment: object, policy: object, **kwargs: object) -> EpisodeResult:
        kwargs["trace_path"].write_text(json.dumps({"state_before": {"seed": 1}, "action": "hold"}) + "\n")
        return EpisodeResult(False, 1, 3, 0.0, "decision_limit")

    monkeypatch.setattr(evaluator, "MetaWorldMT1", lambda *a, **kw: object())
    monkeypatch.setattr(evaluator, "JevPolicy", FakePolicy)
    monkeypatch.setattr(evaluator, "run_episode", fake_run)
    monkeypatch.setattr(evaluator, "export_episode", lambda *a, **kw: None)
    monkeypatch.setattr(evaluator, "rebuild_index", lambda *a, **kw: None)
    output = tmp_path / "single"
    monkeypatch.setattr(sys, "argv", ["evaluate_tasks.py", "--tasks", "reach-v3",
                                       "--output-dir", str(output)])
    evaluator.main()
    result = json.loads((output / "summary.json").read_text())["results"][0]
    assert result["trace"] == "reach-v3-text-seed1.jsonl"
    assert result["episode_index"] == 1 and result["policy_seed"] == 0
