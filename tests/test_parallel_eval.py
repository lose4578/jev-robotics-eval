"""Exercise the parallel launcher without starting MetaWorld or a JEV server."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from threading import Lock
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_parallel.py"
SPEC = importlib.util.spec_from_file_location("evaluate_parallel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
parallel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel)


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_jobs_are_serial_per_endpoint_and_concurrent_across_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    active: dict[str, int] = {}
    peak: dict[str, int] = {}
    total_active = 0
    total_peak = 0
    lock = Lock()
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        nonlocal total_active, total_peak
        endpoint = _argument(command, "--jev-url")
        with lock:
            active[endpoint] = active.get(endpoint, 0) + 1
            peak[endpoint] = max(peak.get(endpoint, 0), active[endpoint])
            total_active += 1
            total_peak = max(total_peak, total_active)
            commands.append(command)
        time.sleep(0.03)
        child_dir = Path(_argument(command, "--output-dir"))
        episode_index = int(_argument(command, "--episode-index"))
        base_policy_seed = int(_argument(command, "--policy-seed"))
        task = _argument(command, "--tasks")
        seed = int(_argument(command, "--seeds"))
        (child_dir / "summary.json").write_text(json.dumps({
            "results": [{"task": task, "seed": seed, "episode_index": episode_index,
                         "policy_seed": base_policy_seed + episode_index - 1,
                         "status": "completed", "success": True,
                         "initial_state_hash": f"initial-{task}-{seed}",
                         "action_sequence_hash": f"actions-{episode_index}"}],
        }))
        with lock:
            active[endpoint] -= 1
            total_active -= 1
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(parallel.subprocess, "run", fake_run)
    output = tmp_path / "batch1"
    code = parallel.main([
        "--jev-urls", "http://gpu0:8000/v1", "http://gpu1:8000/v1",
        "--tasks", "reach-v3", "push-v3", "--seeds", "1", "2", "--episodes", "3",
        "--output-dir", str(output), "--gif-dir", str(tmp_path / "gifs"),
        "--mode", "vision", "--privilege-level", "1", "--information", "nonprivileged",
        "--sensor-policy", "staged", "--proprio-projection", "--guidance", "direct",
        "--move-scale", "0.4", "--image-size", "128", "--max-decisions", "20",
        "--action-selection", "sample", "--policy-seed", "11", "--sampling-temperature", "0.7",
    ])

    assert code == 0
    assert peak == {"http://gpu0:8000/v1": 1, "http://gpu1:8000/v1": 1}
    assert total_peak == 2
    assert len(commands) == 12
    assert all("--privilege-level" in command and "--proprio-projection" in command for command in commands)
    assert all(_argument(command, "--action-selection") == "sample" for command in commands)
    assert all(_argument(command, "--policy-seed") == "11" for command in commands)
    assert all(_argument(command, "--sampling-temperature") == "0.7" for command in commands)
    assert all(_argument(command, "--sensor-policy") == "staged" for command in commands)
    assert all(_argument(command, "--max-decisions") == "20" for command in commands)
    matrix = json.loads((output / "matrix.json").read_text())
    assert matrix["totals"] == {"jobs": 12, "completed": 12, "errors": 0, "successes": 12,
                                "artifact_errors": 0, "worker_errors": 0}
    assert all(Path(command[1]) == output / "_snapshot" / "scripts" / "evaluate_tasks.py"
               for command in commands)
    assert (output / "_snapshot" / "src" / "jev_robo_eval" / "jev.py").is_file()
    assert [job["worker"] for job in matrix["jobs"]] == [0, 1] * 6
    assert len({(job["task"], job["seed"], job["episode_index"]) for job in matrix["jobs"]}) == 12
    assert [job["policy_seed"] for job in matrix["jobs"]] == [11, 12, 13] * 4
    assert all(Path(job["summary"]).exists() for job in matrix["jobs"])
    assert all(Path(job["output_dir"]).name.startswith("batch1-") and "-episode" in Path(job["output_dir"]).name for job in matrix["jobs"])
    assert len({job["output_dir"] for job in matrix["jobs"]}) == 12
    assert len(matrix["task_seed_groups"]) == 4
    assert all(group["attempted"] == group["completed"] == group["successes"] == 3
               and group["errors"] == 0 and group["unique_initial_state_hashes"] == 1
               and group["unique_action_sequence_hashes"] == 3 for group in matrix["task_seed_groups"])
    assert not (output / "summary.json").exists()
    with pytest.raises(SystemExit):
        parallel.main(["--jev-urls", "http://gpu0:8000/v1", "--tasks", "reach-v3", "--output-dir", str(output)])


def test_child_failure_is_indexed_and_other_jobs_continue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        task = _argument(command, "--tasks")
        child_dir = Path(_argument(command, "--output-dir"))
        failed = task == "push-v3"
        (child_dir / "summary.json").write_text(json.dumps({
            "results": [{"task": task, "seed": int(_argument(command, "--seeds")),
                         "episode_index": 1, "policy_seed": int(_argument(command, "--policy-seed")),
                         "status": "error" if failed else "completed",
                         "success": False if failed else True,
                         **({"error": "fake model error"} if failed else {})}],
        }))
        return subprocess.CompletedProcess(command, 1 if failed else 0)

    monkeypatch.setattr(parallel.subprocess, "run", fake_run)
    output = tmp_path / "batch2"
    code = parallel.main([
        "--jev-urls", "http://gpu0:8000/v1", "--tasks", "push-v3", "reach-v3",
        "--output-dir", str(output),
    ])

    assert code == 1
    matrix = json.loads((output / "matrix.json").read_text())
    assert [job["status"] for job in matrix["jobs"]] == ["error", "completed"]
    assert matrix["jobs"][0]["error"] == "fake model error"
    assert matrix["totals"] == {"jobs": 2, "completed": 1, "errors": 1, "successes": 1,
                                "artifact_errors": 0, "worker_errors": 0}


def test_artifact_failure_retains_valid_episode_denominator(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        child_dir = Path(_argument(command, "--output-dir"))
        (child_dir / "summary.json").write_text(json.dumps({"results": [{
            "task": "reach-v3", "seed": 1, "episode_index": 1, "policy_seed": 0,
            "status": "completed", "success": True, "artifact_error": "disk full",
        }]}))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(parallel.subprocess, "run", fake_run)
    output = tmp_path / "artifact-failure"
    assert parallel.main(["--jev-urls", "http://gpu:8000/v1", "--tasks", "reach-v3",
                          "--output-dir", str(output)]) == 1
    matrix = json.loads((output / "matrix.json").read_text())
    assert matrix["totals"] == {"jobs": 1, "completed": 1, "errors": 0, "successes": 1,
                                "artifact_errors": 1, "worker_errors": 0}
    assert matrix["jobs"][0]["artifact_error"] == "disk full"
    assert matrix["task_seed_groups"][0]["successes"] == 1


@pytest.mark.parametrize("args", [
    ["--jev-urls", "http://gpu:8000", "http://gpu:8000", "--tasks", "reach-v3"],
    ["--jev-urls", "not-a-url", "--tasks", "reach-v3"],
    ["--jev-urls", "http://gpu:8000", "--tasks", "reach-v3", "reach-v3"],
    ["--jev-urls", "http://gpu:8000", "--tasks", "../reach-v3"],
])
def test_invalid_jobs_rejected_before_creating_output(args: list[str], tmp_path: Path) -> None:
    output = tmp_path / "batch3"
    with pytest.raises(SystemExit):
        parallel.main([*args, "--output-dir", str(output)])
    assert not output.exists()


def test_plan_and_recovery_flags_reach_worker_command(tmp_path):
    parser = parallel._parser()
    args = parser.parse_args([
        '--jev-urls', 'http://gpu:8000/v1/decision', '--tasks', 'assembly-v3',
        '--output-dir', str(tmp_path / 'conditions'), '--privilege-level', '3',
        '--plan-only', '--stuck-recovery', 'jitter', '--recovery-seed', '300',
        '--recovery-window', '8', '--recovery-steps', '1',
    ])
    parallel._validate(args, parser)
    command = parallel._command(args, args.jev_urls[0],
        {'task': 'assembly-v3', 'seed': 2, 'output_dir': tmp_path / 'child'},
        Path('evaluate_tasks.py'))
    assert '--plan-only' in command
    assert _argument(command, '--stuck-recovery') == 'jitter'
    assert _argument(command, '--recovery-window') == '8'
    assert _argument(command, '--recovery-steps') == '1'
    assert _argument(command, '--recovery-seed') == '300'


def test_benchmark_selects_complete_environment_task_set(tmp_path):
    parser = parallel._parser()
    args = parser.parse_args([
        '--jev-urls', 'http://gpu:8000/v1/decision', '--benchmark',
        '--environment', 'metaworld', '--output-dir', str(tmp_path / 'bench'),
    ])
    parallel._validate(args, parser)
    assert args.tasks == list(parallel.METAWORLD_BENCHMARK_TASKS)
    assert len(args.tasks) == 16
    assert args.action_space == "primitive"
