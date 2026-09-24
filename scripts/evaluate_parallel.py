"""Run a task/seed matrix across independent Jev endpoints, one worker per endpoint.

Each worker runs its assigned jobs sequentially through evaluate_tasks.py. The
matrix file indexes per-job summaries without creating a second summary.json.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Lock
import time
from urllib.parse import urlparse
from jev_robo_eval.observation_access import resolve_access
from jev_robo_eval.experiment_conditions import (add_condition_arguments,
    validate_condition_arguments, condition_config, condition_cli_arguments)

METAWORLD_BENCHMARK_TASKS = (
    "reach-v3", "push-v3", "door-open-v3", "drawer-open-v3",
    "pick-place-v3", "peg-insert-side-v3", "shelf-place-v3",
    "bin-picking-v3", "assembly-v3",
)
ROBOTWIN_BENCHMARK_TASKS = ("click_bell", "press_stapler", "move_pillbottle_pad")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev-urls", nargs="+", required=True, help="One decision endpoint per worker/GPU")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the complete registered task set for the selected environment")
    parser.add_argument("--environment", choices=("metaworld", "robotwin"), default="metaworld")
    parser.add_argument("--robotwin-root", default=os.environ.get("ROBOTWIN_ROOT", str(Path.home() / "code" / "RoboTwin")))
    parser.add_argument("--robotwin-config", default="demo_clean")
    parser.add_argument("--robotwin-initial-pose", choices=("home", "topdown"), default="home")
    parser.add_argument("--simulator-gpus", nargs="+", type=int,
                        help="Optional CUDA_VISIBLE_DEVICES value for each simulator worker")
    parser.add_argument("--active-arm", choices=("left", "right"), default="left")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif-dir", type=Path, default=Path("runs/gifs"))
    parser.add_argument("--action-space", choices=("primitive", "metaworld_atomic"),
                        help="Primitive by default; complete MetaWorld benchmark defaults to bounded intents")
    parser.add_argument("--mode", choices=("text", "vision"), default="text")
    parser.add_argument("--information", choices=("privileged", "nonprivileged"))
    parser.add_argument("--privilege-level", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--sensor-policy", choices=("direct", "staged"), default="direct")
    parser.add_argument("--action-selection", choices=("argmax", "sample"), default="argmax")
    parser.add_argument("--policy-seed", type=int, default=0, help="Base policy seed for episode 1")
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--proprio-projection", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--annotate-vision", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--move-scale", type=float, default=1.0)
    parser.add_argument("--guidance", choices=("direct", "waypoints"))
    parser.add_argument("--max-decisions", type=int, default=170)
    parser.add_argument("--task-index", type=int, default=0)
    add_condition_arguments(parser)
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.benchmark:
        if args.tasks:
            parser.error("--benchmark selects the complete task set; omit --tasks")
        args.tasks = list(METAWORLD_BENCHMARK_TASKS if args.environment == "metaworld"
                          else ROBOTWIN_BENCHMARK_TASKS)
    elif not args.tasks:
        parser.error("--tasks is required unless --benchmark is used")
    if args.action_space is None:
        args.action_space = "metaworld_atomic" if args.benchmark and args.environment == "metaworld" else "primitive"
    try:
        args.information, args.guidance, args.privilege_level = resolve_access(
            args.information or "privileged", args.guidance, args.privilege_level)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_condition_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.environment == "robotwin" and args.action_space == "metaworld_atomic":
        parser.error("metaworld_atomic is currently implemented only for MetaWorld")
    if args.action_space == "metaworld_atomic" and args.privilege_level == 0 and not args.proprio_projection:
        parser.error("L0 metaworld_atomic requires --proprio-projection")
    if len(set(args.jev_urls)) != len(args.jev_urls):
        parser.error("--jev-urls must contain distinct endpoints")
    if args.simulator_gpus is not None and (len(args.simulator_gpus) != len(args.jev_urls)
                                          or any(gpu < 0 for gpu in args.simulator_gpus)):
        parser.error("--simulator-gpus needs one nonnegative GPU index per endpoint")
    for url in args.jev_urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            parser.error(f"Invalid JEV endpoint URL: {url}")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--tasks and --seeds must not contain duplicates")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task) for task in args.tasks):
        parser.error("Task names may contain only letters, digits, dots, underscores and hyphens")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative")
    if args.episodes < 1 or args.policy_seed < 0:
        parser.error("episodes must be >= 1 and policy-seed must be >= 0")
    if not math.isfinite(args.sampling_temperature) or args.sampling_temperature <= 0:
        parser.error("sampling-temperature must be finite and > 0")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.output_dir.name):
        parser.error("Output directory basename must be a simple batch name")
    if args.image_size < 32 or args.action_repeat < 1 or args.max_decisions < 1 or args.task_index < 0:
        parser.error("image-size >= 32, action-repeat >= 1, max-decisions >= 1 and task-index >= 0 are required")
    if not 0 < args.move_scale <= 1:
        parser.error("move-scale must be in (0, 1]")


def _command(args: argparse.Namespace, endpoint: str, job: dict, script: Path) -> list[str]:
    command = [
        sys.executable, str(script), "--tasks", job["task"], "--seeds", str(job["seed"]),
        "--jev-url", endpoint, "--output-dir", str(job["output_dir"]),
        "--gif-dir", str(args.gif_dir), "--mode", args.mode,
        "--sensor-policy", args.sensor_policy, "--image-size", str(args.image_size),
        "--action-repeat", str(args.action_repeat), "--move-scale", str(args.move_scale),
        "--guidance", args.guidance, "--max-decisions", str(args.max_decisions),
        "--task-index", str(args.task_index),
        "--episodes", "1", "--action-selection", args.action_selection,
        "--policy-seed", str(args.policy_seed),
        "--sampling-temperature", str(args.sampling_temperature),
        "--environment", args.environment,
        "--action-space", args.action_space,
    ]
    if args.environment == "robotwin":
        command += ["--robotwin-root", args.robotwin_root,
                    "--robotwin-config", args.robotwin_config, "--active-arm", args.active_arm,
                    "--robotwin-initial-pose", args.robotwin_initial_pose]
    if args.episodes > 1:
        command += ["--episode-index", str(job["episode_index"])]
    if args.information is not None:
        command += ["--information", args.information]
    if args.privilege_level is not None:
        command += ["--privilege-level", str(args.privilege_level)]
    if args.proprio_projection:
        command.append("--proprio-projection")
    if args.annotate_vision:
        command.append("--annotate-vision")
    command.extend(condition_cli_arguments(args))
    return command


def _write_matrix(path: Path, matrix: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _task_seed_groups(jobs: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], dict] = {}
    hashes: dict[tuple[str, int], tuple[set[str], set[str]]] = {}
    for job in jobs:
        key = (job["task"], job["seed"])
        group = groups.setdefault(key, {"task": key[0], "seed": key[1], "attempted": 0,
                                        "completed": 0, "errors": 0, "successes": 0})
        initial, actions = hashes.setdefault(key, (set(), set()))
        group["attempted"] += job.get("status") in {"running", "completed", "error"}
        group["completed"] += job.get("status") == "completed"
        group["errors"] += job.get("status") == "error"
        group["successes"] += job.get("status") == "completed" and job.get("success") is True
        if job.get("initial_state_hash"):
            initial.add(job["initial_state_hash"])
        if job.get("action_sequence_hash"):
            actions.add(job["action_sequence_hash"])
    for key, group in groups.items():
        initial, actions = hashes[key]
        group["unique_initial_state_hashes"] = len(initial)
        group["unique_action_sequence_hashes"] = len(actions)
        group["at_least_one_success"] = group["successes"] > 0
    return list(groups.values())


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(args, parser)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.gif_dir = args.gif_dir.expanduser().resolve()
    args.robotwin_root = str(Path(args.robotwin_root).expanduser().resolve())
    if args.environment == "robotwin" and (args.annotate_vision or args.task_index != 0):
        parser.error("RoboTwin does not support MetaWorld task-index or waypoint image annotations")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")

    batch_name = args.output_dir.name
    task_seeds = [(task, seed, episode_index)
                  for task in args.tasks for seed in args.seeds
                  for episode_index in range(1, args.episodes + 1)]
    jobs = []
    for index, (task, seed, episode_index) in enumerate(task_seeds):
        suffix = f"-episode{episode_index}" if args.episodes > 1 else ""
        basename = f"{batch_name}-{task}-seed{seed}{suffix}"
        child_dir = args.output_dir / basename
        if (args.gif_dir / basename).exists():
            parser.error(f"GIF directory already exists for this job: {args.gif_dir / basename}")
        jobs.append({
            "task": task, "seed": seed, "episode_index": episode_index,
            "policy_seed": args.policy_seed + episode_index - 1,
            "worker": index % len(args.jev_urls),
            "jev_url": args.jev_urls[index % len(args.jev_urls)],
            "output_dir": str(child_dir),
            "summary": str(child_dir / "summary.json"),
            "status": "queued",
        })

    args.output_dir.mkdir(parents=True)
    matrix = {
        "started_at": _now(),
        "config": {
            "tasks": args.tasks, "seeds": args.seeds, "episodes": args.episodes,
            "benchmark": args.benchmark,
            "jev_urls": args.jev_urls,
            "output_dir": str(args.output_dir), "gif_dir": str(args.gif_dir),
            "mode": args.mode, "information": args.information,
            "environment": args.environment,
            "simulator_gpus": args.simulator_gpus,
            **({"robotwin_root": args.robotwin_root, "robotwin_config": args.robotwin_config,
                "robotwin_initial_pose": args.robotwin_initial_pose,
                "active_arm": args.active_arm} if args.environment == "robotwin" else {}),
            "privilege_level": args.privilege_level, "sensor_policy": args.sensor_policy,
            "action_space": args.action_space,
            **condition_config(args),
            "action_selection": args.action_selection, "policy_seed": args.policy_seed,
            "sampling_temperature": args.sampling_temperature,
            "proprio_projection": args.proprio_projection, "image_size": args.image_size,
            "annotate_vision": args.annotate_vision, "action_repeat": args.action_repeat,
            "move_scale": args.move_scale, "guidance": args.guidance,
            "max_decisions": args.max_decisions, "task_index": args.task_index,
        },
        "jobs": jobs,
        "task_seed_groups": _task_seed_groups(jobs),
    }
    matrix_path = args.output_dir / "matrix.json"
    lock = Lock()
    _write_matrix(matrix_path, matrix)
    # Pin executable code for the whole matrix. Subsequent edits in the shared
    # workspace must not change the policy halfway through an experiment.
    snapshot = args.output_dir / "_snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    script = snapshot / "scripts" / "evaluate_tasks.py"
    shutil.copy2(Path(__file__).resolve().with_name("evaluate_tasks.py"), script)
    source = snapshot / "src"
    shutil.copytree(Path(__file__).resolve().parents[1] / "src", source,
                    ignore=shutil.ignore_patterns("__pycache__"))
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(source) + (os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")

    def update(job: dict, **fields: object) -> None:
        with lock:
            job.update(fields)
            matrix["task_seed_groups"] = _task_seed_groups(jobs)
            _write_matrix(matrix_path, matrix)

    def worker(worker_index: int) -> None:
        endpoint = args.jev_urls[worker_index]
        worker_env = child_env.copy()
        if args.simulator_gpus is not None:
            worker_env["CUDA_VISIBLE_DEVICES"] = str(args.simulator_gpus[worker_index])
        for job in jobs:
            if job["worker"] != worker_index:
                continue
            started = time.monotonic()
            update(job, status="running", started_at=_now())
            child_dir = Path(job["output_dir"])
            child_dir.mkdir()
            command = _command(args, endpoint, job, script)
            with (child_dir / "stdout.log").open("w") as stdout, (child_dir / "stderr.log").open("w") as stderr:
                try:
                    completed = subprocess.run(command, env=worker_env, stdout=stdout, stderr=stderr, check=False)
                    returncode = completed.returncode
                    launch_error = None
                except OSError as exc:
                    returncode = None
                    launch_error = f"{type(exc).__name__}: {exc}"
            summary_path = Path(job["summary"])
            result = None
            if summary_path.exists():
                try:
                    results = json.loads(summary_path.read_text()).get("results", [])
                    if (len(results) == 1 and results[0].get("episode_index") == job["episode_index"]
                            and results[0].get("task") == job["task"]
                            and results[0].get("seed") == job["seed"]
                            and results[0].get("policy_seed") == job["policy_seed"]):
                        result = results[0]
                except (OSError, ValueError, TypeError):
                    pass
            # The simulator outcome remains valid if a later GIF export fails.
            # Keep recording/process errors separate from the task denominator.
            ok = result is not None and result.get("status") == "completed"
            fields = {
                "status": "completed" if ok else "error",
                "returncode": returncode,
                "wall_seconds": round(time.monotonic() - started, 3),
                "finished_at": _now(),
                "success": result.get("success") if result else None,
                "intervention_count": result.get("intervention_count", 0) if result else None,
                "intervention_steps": result.get("intervention_steps", 0) if result else None,
                "initial_state_hash": result.get("initial_state_hash") if result else None,
                "action_sequence_hash": result.get("action_sequence_hash") if result else None,
            }
            if launch_error:
                fields["error"] = launch_error
            elif not ok:
                fields["error"] = (result.get("error") if result else None) or "Child evaluator failed or wrote no single-result summary"
            if result and result.get("artifact_error"):
                fields["artifact_error"] = result["artifact_error"]
            if ok and returncode != 0 and "artifact_error" not in fields:
                fields["worker_error"] = f"Worker exited {returncode} after recording a completed outcome"
            update(job, **fields)
            print(f"[{worker_index}] {job['task']} seed={job['seed']} episode={job['episode_index']}: {fields['status']}", flush=True)

    with ThreadPoolExecutor(max_workers=len(args.jev_urls)) as pool:
        futures = [pool.submit(worker, index) for index in range(len(args.jev_urls))]
        for future in futures:
            future.result()

    matrix["finished_at"] = _now()
    matrix["totals"] = {
        "jobs": len(jobs),
        "completed": sum(job["status"] == "completed" for job in jobs),
        "errors": sum(job["status"] == "error" for job in jobs),
        "successes": sum(job["status"] == "completed" and job.get("success") is True for job in jobs),
        "artifact_errors": sum(bool(job.get("artifact_error")) for job in jobs),
        "worker_errors": sum(bool(job.get("worker_error")) for job in jobs),
    }
    _write_matrix(matrix_path, matrix)
    return int(any(matrix["totals"][key] for key in ("errors", "artifact_errors", "worker_errors")))


if __name__ == "__main__":
    raise SystemExit(main())
