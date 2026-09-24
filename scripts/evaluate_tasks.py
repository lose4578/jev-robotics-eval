"""Run a reproducible task/seed matrix and retain per-episode traces."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone

from jev_robo_eval.jev import JevPolicy
from jev_robo_eval.metaworld_env import MetaWorldMT1
from jev_robo_eval.runner import run_episode
from jev_robo_eval.tasks import task_goal
from jev_robo_eval.artifacts import export_episode, rebuild_index
from jev_robo_eval.observation_access import resolve_access
from jev_robo_eval.recording import RecordingEnvironment
from jev_robo_eval.experiment_conditions import (add_condition_arguments,
    validate_condition_arguments, recovery_for_episode, condition_config)


def _trace_hashes(trace: Path) -> dict[str, str]:
    """Hash the initial simulator snapshot and actions after an episode ends."""
    actions = []
    initial_state = None
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if initial_state is None:
                initial_state = row["state_before"]
            actions.append((row["action"], row.get("action_scale", 1.0)))
    if initial_state is None:
        return {}

    def digest(value: object) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {"initial_state_hash": digest(initial_state), "action_sequence_hash": digest(actions)}


def _robotwin_runtime(root: str) -> dict:
    runtime = {"python": sys.version, "upstream_root": root, "packages": {},
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for name in ("sapien", "mplib", "torch", "numpy", "Pillow"):
        try:
            runtime["packages"][name] = version(name)
        except PackageNotFoundError:
            runtime["packages"][name] = None
    for key, command in (("upstream_revision", ["rev-parse", "HEAD"]),
                         ("upstream_tracked_changes", ["status", "--porcelain", "--untracked-files=no"])):
        try:
            runtime[key] = subprocess.check_output(["git", "-C", root, *command],
                text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            runtime[key] = None
            runtime[key + "_error"] = type(exc).__name__
    return runtime


def _task_seed_groups(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], dict] = {}
    hashes: dict[tuple[str, int], tuple[set[str], set[str]]] = {}
    for result in results:
        key = (result["task"], result["seed"])
        group = groups.setdefault(key, {"task": key[0], "seed": key[1], "attempted": 0,
                                        "completed": 0, "errors": 0, "successes": 0})
        initial, actions = hashes.setdefault(key, (set(), set()))
        group["attempted"] += 1
        group["completed"] += result.get("status") == "completed"
        group["errors"] += result.get("status") == "error"
        group["successes"] += result.get("status") == "completed" and result.get("success") is True
        if result.get("initial_state_hash"):
            initial.add(result["initial_state_hash"])
        if result.get("action_sequence_hash"):
            actions.add(result["action_sequence_hash"])
    for key, group in groups.items():
        initial, actions = hashes[key]
        group["unique_initial_state_hashes"] = len(initial)
        group["unique_action_sequence_hashes"] = len(actions)
        group["at_least_one_success"] = group["successes"] > 0
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--environment", choices=["metaworld", "robotwin"], default="metaworld")
    parser.add_argument("--robotwin-root", default=os.environ.get("ROBOTWIN_ROOT", str(Path.home() / "code" / "RoboTwin")))
    parser.add_argument("--robotwin-config", default="demo_clean")
    parser.add_argument("--robotwin-initial-pose", choices=["home", "topdown"], default="home")
    parser.add_argument("--active-arm", choices=["left", "right"], default="left")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-index", type=int, help="First episode index; used by parallel workers")
    parser.add_argument("--mode", choices=["text", "vision"], default="text")
    parser.add_argument("--information", choices=["privileged", "nonprivileged"], default="privileged")
    parser.add_argument("--privilege-level", type=int, choices=range(4))
    parser.add_argument("--sensor-policy", choices=["direct", "staged"], default="direct")
    parser.add_argument("--action-selection", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--policy-seed", type=int, default=0, help="Base policy seed for episode 1")
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--proprio-projection", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--annotate-vision", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--move-scale", type=float, default=1.0)
    parser.add_argument("--guidance", choices=["direct", "waypoints"])
    parser.add_argument("--max-decisions", type=int, default=170)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--jev-url")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif-dir", type=Path, default=Path("runs/gifs"))
    parser.add_argument("--action-space", choices=["primitive", "metaworld_atomic"], default="primitive")
    add_condition_arguments(parser)
    args = parser.parse_args()
    # Upstream simulators may change cwd while loading relative asset paths.
    args.output_dir = args.output_dir.expanduser().resolve()
    args.gif_dir = args.gif_dir.expanduser().resolve()
    args.robotwin_root = str(Path(args.robotwin_root).expanduser().resolve())
    if args.environment == "robotwin" and args.action_space == "metaworld_atomic":
        parser.error("metaworld_atomic is currently implemented only for MetaWorld")
    if args.action_space == "metaworld_atomic" and not args.proprio_projection and args.privilege_level == 0:
        parser.error("L0 metaworld_atomic requires --proprio-projection for calibrated visual lowering")
    if args.environment == "robotwin" and (args.annotate_vision or args.task_index != 0):
        parser.error("RoboTwin does not support MetaWorld task-index or waypoint image annotations")
    if args.episodes < 1 or (args.episode_index is not None and args.episode_index < 1):
        parser.error("episodes and episode-index must be >= 1")
    if args.policy_seed < 0 or not math.isfinite(args.sampling_temperature) or args.sampling_temperature <= 0:
        parser.error("policy-seed must be >= 0 and sampling-temperature must be finite and > 0")
    args.information, args.guidance, args.privilege_level = resolve_access(
        args.information, args.guidance, args.privilege_level)
    validate_condition_arguments(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("This experiment already has a summary; use a new output directory")
    shutil.copytree(Path(__file__).resolve().parents[1] / "src" / "jev_robo_eval",
                    args.output_dir / "source", ignore=shutil.ignore_patterns("__pycache__"))
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(),
                "config": {**vars(args), "output_dir": str(args.output_dir), "gif_dir": str(args.gif_dir)}, "results": []}
    if args.environment == "robotwin":
        manifest["runtime"] = _robotwin_runtime(args.robotwin_root)
    for task in args.tasks:
        for seed in args.seeds:
            first_index = args.episode_index or 1
            for episode_index in range(first_index, first_index + args.episodes):
                started = time.monotonic()
                suffix = f"-episode{episode_index}" if args.episodes > 1 or args.episode_index is not None else ""
                trace = args.output_dir / f"{task}-{args.mode}-seed{seed}{suffix}.jsonl"
                if trace.exists():
                    raise FileExistsError(f"Refusing to overwrite {trace}; use a new output directory")
                effective_policy_seed = args.policy_seed + episode_index - 1
                result = {"task": task, "seed": seed, "episode_index": episode_index,
                          "environment": args.environment,
                          "policy_seed": effective_policy_seed, "action_selection": args.action_selection,
                          "sampling_temperature": args.sampling_temperature,
                          "mode": args.mode, "information": args.information,
                          "privilege_level": args.privilege_level, "trace": trace.name,
                          "action_space": args.action_space,
                          **condition_config(args),
                          "recovery_seed": args.recovery_seed + episode_index - 1}
                recording = None
                recovery = None
                try:
                    recovery = recovery_for_episode(args, episode_index)
                    policy = JevPolicy(mode=args.mode, url=args.jev_url, api_key=os.environ.get("JEV_API_KEY"),
                                       sensor_policy=args.sensor_policy, action_selection=args.action_selection,
                                       policy_seed=effective_policy_seed,
                                       sampling_temperature=args.sampling_temperature,
                                       action_space=args.action_space,
                                       action_granularity=args.action_granularity)
                    if args.environment == "robotwin":
                        from jev_robo_eval.robotwin_env import RoboTwinEnvironment
                        env = RoboTwinEnvironment(task, root=args.robotwin_root,
                            task_config=args.robotwin_config, active_arm=args.active_arm,
                            initial_pose=args.robotwin_initial_pose,
                            seed=seed, mode=args.mode, privilege_level=args.privilege_level,
                            image_size=args.image_size, action_repeat=args.action_repeat,
                            move_scale=args.move_scale, proprio_projection=args.proprio_projection,
                            plan_only=args.plan_only)
                        recording = RecordingEnvironment(env)
                        env = recording
                    else:
                        env = MetaWorldMT1(task, seed=seed, task_index=args.task_index,
                                       mode=args.mode, action_repeat=args.action_repeat,
                                       move_scale=args.move_scale, guidance=args.guidance, image_size=args.image_size,
                                       annotate_vision=args.annotate_vision, information=args.information,
                                       proprio_projection=args.proprio_projection, privilege_level=args.privilege_level,
                                       plan_only=args.plan_only)
                        if args.mode == "vision":
                            # Keep the exact observed frames even if a model call fails.
                            recording = RecordingEnvironment(env)
                            env = recording
                    episode = run_episode(env, policy, task=task_goal(task, information=args.information), seed=seed,
                                          max_decisions=args.max_decisions, trace_path=trace, verbose=False,
                                          recovery=recovery)
                    result.update(status="completed", **asdict(episode))
                except Exception as exc:
                    result.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    result.update(intervention_count=recovery.intervention_count if recovery else 0,
                                  intervention_steps=recovery.intervention_steps if recovery else 0)
                if trace.exists():
                    try:
                        result.update(_trace_hashes(trace))
                    except (OSError, ValueError, KeyError) as exc:
                        result["trace_hash_error"] = f"{type(exc).__name__}: {exc}"
                result["wall_seconds"] = round(time.monotonic() - started, 3)
                if result["status"] == "completed" or (recording is not None and recording.frames):
                    status = ("error" if result["status"] == "error" else
                              "success" if result.get("success") else "failed")
                    gif = args.gif_dir / args.output_dir.name / f"{trace.stem}-{status}.gif"
                    try:
                        if recording is not None:
                            recording.export(gif, metadata={**manifest["config"], **result,
                                             "trace": str(trace)},
                                             outcome=status)
                        else:
                            export_episode(trace, gif, task=task, seed=seed, task_index=args.task_index,
                                       action_repeat=args.action_repeat, move_scale=args.move_scale, guidance=args.guidance,
                                       annotate_vision=args.annotate_vision, information=args.information,
                                       proprio_projection=args.proprio_projection, privilege_level=args.privilege_level,
                                       plan_only=args.plan_only,
                                       metadata={**manifest["config"], **result, "trace": str(trace)})
                        result["gif"] = str(gif)
                        rebuild_index(args.gif_dir)
                    except Exception as exc:
                        result["artifact_error"] = f"{type(exc).__name__}: {exc}"
                manifest["results"].append(result)
                manifest["task_seed_groups"] = _task_seed_groups(manifest["results"])
                (args.output_dir / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(json.dumps(result), flush=True)
    if any(result["status"] == "error" or "artifact_error" in result for result in manifest["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
