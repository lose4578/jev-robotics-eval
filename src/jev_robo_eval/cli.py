"""Command-line runner for a MetaWorld MT1 episode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from .jev import JevPolicy
from .metaworld_env import MetaWorldMT1
from .runner import run_episode
from .tasks import task_goal
from .artifacts import export_episode, rebuild_index
from .observation_access import resolve_access
from .experiment_conditions import (add_condition_arguments, validate_condition_arguments,
                                    recovery_for_episode, condition_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a JEV controlled MetaWorld MT1 episode")
    parser.add_argument("--task", default="reach-v3", help="MetaWorld MT1 task name")
    parser.add_argument("--goal", help="Natural-language task instruction sent to Jev")
    parser.add_argument("--mode", choices=("text", "vision"), default="text")
    parser.add_argument("--information", choices=("privileged", "nonprivileged"), default="privileged")
    parser.add_argument("--privilege-level", type=int, choices=range(4))
    parser.add_argument("--sensor-policy", choices=("direct", "staged"), default="direct")
    parser.add_argument("--action-selection", choices=("argmax", "sample"), default="argmax")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--proprio-projection", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--move-scale", type=float, default=1.0)
    parser.add_argument("--guidance", choices=("direct", "waypoints"))
    parser.add_argument("--max-decisions", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--annotate-vision", action="store_true")
    parser.add_argument("--camera", default="corner2")
    parser.add_argument("--jev-url", help="Override text or vision decision endpoint")
    parser.add_argument("--trace", type=Path, help="Write per-decision JSONL trace")
    parser.add_argument("--gif-dir", type=Path, default=Path("runs/gifs"))
    parser.add_argument("--action-space", choices=("primitive", "metaworld_atomic"), default="primitive")
    add_condition_arguments(parser)
    args = parser.parse_args()
    args.information, args.guidance, args.privilege_level = resolve_access(
        args.information, args.guidance, args.privilege_level)
    validate_condition_arguments(args)
    run_name = f"{args.task}-{args.mode}-L{args.privilege_level}-seed{args.seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    trace = args.trace or Path("runs/episodes") / f"{run_name}.jsonl"
    goal = args.goal or task_goal(args.task, information=args.information)
    environment = MetaWorldMT1(
        args.task,
        mode=args.mode,
        seed=args.seed,
        task_index=args.task_index,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        camera_name=args.camera,
        move_scale=args.move_scale,
        guidance=args.guidance,
        annotate_vision=args.annotate_vision,
        information=args.information,
        proprio_projection=args.proprio_projection,
        privilege_level=args.privilege_level,
        plan_only=args.plan_only,
    )
    policy = JevPolicy(mode=args.mode, url=args.jev_url, api_key=os.environ.get("JEV_API_KEY"),
                       sensor_policy=args.sensor_policy, action_selection=args.action_selection,
                       policy_seed=args.policy_seed, sampling_temperature=args.sampling_temperature,
                       action_space=args.action_space)
    result = run_episode(
        environment,
        policy,
        task=goal,
        max_decisions=args.max_decisions,
        seed=args.seed,
        trace_path=trace,
        recovery=recovery_for_episode(args),
    )
    gif = args.gif_dir / "cli" / f"{run_name}-{'success' if result.success else 'failed'}.gif"
    export_episode(trace, gif, task=args.task, seed=args.seed, task_index=args.task_index,
                   action_repeat=args.action_repeat, move_scale=args.move_scale, guidance=args.guidance,
                   annotate_vision=args.annotate_vision, information=args.information,
                   proprio_projection=args.proprio_projection, privilege_level=args.privilege_level,
                   plan_only=args.plan_only, metadata={**condition_config(args), **result.__dict__})
    rebuild_index(args.gif_dir)
    print(json.dumps({**result.__dict__, "information": args.information, "trace": str(trace), "gif": str(gif)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
