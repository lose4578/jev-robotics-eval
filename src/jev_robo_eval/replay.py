"""Render a saved MetaWorld action trace; no model requests are made.

The seed, task index, and action repeat must match the original episode.
Reward, step count, and success are checked to detect a mismatched replay.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from jev_robo_eval.core import Action
from jev_robo_eval.metaworld_env import MetaWorldMT1
from jev_robo_eval.vision_annotations import annotate_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--move-scale", type=float, default=1.0)
    parser.add_argument("--guidance", choices=["direct", "waypoints"], default="direct")
    parser.add_argument("--information", choices=["privileged", "nonprivileged"], default="privileged")
    parser.add_argument("--privilege-level", type=int, choices=range(4))
    parser.add_argument("--proprio-projection", action="store_true")
    parser.add_argument("--annotate-vision", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Trace is empty")
    env = MetaWorldMT1(args.task, mode="vision", seed=args.seed,
                       task_index=args.task_index, action_repeat=args.action_repeat,
                       image_size=384, move_scale=args.move_scale, guidance=args.guidance,
                       annotate_vision=False, information=args.information,
                       proprio_projection=args.proprio_projection, privilege_level=args.privilege_level,
                       plan_only=args.plan_only)
    frames = []

    def capture(observation, number, action, status, phase=None, recorded_state=None):
        frame = observation.image
        if args.annotate_vision:
            frame = annotate_frame(frame, env.env, env.camera_name, recorded_state or observation.state)
        canvas = Image.new("RGB", (frame.width, frame.height + 84), "#172032")
        canvas.paste(frame, (0, 84))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"{args.task} | seed {args.seed} | decision {number}/{len(rows)}", fill="white")
        state = observation.evaluation_state or observation.state
        distance = state.get("success_distance_m", state["control_to_goal_distance"])
        if args.task == "assembly-v3":
            below = state["nut_center_below_peg_top_m"]
            metric = f"XY {distance:.3f}m, below top {below:+.3f}m"
        else:
            metric = f"distance {distance:.3f} m"
        draw.text((8, 22), f"{action} | {status}", fill="#80efbd")
        draw.text((8, 37), metric, fill="#80efbd")
        if phase is not None:
            draw.text((8, 52), f"Phase: {phase}", fill="white")
        draw.text((8, 68), f"Input: {args.information} | metrics for review only", fill="#fbd38d")
        frames.append(canvas)

    try:
        capture(env.reset(seed=args.seed), 0, "reset", "running", recorded_state=rows[0].get("state_before"))
        for number, row in enumerate(rows, 1):
            transition = env.step(Action(row["action"]), scale=row.get("action_scale", 1.0))
            if (transition.success != row["success"]
                    or transition.observation.state["simulator_steps"] != row["state"]["simulator_steps"]
                    or not np.isclose(transition.reward, row["reward"], rtol=1e-6, atol=1e-6)):
                raise RuntimeError(f"Replay diverged at decision {number}; verify task/seed/action-repeat")
            status = "SUCCESS" if transition.success else "time limit" if transition.truncated else "decision limit" if number == len(rows) else "running"
            capture(transition.observation, number,
                    row["action"] + f" x{row.get('action_scale', 1.0):g}" + (" [ASSIST]" if row.get("intervention") else ""), status,
                    row["state"].get("active_waypoint", {}).get("phase"), row["state"])
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=[180] * (len(frames) - 1) + [1200], loop=0)
    frames[-1].save(args.output.with_suffix(".png"))
    print(f"Validated {len(rows)} decisions; saved {args.output}")


if __name__ == "__main__":
    main()
