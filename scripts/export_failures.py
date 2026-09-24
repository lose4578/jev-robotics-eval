"""Collect failed evaluation episodes into a separate GIF folder with an index."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/failure-gifs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# 失败回放索引", "", "按实验轮次分组。GIF 从保存的动作序列重放，并校验奖励、步数和成功标志。", ""]
    replay_script = Path(__file__).with_name("replay_trace.py")
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    for folder in args.experiments:
        manifest = json.loads((folder / "summary.json").read_text())
        config = manifest["config"]
        lines += [f"## {folder.name}", ""]
        for result in manifest["results"]:
            label = f"{result['task']} / {result['mode']} / seed {result['seed']}"
            if result["status"] == "error":
                lines += [f"- {label}：运行错误，没有完整回放。`{result['error']}`"]
                continue
            if result["success"]:
                continue
            output = args.output_dir / folder.name / Path(result["trace"]).with_suffix(".gif")
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(replay_script), "--trace", str(folder / result["trace"]),
                       "--output", str(output), "--task", result["task"], "--seed", str(result["seed"]),
                       "--task-index", str(config.get("task_index", 0)),
                       "--action-repeat", str(config["action_repeat"]),
                       "--move-scale", str(config.get("move_scale", 1)),
                       "--guidance", config.get("guidance", "direct"),
                       "--information", config.get("information", "privileged")]
            if not output.exists():
                subprocess.run(command, env=env, check=True)
            relative = output.relative_to(args.output_dir)
            lines += [f"- [{label}]({relative}) — {result['decisions']} 次决策，{result['simulator_steps']} 仿真步，{result['stopped_by']}"]
            (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
        lines.append("")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
