"""Export both successful and failed episodes to one shared GIF collection."""

import argparse
import json
from pathlib import Path

from jev_robo_eval.artifacts import export_episode, rebuild_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gifs"))
    args = parser.parse_args()
    for folder in args.experiments:
        manifest = json.loads((folder / "summary.json").read_text())
        config = manifest["config"]
        for result in manifest["results"]:
            if result["status"] != "completed":
                print(f"No complete trajectory: {folder.name}/{result['trace']}: {result.get('error')}", flush=True)
                continue
            status = "success" if result["success"] else "failed"
            output = args.output_dir / folder.name / (Path(result["trace"]).stem + f"-{status}.gif")
            export_episode(folder / result["trace"], output, task=result["task"], seed=result["seed"],
                           task_index=config.get("task_index", 0), action_repeat=config["action_repeat"],
                           move_scale=config.get("move_scale", 1.0), guidance=config.get("guidance", "direct"),
                           annotate_vision=config.get("annotate_vision", False),
                           information=config.get("information", "privileged"),
                           proprio_projection=config.get("proprio_projection", False),
                           privilege_level=config.get("privilege_level"))
            rebuild_index(args.output_dir)
            print(output, flush=True)
    rebuild_index(args.output_dir)


if __name__ == "__main__":
    main()
