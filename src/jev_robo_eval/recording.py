"""Record observed frames directly when a simulator cannot guarantee replay."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .core import Action, Observation, RobotEnvironment, Transition


class RecordingEnvironment:
    """A transparent environment wrapper; recorded frames never enter policy state."""

    def __init__(self, environment: RobotEnvironment):
        self.environment = environment
        self.frames: list[Image.Image] = []
        self.labels: list[str] = []

    def _capture(self, observation: Observation, label: str) -> None:
        if observation.image is not None:
            self.frames.append(observation.image.convert("RGB").copy())
            self.labels.append(label)

    def reset(self, *, seed: int | None = None) -> Observation:
        self.frames.clear()
        self.labels.clear()
        observation = self.environment.reset(seed=seed)
        self._capture(observation, "reset")
        return observation

    def step(self, action: Action) -> Transition:
        transition = self.environment.step(action)
        self._capture(transition.observation, f"decision {len(self.frames)} | {action.value}")
        return transition

    def close(self) -> None:
        self.environment.close()

    def export(self, output: Path, *, metadata: dict, outcome: str) -> Path:
        """Export actual observations, including a partial run after an exception."""
        if not self.frames:
            raise ValueError("No camera frame was captured; cannot create a real rollout GIF")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        condition = f"L{metadata['privilege_level']}" + ("-P" if metadata.get("plan_only") else "")
        assisted = " + jitter" if metadata.get("stuck_recovery") == "jitter" else ""
        title = (f"{metadata['task']} | {condition}{assisted} | "
                 f"seed {metadata['seed']} ep {metadata.get('episode_index', 1)} | {outcome}")
        interventions = {}
        trace = metadata.get("trace")
        if trace and Path(trace).is_file():
            for line in Path(trace).read_text().splitlines():
                row = json.loads(line)
                if row.get("intervention"):
                    interventions[row["decision"]] = row["proposed_action"]
        frames = []
        for index, (source, label) in enumerate(zip(self.frames, self.labels)):
            if index in interventions:
                label += f" | ASSIST (JEV: {interventions[index]})"
            frame = Image.new("RGB", (source.width, source.height + 34), "#111827")
            frame.paste(source, (0, 34))
            draw = ImageDraw.Draw(frame)
            draw.text((5, 3), title, fill="white")
            draw.text((5, 18), label, fill="white")
            frames.append(frame)
        temporary = output.with_suffix(".gif.tmp")
        frames[0].save(temporary, format="GIF", save_all=True, append_images=frames[1:],
                       duration=100, loop=0, optimize=False)
        temporary.replace(output)
        frames[-1].save(output.with_suffix(".png"))
        output.with_suffix(".json").write_text(json.dumps({
            **metadata, "outcome": outcome, "recording": "actual_observations",
            "captured_frames": len(frames),
        }, indent=2) + "\n")
        return output
