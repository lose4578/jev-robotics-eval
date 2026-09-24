"""Contracts shared by environments, policies, and the evaluation loop."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from PIL import Image


class Action(str, Enum):
    X_POS = "x_pos"
    X_NEG = "x_neg"
    Y_POS = "y_pos"
    Y_NEG = "y_neg"
    Z_POS = "z_pos"
    Z_NEG = "z_neg"
    GRIP_OPEN = "grip_open"
    GRIP_CLOSE = "grip_close"
    HOLD = "hold"

    def __str__(self) -> str:
        return self.value


ACTION_DESCRIPTIONS = {
    Action.X_POS: "Move end effector in positive world X direction (meters).",
    Action.X_NEG: "Move end effector in negative world X direction (meters).",
    Action.Y_POS: "Move end effector in positive world Y direction (meters).",
    Action.Y_NEG: "Move end effector in negative world Y direction (meters).",
    Action.Z_POS: "Move end effector upward in positive world Z direction (meters).",
    Action.Z_NEG: "Move end effector downward in negative world Z direction (meters).",
    Action.GRIP_OPEN: "Open the gripper; keep this setting during later moves.",
    Action.GRIP_CLOSE: "Close the gripper; keep this setting during later moves.",
    Action.HOLD: "Keep end effector still and preserve the gripper setting.",
}


@dataclass(frozen=True)
class Observation:
    state: dict[str, Any]
    image: Image.Image | None = None
    # Full simulator state for evaluation and logging; policies only receive state.
    evaluation_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class Transition:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    action: Action
    probabilities: dict[str, float] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    # Values reported by the inference service, not inferred from the request.
    usage: dict[str, Any] = field(default_factory=dict)
    # Digest and dimensions of the exact JPEG sent with a visual decision.
    image_audit: dict[str, Any] = field(default_factory=dict)
    # Client action selection, separate from the model's original probabilities.
    selection: dict[str, Any] = field(default_factory=dict)


class RobotEnvironment(Protocol):
    def reset(self, *, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> Transition: ...
    def close(self) -> None: ...


class Policy(Protocol):
    def decide(self, observation: Observation, task: str) -> Decision: ...
