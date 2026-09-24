"""Bounded intent actions for environments with a finite action vocabulary.

The legacy :class:`~jev_robo_eval.core.Action` enum remains the execution
interface used by the runner.  This module adds a deliberately small middle
layer: JEV chooses one named candidate, and an environment adapter lowers that
candidate to at most one legacy action.  Candidate generation and lowering use
only the policy-visible state passed to them; ``evaluation_state`` is never
accepted as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

from .core import Action


ATOMIC_PROTOCOL = "bounded_intent_v1"
METAWORLD_ATOMIC_VERSION = "metaworld-v1"


class AtomicActionId(str, Enum):
    """Stable IDs presented to JEV for the first MetaWorld pilot."""

    ALIGN_XY = "align_xy"
    ADJUST_Z = "adjust_z"
    LIFT = "lift"
    TRACK_WAYPOINT = "track_waypoint"
    GRIP_OPEN = "grip_open"
    GRIP_CLOSE = "grip_close"
    HOLD = "hold"
    # L0 has no policy-visible 3-D target.  These candidates express a
    # camera-relative direction; calibration lowers them to a world-axis
    # primitive without inventing a simulator coordinate.
    VISUAL_LEFT = "visual_left"
    VISUAL_RIGHT = "visual_right"
    VISUAL_UP = "visual_up"
    VISUAL_DOWN = "visual_down"

    def __str__(self) -> str:
        return self.value


# Short alias for callers that want to refer to the candidate vocabulary as an
# action enum.  ``Action`` itself is intentionally unchanged for baseline
# compatibility.
AtomicAction = AtomicActionId


@dataclass(frozen=True)
class AtomicActionCandidate:
    """A finite, bounded action choice.

    ``target`` is a reference, not an untrusted simulator value.  For example,
    ``{"id": "current_target", "source_key": "goal_xyz"}`` tells the
    executor which policy-visible estimate may be used; it does not put a new
    continuous control value in JEV's choice space.
    """

    id: str
    intent: str
    target: Mapping[str, Any]
    parameters: Mapping[str, Any]
    termination: Mapping[str, Any]
    provenance: str
    requires: tuple[str, ...] = ("cartesian_translation",)
    generator_version: str = METAWORLD_ATOMIC_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("atomic candidate id must be a non-empty string")
        if not isinstance(self.intent, str) or not self.intent:
            raise ValueError("atomic candidate intent must be a non-empty string")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("atomic candidate provenance must be a non-empty string")
        if not isinstance(self.target, Mapping):
            raise TypeError("atomic candidate target must be a mapping")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("atomic candidate parameters must be a mapping")
        if not isinstance(self.termination, Mapping):
            raise TypeError("atomic candidate termination must be a mapping")
        max_steps = self.parameters.get("max_executor_steps", 1)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps != 1:
            raise ValueError("atomic candidates must be bounded to one executor step")
        if not isinstance(self.requires, tuple) or not all(isinstance(item, str) and item for item in self.requires):
            raise TypeError("atomic candidate requires must be a tuple of non-empty strings")

    @property
    def action_id(self) -> AtomicActionId:
        try:
            return AtomicActionId(self.id)
        except ValueError as exc:
            raise ValueError(f"unknown MetaWorld atomic candidate: {self.id!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe descriptor suitable for a JEV request/trace."""
        return {
            "id": self.id,
            "intent": self.intent,
            "target": dict(self.target),
            "parameters": dict(self.parameters),
            "termination": dict(self.termination),
            "provenance": self.provenance,
            "requires": list(self.requires),
            "generator_version": self.generator_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AtomicActionCandidate":
        """Parse a descriptor while retaining the one-step safety invariant."""
        if not isinstance(value, Mapping):
            raise TypeError("atomic candidate descriptor must be a mapping")
        return cls(
            id=str(value["id"]),
            intent=str(value["intent"]),
            target=dict(value.get("target", {})),
            parameters=dict(value.get("parameters", {})),
            termination=dict(value.get("termination", {})),
            provenance=str(value["provenance"]),
            requires=tuple(value.get("requires", ("cartesian_translation",))),
            generator_version=str(value.get("generator_version", METAWORLD_ATOMIC_VERSION)),
        )


# Names are explicit and ordered so that requests/traces are reproducible.
METAWORLD_ATOMIC_IDS: tuple[AtomicActionId, ...] = tuple(AtomicActionId)

# Transfer candidates are meaningful after the gripper has been commanded
# closed. This is an actuator-visible scheduling rule, not an evaluation fact.
_TRANSFER_TASKS = {
    "pick-place-v3", "peg-insert-side-v3", "shelf-place-v3",
    "bin-picking-v3", "assembly-v3",
}
_CONTACT_TASKS = {"push-v3", "door-open-v3", "drawer-open-v3"}


def _target_reference(state: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the best *policy-visible* target without copying coordinates."""
    if "active_waypoint" in state:
        waypoint = state["active_waypoint"]
        if isinstance(waypoint, Mapping) and "target_xyz" in waypoint:
            return {"id": "current_waypoint", "frame": "world",
                    "source_key": "active_waypoint.target_xyz",
                    "provenance": "oracle_waypoint"}
    for key in ("atomic_target_xyz", "visible_target_xyz", "target_xyz", "goal_xyz", "object_xyz"):
        if key in state:
            source = "policy_visible_state" if key != "visible_target_xyz" else "rgb_observation"
            return {"id": "current_target", "frame": "world",
                    "source_key": key, "provenance": source}
    # L0 may have only RGB.  The candidate remains selectable; an upstream
    # visual estimator can add visible_target_xyz before lowering it.  The
    # executor reports a bounded unavailable result if it is still absent.
    return {"id": "visible_target", "frame": "camera_or_world",
            "provenance": "rgb_observation"}


def _candidate(*, action_id: AtomicActionId, intent: str,
               target: Mapping[str, Any], axes: tuple[str, ...] = (),
               provenance: str | None = None,
               description: str | None = None) -> AtomicActionCandidate:
    parameters: dict[str, Any] = {
        "axes": list(axes),
        "max_executor_steps": 1,
        "selection_uses_continuous_value": False,
    }
    if description:
        parameters["description"] = description
    termination = {
        "max_executor_steps": 1,
        "return_after_step": True,
        "return_on_target_revision": True,
        "return_on_budget_exhaustion": True,
    }
    if action_id in {AtomicActionId.GRIP_OPEN, AtomicActionId.GRIP_CLOSE}:
        requires = ("gripper",)
    elif action_id == AtomicActionId.HOLD:
        requires = ("cartesian_translation", "gripper")
    else:
        requires = ("cartesian_translation",)
    return AtomicActionCandidate(
        id=action_id.value,
        intent=intent,
        target=dict(target),
        parameters=parameters,
        termination=termination,
        provenance=provenance or str(target.get("provenance", "policy_visible_state")),
        requires=requires,
    )


def metaworld_atomic_candidates(state: Mapping[str, Any]) -> tuple[AtomicActionCandidate, ...]:
    """Generate the fixed MetaWorld candidate set from visible state only.

    This function intentionally accepts a state mapping rather than an
    ``Observation``.  Passing an evaluation-state field is treated as a caller
    error, which makes accidental truth leakage fail closed in tests and in
    future adapters.
    """
    if not isinstance(state, Mapping):
        raise TypeError("policy-visible state must be a mapping")
    if "evaluation_state" in state:
        raise ValueError("atomic candidate generation cannot inspect evaluation_state")
    target = _target_reference(state)
    targetless = "source_key" not in target
    gripper_target = {"id": "active_gripper", "frame": "tool",
                      "provenance": "robot_proprioception"}
    hold_target = {"id": "control_point", "frame": "world",
                   "provenance": "robot_proprioception"}
    if targetless:
        camera_target = {"id": "visible_target", "frame": "camera",
                         "provenance": "rgb_observation"}
        visual_candidates = (
            (AtomicActionId.VISUAL_LEFT, "visual_direction", "Move the gripper in the calibrated screen-left direction."),
            (AtomicActionId.VISUAL_RIGHT, "visual_direction", "Move the gripper in the calibrated screen-right direction."),
            (AtomicActionId.VISUAL_UP, "visual_direction", "Move the gripper in the calibrated screen-up direction."),
            (AtomicActionId.VISUAL_DOWN, "visual_direction", "Move the gripper in the calibrated screen-down direction."),
            (AtomicActionId.GRIP_OPEN, "release", "Open the gripper for one bounded actuator step."),
            (AtomicActionId.GRIP_CLOSE, "engage", "Close the gripper for one bounded actuator step."),
            (AtomicActionId.HOLD, "observe", "Keep the current pose and gripper setting for one step."),
        )
        candidates = tuple(
            _candidate(action_id=action_id, intent=intent,
                       target=(camera_target if action_id in {
                           AtomicActionId.VISUAL_LEFT, AtomicActionId.VISUAL_RIGHT,
                           AtomicActionId.VISUAL_UP, AtomicActionId.VISUAL_DOWN,
                       } else gripper_target if action_id in {
                           AtomicActionId.GRIP_OPEN, AtomicActionId.GRIP_CLOSE,
                       } else hold_target),
                       axes=(),
                       provenance=("rgb_observation" if action_id in {
                           AtomicActionId.VISUAL_LEFT, AtomicActionId.VISUAL_RIGHT,
                           AtomicActionId.VISUAL_UP, AtomicActionId.VISUAL_DOWN,
                       } else "robot_proprioception"),
                       description=description)
            for action_id, intent, description in visual_candidates
        )
        gripper_command = str(state.get("gripper_command", "")).lower()
        if gripper_command == "open":
            candidates = tuple(item for item in candidates if item.id != AtomicActionId.GRIP_OPEN.value)
        elif gripper_command == "closed":
            candidates = tuple(item for item in candidates if item.id != AtomicActionId.GRIP_CLOSE.value)
        return candidates
    # The simulator's object interaction point is the reliable grasp target.
    # Assembly also exposes ``nut_center_xyz`` for later hole alignment, but
    # that site is not the ring body where the gripper should close.
    object_key = "object_xyz"
    object_target = ({"id": "current_object", "frame": "world",
                      "source_key": object_key, "provenance": "policy_visible_state"}
                     if object_key in state else target)
    goal_key = "atomic_target_xyz" if "atomic_target_xyz" in state else "goal_xyz"
    goal_target = ({"id": "current_goal", "frame": "world",
                    "source_key": goal_key, "provenance": "policy_visible_state"}
                   if goal_key in state else target)
    candidates = [
        _candidate(action_id=AtomicActionId.ALIGN_XY, intent="align",
                   target=object_target, axes=("x", "y"),
                   description="Align the gripper horizontally with the visible object by one signed axis step."),
        _candidate(action_id=AtomicActionId.ADJUST_Z, intent="adjust_height",
                   target=object_target, axes=("z",),
                   description="Adjust height toward the visible object by one signed world Z step."),
        _candidate(action_id=AtomicActionId.LIFT, intent="lift",
                   target=object_target, axes=("z",),
                   description="Raise the closed gripper and held object by one bounded vertical step before transfer."),
        _candidate(action_id=AtomicActionId.TRACK_WAYPOINT, intent="track",
                   target=goal_target, axes=("x", "y", "z"),
                   description="Move one signed world axis step toward the current task target."),
        _candidate(action_id=AtomicActionId.GRIP_OPEN, intent="release",
                   target=gripper_target, axes=(), provenance="robot_proprioception",
                   description="Open the gripper for one bounded actuator step."),
        _candidate(action_id=AtomicActionId.GRIP_CLOSE, intent="engage",
                   target=gripper_target, axes=(), provenance="robot_proprioception",
                   description="Close the gripper for one bounded actuator step."),
        _candidate(action_id=AtomicActionId.HOLD, intent="observe",
                   target=hold_target, axes=(), provenance="robot_proprioception",
                   description="Keep the current pose and gripper setting for one step."),
    ]
    gripper_command = str(state.get("gripper_command", "")).lower()
    if gripper_command == "open":
        candidates = [item for item in candidates if item.id != AtomicActionId.GRIP_OPEN.value]
        if state.get("task_name") in _TRANSFER_TASKS | _CONTACT_TASKS:
            candidates = [item for item in candidates if item.id != AtomicActionId.TRACK_WAYPOINT.value]
            candidates = [item for item in candidates if item.id != AtomicActionId.LIFT.value]
            candidates = [item for item in candidates if item.id != AtomicActionId.HOLD.value]
            control = _finite_xyz(state.get("control_xyz"))
            target_xyz = _finite_xyz(state.get(object_key))
            # Closing far from the object is an irreversible one-step choice
            # that repeatedly caused empty-hand transfers.  Only expose the
            # close intent once the visible TCP-to-object distance is within
            # the bounded grasp approach radius.
            if control is None or target_xyz is None or math.dist(control, target_xyz) > 0.026:
                candidates = [item for item in candidates if item.id != AtomicActionId.GRIP_CLOSE.value]
            if control is not None and target_xyz is not None:
                horizontal_error = math.hypot(control[0] - target_xyz[0], control[1] - target_xyz[1])
                vertical_error = abs(control[2] - target_xyz[2])
                if horizontal_error > 0.018:
                    # Finish horizontal alignment before descending.
                    candidates = [item for item in candidates if item.id != AtomicActionId.ADJUST_Z.value]
                elif vertical_error > 0.020:
                    # Once horizontally aligned, descend/ascend to object height.
                    candidates = [item for item in candidates if item.id != AtomicActionId.ALIGN_XY.value]
                else:
                    # At the grasp point, the close choice is the only motion
                    # that advances the next stage.
                    candidates = [item for item in candidates if item.id not in {
                        AtomicActionId.ALIGN_XY.value, AtomicActionId.ADJUST_Z.value,
                    }]
    elif gripper_command == "closed":
        candidates = [item for item in candidates if item.id != AtomicActionId.GRIP_CLOSE.value]
        if state.get("task_name") in _CONTACT_TASKS:
            candidates = [item for item in candidates if item.id not in {
                AtomicActionId.ALIGN_XY.value, AtomicActionId.ADJUST_Z.value,
                AtomicActionId.LIFT.value, AtomicActionId.HOLD.value,
                AtomicActionId.GRIP_OPEN.value,
            }]
            return tuple(candidates)
        if state.get("task_name") in _TRANSFER_TASKS:
            control = _finite_xyz(state.get("control_xyz"))
            object_xyz = _finite_xyz(state.get(object_key))
            opening = float(state.get("gripper_opening", 0.0) or 0.0)
            # Give the actuator a few bounded steps to finish closing before
            # moving the arm. Then require a short vertical lift before a
            # horizontal destination track can be selected.
            if opening > 0.75:
                candidates = [item for item in candidates if item.id in {
                    AtomicActionId.HOLD.value,
                }]
            elif (control is not None and object_xyz is not None
                  and int(state.get("atomic_lift_steps", 0) or 0) < 18):
                candidates = [item for item in candidates if item.id in {
                    AtomicActionId.LIFT.value,
                }]
            elif int(state.get("atomic_lift_steps", 0) or 0) >= 18:
                candidates = [item for item in candidates if item.id not in {
                    AtomicActionId.LIFT.value, AtomicActionId.ALIGN_XY.value,
                    AtomicActionId.ADJUST_Z.value,
                }]
            goal_xyz = _finite_xyz(state.get(goal_key))
            if (opening <= 0.75 and control is not None
                    and int(state.get("atomic_lift_steps", 0) or 0) >= 18
                    and goal_xyz is not None and math.dist(control, goal_xyz) <= 0.03):
                candidates = [item for item in candidates if item.id not in {
                    AtomicActionId.TRACK_WAYPOINT.value, AtomicActionId.HOLD.value,
                }]
            elif opening <= 0.75 and int(state.get("atomic_lift_steps", 0) or 0) >= 18:
                candidates = [item for item in candidates if item.id != AtomicActionId.GRIP_OPEN.value]
    return tuple(candidates)


def _finite_xyz(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        xyz = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return xyz if all(math.isfinite(item) for item in xyz) else None


def _visible_target_xyz(state: Mapping[str, Any], candidate: AtomicActionCandidate) -> tuple[float, float, float] | None:
    """Resolve only references advertised by the candidate descriptor."""
    if "evaluation_state" in state:
        raise ValueError("atomic lowering cannot inspect evaluation_state")
    target = candidate.target
    source_key = target.get("source_key")
    if source_key == "active_waypoint.target_xyz":
        waypoint = state.get("active_waypoint")
        if isinstance(waypoint, Mapping):
            return _finite_xyz(waypoint.get("target_xyz"))
        return None
    if isinstance(source_key, str) and source_key:
        # Nested waypoint references are handled above; ordinary keys are
        # looked up literally, never through evaluation_state.
        return _finite_xyz(state.get(source_key))
    return None


def _direction_action(delta: tuple[float, float, float], axes: tuple[str, ...]) -> Action:
    indexes = {"x": 0, "y": 1, "z": 2}
    if not axes:
        return Action.HOLD
    valid = [(axis, abs(delta[indexes[axis]]), delta[indexes[axis]]) for axis in axes]
    axis, magnitude, signed = max(valid, key=lambda item: item[1])
    if magnitude <= 1e-9:
        return Action.HOLD
    if axis == "x":
        return Action.X_POS if signed > 0 else Action.X_NEG
    if axis == "y":
        return Action.Y_POS if signed > 0 else Action.Y_NEG
    return Action.Z_POS if signed > 0 else Action.Z_NEG


def _visual_direction_action(state: Mapping[str, Any], action_id: AtomicActionId) -> Action:
    directions = state.get("action_screen_directions")
    if not isinstance(directions, Mapping):
        raise AtomicActionUnavailable("visual direction requires calibrated action_screen_directions")
    desired = {
        AtomicActionId.VISUAL_LEFT: (-1.0, 0.0),
        AtomicActionId.VISUAL_RIGHT: (1.0, 0.0),
        AtomicActionId.VISUAL_UP: (0.0, -1.0),
        AtomicActionId.VISUAL_DOWN: (0.0, 1.0),
    }[action_id]
    best: tuple[float, str] | None = None
    for primitive, descriptor in directions.items():
        if primitive not in {item.value for item in Action if item in {
                Action.X_POS, Action.X_NEG, Action.Y_POS, Action.Y_NEG, Action.Z_POS, Action.Z_NEG}}:
            continue
        delta = descriptor.get("delta_px") if isinstance(descriptor, Mapping) else None
        if not isinstance(delta, (list, tuple)) or len(delta) != 2:
            continue
        try:
            dx, dy = float(delta[0]), float(delta[1])
        except (TypeError, ValueError):
            continue
        norm = math.hypot(dx, dy)
        if norm <= 1e-9 or not math.isfinite(norm):
            continue
        score = (desired[0] * dx + desired[1] * dy) / norm
        if best is None or score > best[0]:
            best = (score, primitive)
    # A partial calibration must fail closed.  Selecting the least-opposed
    # primitive would silently move in the opposite screen direction and
    # turn a missing calibration sample into an unsafe control command.
    if best is None or best[0] <= 0.0:
        raise AtomicActionUnavailable("visual direction has no calibrated finite primitive")
    return Action(best[1])


class AtomicActionUnavailable(RuntimeError):
    """Raised when a semantic candidate has no policy-visible target estimate."""


@dataclass(frozen=True)
class AtomicExecution:
    """The bounded lowering result consumed by an environment adapter."""

    candidate: AtomicActionCandidate
    action: Action
    steps: int = 1
    termination: str = "one_environment_step"
    provenance: str = "policy_visible_state"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.id,
            "intent": self.candidate.intent,
            "action": self.action.value,
            "steps": self.steps,
            "termination": self.termination,
            "provenance": self.provenance,
        }


class MetaWorldAtomicExecutor:
    """Lower one MetaWorld intent to exactly one legacy :class:`Action`."""

    def lower(self, candidate: AtomicActionCandidate,
              state: Mapping[str, Any]) -> AtomicExecution:
        if not isinstance(candidate, AtomicActionCandidate):
            candidate = AtomicActionCandidate.from_dict(candidate)
        if "evaluation_state" in state:
            raise ValueError("atomic lowering cannot inspect evaluation_state")
        action_id = candidate.action_id
        if action_id in {AtomicActionId.VISUAL_LEFT, AtomicActionId.VISUAL_RIGHT,
                         AtomicActionId.VISUAL_UP, AtomicActionId.VISUAL_DOWN}:
            action = _visual_direction_action(state, action_id)
        elif action_id == AtomicActionId.LIFT:
            action = Action.Z_POS
        elif action_id == AtomicActionId.GRIP_OPEN:
            action = Action.GRIP_OPEN
        elif action_id == AtomicActionId.GRIP_CLOSE:
            action = Action.GRIP_CLOSE
        elif action_id == AtomicActionId.HOLD:
            action = Action.HOLD
        else:
            target = _visible_target_xyz(state, candidate)
            control = _finite_xyz(state.get("control_xyz"))
            if target is None or control is None:
                raise AtomicActionUnavailable(
                    f"{candidate.id} requires a policy-visible target_xyz and control_xyz")
            delta = tuple(target[index] - control[index] for index in range(3))
            axes = tuple(str(axis) for axis in candidate.parameters.get("axes", ()))
            action = _direction_action(delta, axes)
        return AtomicExecution(candidate=candidate, action=action,
                               steps=1, termination="one_environment_step",
                               provenance=candidate.provenance)

    def lower_to_action(self, candidate: AtomicActionCandidate,
                        state: Mapping[str, Any]) -> Action:
        """Compatibility helper for runners that still accept only ``Action``."""
        return self.lower(candidate, state).action


# Generic names make the protocol discoverable without coupling callers to the
# MetaWorld implementation class, while the explicit class remains available.
AtomicExecutor = MetaWorldAtomicExecutor

def lower_atomic_action(candidate: AtomicActionCandidate,
                        state: Mapping[str, Any]) -> AtomicExecution:
    return MetaWorldAtomicExecutor().lower(candidate, state)


__all__ = [
    "ATOMIC_PROTOCOL", "METAWORLD_ATOMIC_VERSION", "AtomicActionId",
    "AtomicAction", "AtomicActionCandidate", "AtomicActionUnavailable",
    "AtomicExecution", "AtomicExecutor", "MetaWorldAtomicExecutor",
    "METAWORLD_ATOMIC_IDS", "metaworld_atomic_candidates",
    "lower_atomic_action",
]
