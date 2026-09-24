"""Optional, reproducible escape actions based only on policy-visible proprioception.

This module never receives evaluation_state, simulator object poses, waypoints,
or success verdicts. One escape primitive consumes one ordinary decision slot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Any

from .core import Action


_MOVES = (
    Action.X_POS, Action.X_NEG, Action.Y_POS, Action.Y_NEG,
    Action.Z_POS, Action.Z_NEG,
)
_OPPOSITE = {
    Action.X_POS: Action.X_NEG, Action.X_NEG: Action.X_POS,
    Action.Y_POS: Action.Y_NEG, Action.Y_NEG: Action.Y_POS,
    Action.Z_POS: Action.Z_NEG, Action.Z_NEG: Action.Z_POS,
}


def _tcp_xyz(state: dict[str, Any]) -> tuple[float, float, float] | None:
    """Validate just the one sensor field needed by the escape detector."""
    values = state.get("control_xyz")
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    try:
        point = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(value) for value in point) else None


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.dist(a, b)


@dataclass(frozen=True)
class RecoveryChoice:
    action: Action
    intervention: dict[str, Any] | None = None


@dataclass(frozen=True)
class _History:
    number: int
    tcp_xyz: tuple[float, float, float]
    proposed_action: Action
    executed_action: Action
    gripper_command: str | None


class StuckRecovery:
    """Insert at most two seeded translation primitives after observed stalls."""

    def __init__(
        self, *, seed: int, window: int = 6,
        displacement_threshold_m: float = 0.003,
        cooldown: int = 6, max_interventions: int = 3,
        perturb_steps: int = 2, cycle_path_limit_m: float = 0.08,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("recovery seed must be a nonnegative integer")
        if window < 4 or window % 2:
            raise ValueError("recovery window must be an even integer >= 4")
        if not math.isfinite(displacement_threshold_m) or displacement_threshold_m <= 0:
            raise ValueError("displacement_threshold_m must be positive and finite")
        if cooldown < 0 or max_interventions < 0:
            raise ValueError("cooldown and max_interventions must be nonnegative")
        if perturb_steps not in {1, 2}:
            raise ValueError("perturb_steps must be 1 or 2")
        if not math.isfinite(cycle_path_limit_m) or cycle_path_limit_m <= 0:
            raise ValueError("cycle_path_limit_m must be positive and finite")
        self.seed = seed
        self.window = window
        self.displacement_threshold_m = displacement_threshold_m
        self.cooldown = cooldown
        self.max_interventions = max_interventions
        self.perturb_steps = perturb_steps
        self.cycle_path_limit_m = cycle_path_limit_m
        self.reset()

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._history: deque[_History] = deque(maxlen=self.window)
        self._pending: deque[RecoveryChoice] = deque()
        self.intervention_count = 0
        self.intervention_steps = 0
        self._last_intervention_step = -math.inf

    def _trigger(self, point: tuple[float, float, float],
                 proposed: Action, number: int,
                 gripper_command: str | None) -> dict[str, Any] | None:
        if len(self._history) < self.window - 1:
            return None
        rows = list(self._history)[-(self.window - 1):]
        recent_contiguous = [row.number for row in rows] == list(range(number - self.window + 1, number))
        actions = [row.proposed_action for row in rows] + [proposed]
        positions = [row.tcp_xyz for row in rows] + [point]
        extent = max(_distance(positions[0], position) for position in positions)
        gripper_set = (
            proposed == Action.GRIP_CLOSE and gripper_command == "closed"
            and all(row.gripper_command == "closed" for row in rows)
        ) or (
            proposed == Action.GRIP_OPEN and gripper_command == "open"
            and all(row.gripper_command == "open" for row in rows)
        )
        eligible_repeat = proposed in _MOVES or proposed == Action.HOLD or gripper_set
        if (recent_contiguous and eligible_repeat
                and all(action == proposed for action in actions)
                and extent <= self.displacement_threshold_m):
            reason = ("repeated_translation_stall" if proposed in _MOVES
                      else "repeated_hold_stall" if proposed == Action.HOLD
                      else "repeated_gripper_stall")
            excluded = {proposed} if proposed in _MOVES else set()
        else:
            if len(self._history) < self.window:
                return None
            rows = list(self._history)[-self.window:]
            if [row.number for row in rows] != list(range(number - self.window, number)):
                return None
            completed_actions = [row.proposed_action for row in rows]
            positions = [row.tcp_xyz for row in rows] + [point]
            cycle_path = sum(_distance(a, b) for a, b in zip(positions, positions[1:]))
            if not (proposed in _MOVES and completed_actions[0] in _MOVES
                    and completed_actions[1] == _OPPOSITE[completed_actions[0]]
                    and all(action == completed_actions[index % 2]
                            for index, action in enumerate(completed_actions))
                    and proposed == completed_actions[0]
                    and _distance(positions[0], positions[-1]) <= self.displacement_threshold_m
                    and cycle_path <= self.cycle_path_limit_m):
                return None
            reason = "opposite_action_cycle"
            excluded = {completed_actions[0], completed_actions[1]}
            actions = completed_actions + [proposed]
            extent = max(_distance(positions[0], position) for position in positions)
        displacement = _distance(positions[0], positions[-1])
        path_length = sum(_distance(a, b) for a, b in zip(positions, positions[1:]))
        return {
            "reason": reason,
            "window": self.window,
            "window_steps": [row.number for row in rows] + [number],
            "tcp_positions_xyz": [list(position) for position in positions],
            "proposed_actions": [action.value for action in actions],
            "executed_actions": [row.executed_action.value for row in rows],
            "gripper_commands": [row.gripper_command for row in rows] + [gripper_command],
            "window_displacement_m": round(displacement, 6),
            "window_extent_m": round(extent, 6),
            "window_path_length_m": round(path_length, 6),
            "displacement_threshold_m": self.displacement_threshold_m,
            "cycle_path_limit_m": self.cycle_path_limit_m,
            "excluded_actions": sorted(action.value for action in excluded),
        }

    def select(self, state: dict[str, Any], proposed_action: Action,
               number: int) -> RecoveryChoice:
        """Choose an actual action after the policy has been queried this slot."""
        proposed = Action(proposed_action)
        if self._pending:
            return self._pending.popleft()
        if (self.intervention_count >= self.max_interventions
                or number <= self._last_intervention_step + self.cooldown):
            return RecoveryChoice(proposed)
        point = _tcp_xyz(state)
        if point is None:
            return RecoveryChoice(proposed)
        command = state.get("gripper_command")
        trigger = self._trigger(point, proposed, number,
                                command if isinstance(command, str) else None)
        if trigger is None:
            return RecoveryChoice(proposed)
        excluded = set(trigger["excluded_actions"])
        pool = [action for action in _MOVES if action.value not in excluded]
        intervention_id = self.intervention_count + 1
        previous: Action | None = None
        for index in range(1, self.perturb_steps + 1):
            eligible = ([action for action in pool if action != _OPPOSITE[previous]]
                        if previous is not None else pool)
            action = self._rng.choice(eligible)
            previous = action
            self._pending.append(RecoveryChoice(action, {
                "id": intervention_id, "step_index": index,
                "steps_planned": self.perturb_steps, "executed_action": action.value,
                "rng_seed": self.seed, "trigger": trigger,
            }))
        return self._pending.popleft()

    def record_step(self, number: int, state_before: dict[str, Any],
                    proposed_action: Action, choice: RecoveryChoice) -> None:
        """Commit only actions that the environment actually accepted."""
        if choice.intervention is not None:
            if choice.intervention["step_index"] == 1:
                self.intervention_count += 1
            self.intervention_steps += 1
            self._last_intervention_step = number
            self._history.clear()
            return
        point = _tcp_xyz(state_before)
        if point is None:
            self._history.clear()
            return
        command = state_before.get("gripper_command")
        self._history.append(_History(number, point, Action(proposed_action), choice.action,
                                      command if isinstance(command, str) else None))
