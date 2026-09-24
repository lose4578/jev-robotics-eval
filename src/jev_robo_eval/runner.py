"""Policy/environment loop with a finite decision budget."""

from __future__ import annotations

import json
import math
import time
from numbers import Real
from dataclasses import dataclass
from pathlib import Path

from .core import Observation, Policy, RobotEnvironment
from .recovery import StuckRecovery


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    decisions: int
    simulator_steps: int
    reward: float
    stopped_by: str
    intervention_count: int = 0
    intervention_steps: int = 0


def run_episode(
    environment: RobotEnvironment,
    policy: Policy,
    *,
    task: str,
    max_decisions: int = 120,
    seed: int | None = None,
    trace_path: Path | None = None,
    verbose: bool = True,
    recovery: StuckRecovery | None = None,
) -> EpisodeResult:
    if max_decisions < 1:
        raise ValueError("max_decisions must be >= 1")
    total_reward = 0.0
    simulator_steps = 0
    trace = None
    if recovery is not None:
        recovery.reset()
    try:
        observation = environment.reset(seed=seed)
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace = trace_path.open("w", encoding="utf-8")
        for number in range(1, max_decisions + 1):
            state_before = observation.evaluation_state or observation.state
            policy_state_before = observation.state
            started = time.monotonic()
            # Evaluation truth stays in the runner, outside the policy boundary.
            decision = policy.decide(Observation(state=observation.state, image=observation.image), task)
            decision_seconds = time.monotonic() - started
            recovery_state = {key: policy_state_before[key] for key in ("control_xyz", "gripper_command")
                              if key in policy_state_before}
            choice = (recovery.select(recovery_state, decision.action, number)
                      if recovery is not None else None)
            actual_action = choice.action if choice is not None else decision.action
            proposed_scale = decision.action_scale
            if (isinstance(proposed_scale, bool) or not isinstance(proposed_scale, Real)
                    or not math.isfinite(proposed_scale) or not 0 < proposed_scale <= 1):
                raise ValueError("Policy action_scale must be finite and in (0, 1]")
            # Recovery has its own execution budget; it must not inherit the
            # model's fine-motion scale when replacing an action.
            action_scale = 1.0 if choice is not None and choice.intervention else float(proposed_scale)
            transition = (environment.step(actual_action) if action_scale == 1.0 else
                          environment.step(actual_action, scale=action_scale))
            if recovery is not None:
                recovery.record_step(number, recovery_state, decision.action, choice)
            on_motion_executed = getattr(policy, "on_motion_executed", None)
            if callable(on_motion_executed):
                on_motion_executed(actual_action, action_scale)
            else:
                on_action_executed = getattr(policy, "on_action_executed", None)
                if callable(on_action_executed):
                    on_action_executed(actual_action)
            observation = transition.observation
            total_reward += transition.reward
            evaluation_state = observation.evaluation_state or observation.state
            simulator_steps = int(evaluation_state.get("simulator_steps", number))
            record = {
                "decision": number,
                "action": actual_action.value,
                "proposed_action": decision.action.value,
                "action_scale": action_scale,
                "proposed_action_scale": float(proposed_scale),
                "proposed_candidate": decision.selection.get("selected_candidate"),
                "intervention": choice.intervention if choice is not None else None,
                "decision_seconds": round(decision_seconds, 4),
                "probabilities": decision.probabilities,
                "request": decision.request,
                "usage": decision.usage,
                "image_audit": decision.image_audit,
                "selection": decision.selection,
                "reward": transition.reward,
                "success": transition.success,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "metrics": {key: float(value) for key, value in transition.info.items() if isinstance(value, Real)},
                "state_before": state_before,
                "policy_state_before": policy_state_before,
                "state": evaluation_state,
            }
            if trace is not None:
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
            if verbose:
                print(json.dumps({k: v for k, v in record.items() if k not in {"probabilities", "request", "state", "state_before", "policy_state_before", "metrics"}}, ensure_ascii=False), flush=True)
            if transition.success or transition.terminated or transition.truncated:
                reason = "success" if transition.success else "terminated" if transition.terminated else "truncated"
                return EpisodeResult(transition.success, number, simulator_steps, total_reward, reason,
                                     recovery.intervention_count if recovery is not None else 0,
                                     recovery.intervention_steps if recovery is not None else 0)
        return EpisodeResult(False, max_decisions, simulator_steps, total_reward, "decision_limit",
                             recovery.intervention_count if recovery is not None else 0,
                             recovery.intervention_steps if recovery is not None else 0)
    finally:
        if trace is not None:
            trace.close()
        environment.close()
