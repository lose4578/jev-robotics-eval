"""JEV Choice API adapters."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .atomic_actions import (
    ATOMIC_PROTOCOL,
    AtomicActionCandidate,
    AtomicActionUnavailable,
    MetaWorldAtomicExecutor,
    metaworld_atomic_candidates,
)
from .core import ACTION_DESCRIPTIONS, Action, Decision, Observation
from .hierarchy import HIERARCHY_PROTOCOL, PHASE_FIXED_PROTOCOL, MotionCandidate, eligible_phases, phase_action_candidates, schema_for_task
from .tasks import ROBOTWIN_TASK_GUIDES, TASK_GUIDES, task_goal
from .observation_access import filter_policy_state


MODEL = os.environ.get("JEV_MODEL", "jev")

SENSOR_PHASES = {
    "approach": "The gripper still needs to move above the visible task object.",
    "lower": "The open gripper is aligned over the object and needs to descend toward it.",
    "grasp": "The gripper surrounds the object; close it or wait for closure before lifting.",
    "lift": "The object appears held near the gripper; raise it clear of the table or container.",
    "transfer": "Carry the raised object toward the visible shelf, target bin, or peg.",
    "place": "The held object is over the destination; lower it and release or mount it.",
    "recover": "The object was missed or dropped, or the prior motion did not make progress; reacquire it.",
}

SENSOR_PHASE_TASK = {
    "shelf-place-v3": "During transfer, move the held block above the shelf opening; place it inside the shelf.",
    "bin-picking-v3": "During lift, clear the source bin rim; transfer over the separate target bin before lowering.",
    "assembly-v3": "During transfer, center the nut's hole over the upright peg; place by lowering the nut onto it.",
}

# A local temporal prior over *model-inferred* phases. No simulator object,
# goal, contact, or success fact enters this transition set.
SENSOR_PHASE_NEIGHBORS = {
    "approach": ("approach", "lower", "recover"),
    "lower": ("lower", "approach", "grasp", "recover"),
    "grasp": ("grasp", "lift", "approach", "recover"),
    "lift": ("lift", "transfer", "recover"),
    "transfer": ("transfer", "place", "recover"),
    "place": ("place", "recover"),
    "recover": ("recover", "approach", "lower"),
}

PRESS_PHASES = {
    "prepare": "The active gripper is open; close it to form a pressing tip.",
    "align": "Move horizontally above the visible button or stapler top before descending.",
    "press": "The closed gripper is aligned above the contact surface; move downward to press it.",
    "recover": "The last motion was blocked or missed the object; reposition before trying again.",
}
PRESS_PHASE_NEIGHBORS = {
    "prepare": ("prepare", "align", "recover"),
    "align": ("align", "press", "prepare", "recover"),
    "press": ("press", "align", "recover"),
    "recover": ("recover", "align", "prepare"),
}
PRESS_TASKS = {"click_bell", "press_stapler"}


def _eligible_sensor_phases(previous_phase: str | None, gripper_command: str,
                            task_name: str | None = None) -> tuple[dict[str, str], str]:
    pressing = task_name in PRESS_TASKS
    neighbors = PRESS_PHASE_NEIGHBORS if pressing else SENSOR_PHASE_NEIGHBORS
    descriptions = PRESS_PHASES if pressing else SENSOR_PHASES
    prior = previous_phase if previous_phase in neighbors else "prepare" if pressing else "approach"
    candidates = neighbors[prior]
    reason = (
        f"Episode start uses the {prior} neighborhood as a scheduling prior, not as proof that "
        "any task phase has been completed. " if previous_phase is None else
        f"The previous model-inferred phase was {prior}; consider only that phase, its adjacent "
        "phases, or recovery to avoid unsupported jumps. "
    )
    if str(gripper_command).lower() == "open":
        excluded = {"press"} if pressing else {"lift", "transfer", "place"}
        candidates = tuple(phase for phase in candidates if phase not in excluded)
        reason += (
            "The robot's own gripper command is OPEN, so lifting, transferring, and placing a "
            "held object are ineligible. A CLOSED command would not prove a successful grasp."
        )
        if pressing:
            reason = reason.split("The robot's own gripper command")[0] + (
                "The robot's own gripper command is OPEN, so pressing with closed fingers is ineligible."
            )
    return {phase: descriptions[phase] for phase in candidates}, reason


def _sensor_state(state: dict, task_name: str) -> dict:
    """Allow only robot proprioception and fixed observation metadata to reach JEV."""
    if "active_waypoint" in state:
        raise ValueError("Nonprivileged policy cannot receive an oracle waypoint")
    if "WAYPOINT" in str(state.get("image_annotations", "")).upper():
        raise ValueError("Nonprivileged policy cannot receive waypoint image annotations")
    required = ("control_xyz", "gripper_opening", "gripper_command")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"Nonprivileged observation lacks robot proprioception: {missing}")
    result = {"information": "nonprivileged", "task_name": task_name}
    for name in ("coordinate_frame", "control_point", "control_xyz", "gripper_opening",
                 "gripper_command", "simulator_steps", "camera_name",
                 "action_screen_directions", "environment", "active_arm", "control_steps", "tcp_pixel"):
        if name in state:
            result[name] = state[name]
    robot = state.get("robot")
    if isinstance(robot, dict):
        safe_robot = {key: robot[key] for key in (
            "hand_body_xyz", "hand_quat_wxyz", "arm_joint_order",
            "arm_joint_position_rad", "arm_joint_velocity_rad_s") if key in robot}
        if safe_robot:
            result["robot"] = safe_robot
    return result


class JevPolicy:
    def __init__(
        self,
        *,
        mode: str = "text",
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        busy_retries: int = 3,
        sensor_policy: str = "direct",
        vision_transport: str = "auto",
        action_selection: str = "argmax",
        policy_seed: int = 0,
        sampling_temperature: float = 1.0,
        action_space: str = "primitive",
        action_granularity: str = "fixed",
    ) -> None:
        if mode not in {"text", "vision"}:
            raise ValueError("mode must be text or vision")
        if sensor_policy not in {"direct", "staged"}:
            raise ValueError("sensor_policy must be direct or staged")
        if vision_transport not in {"auto", "bridge", "native"}:
            raise ValueError("vision_transport must be auto, bridge, or native")
        if action_selection not in {"argmax", "sample"}:
            raise ValueError("action_selection must be argmax or sample")
        if not isinstance(policy_seed, int) or policy_seed < 0:
            raise ValueError("policy_seed must be a nonnegative integer")
        if not math.isfinite(sampling_temperature) or sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be finite and > 0")
        aliases = {"primitive": "primitive", "legacy": "primitive",
                   "metaworld_atomic": "metaworld_atomic", "atomic": "metaworld_atomic"}
        if action_space not in aliases:
            raise ValueError("action_space must be primitive or metaworld_atomic")
        if action_granularity not in {"fixed", "adaptive", "phase-fixed"}:
            raise ValueError("action_granularity must be fixed, adaptive, or phase-fixed")
        if action_granularity != "fixed" and (
                mode != "vision" or sensor_policy != "staged" or aliases[action_space] != "primitive"):
            raise ValueError(f"{action_granularity} action granularity requires vision mode, staged sensor policy, and primitive actions")
        self.action_granularity = action_granularity
        self.action_selection = action_selection
        self.policy_seed = policy_seed
        self.sampling_temperature = sampling_temperature
        self.action_space = aliases[action_space]
        self._action_rng = random.Random(policy_seed)
        self.mode = mode
        self.sensor_policy = sensor_policy
        self.url = url or ("http://127.0.0.1:8080/v1/decision" if mode == "text" else "http://127.0.0.1:8081/v1/vision-decision")
        if mode == "vision" and vision_transport == "auto":
            path = urlparse(self.url).path.rstrip("/")
            if path == "/v1/decision":
                vision_transport = "native"
            elif path == "/v1/vision-decision":
                vision_transport = "bridge"
            else:
                raise ValueError("Cannot infer vision transport from URL path; choose bridge or native")
        self.vision_transport = vision_transport
        self.api_key = api_key
        self.timeout = timeout
        self.busy_retries = busy_retries
        self._sensor_last_action: str | None = None
        self._sensor_last_phase: str | None = None
        self._sensor_last_tcp: list[float] | None = None
        self._sensor_last_task: str | None = None
        self._sensor_last_step: int | None = None
        self._atomic_lift_steps = 0
        self._atomic_executor = MetaWorldAtomicExecutor()
        self._adaptive_last_candidate: str | None = None
        self._adaptive_last_scale: float | None = None
        self._adaptive_action_repeats = 0
        self._adaptive_phase_steps = 0
        self._adaptive_stall_steps = 0
        self._adaptive_axis_reversals = 0

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = Request(self.url, body, headers=headers, method="POST")
        for attempt in range(self.busy_retries + 1):
            try:
                with urlopen(req, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                detail = exc.read(2048).decode("utf-8", errors="replace")
                if exc.code in (429, 529) and attempt < self.busy_retries:
                    time.sleep(min(float(exc.headers.get("Retry-After", "1")), 5.0))
                    continue
                raise RuntimeError(f"Jev HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                raise RuntimeError(f"Cannot reach Jev at {self.url}: {exc.reason}") from exc
        raise RuntimeError("Jev remained busy")

    def decide(self, observation: Observation, task: str) -> Decision:
        if self.action_granularity != "fixed":
            return self._decide_adaptive(observation, task)
        level = observation.state.get("privilege_level")
        if observation.state.get("plan_only"):
            observation = Observation(state=filter_policy_state(observation.state, level, plan_only=True),
                                      image=observation.image)
        if level in {1, 2}:
            if "active_waypoint" in observation.state:
                raise ValueError("Oracle waypoints require privilege level 3")
            observation = Observation(state=filter_policy_state(observation.state, level), image=observation.image)
        if self.action_space == "metaworld_atomic":
            return self._decide_atomic(observation, task)
        if (observation.state.get("information") == "nonprivileged"
                or (observation.state.get("environment") == "robotwin" and level in {0, 1, 2})
                or (self.sensor_policy == "staged" and "active_waypoint" not in observation.state)):
            return self._decide_visual(observation, task)
        instructions = (
            "Choose exactly one safe discrete action that makes progress on the task. "
            "XYZ coordinates are world coordinates. Movement choices change only one axis; "
            "the gripper setting persists until changed. Use the current positions and goal."
        )
        if observation.state.get("task_name") == "reach-v3":
            instructions = (
                "Move the gripper center control_xyz to goal_xyz. "
                "Read goal_minus_control_xyz = goal_xyz - control_xyz. "
                "A positive error needs the positive action on that axis; a negative error needs "
                "the negative action. Choose the axis with the largest absolute error, and never "
                "move in the opposite direction. Keep the gripper open."
            )
        elif observation.state.get("task_name") in TASK_GUIDES:
            instructions = TASK_GUIDES[observation.state["task_name"]] + (
                " Use the relative direction descriptions: up is z_pos, down is z_neg, "
                "right is x_pos, left is x_neg, forward is y_pos, backward is y_neg."
            )
        model_state = {"task": task, **observation.state}
        if "active_waypoint" in observation.state:
            waypoint = observation.state["active_waypoint"]
            instructions = (
                "Move the gripper center control_xyz to goal_xyz. "
                "Read goal_minus_control_xyz = goal_xyz - control_xyz. "
                "A positive error needs the positive action on that axis; a negative error needs "
                "the negative action. Choose the axis with the largest absolute error. "
                "Preserve the current gripper setting during this movement."
            )
            immediate_task = waypoint.get("instruction", "Move the gripper center to the target point.")
            if observation.state.get("environment") == "robotwin":
                immediate_task = "Move the gripper center to the target point while preserving the gripper setting."
            if waypoint["gripper_change_needed"]:
                operation = "CLOSE" if waypoint["desired_gripper"] == "closed" else "OPEN"
                immediate_task = f"The gripper is {observation.state['gripper_command'].upper()}. The next operation is to {operation} the gripper."
                instructions = f"Choose the action that {operation.lower()}s the gripper."
            model_state = {
                "task": immediate_task,
                "control_xyz": observation.state["control_xyz"],
                "goal_xyz": waypoint["target_xyz"],
                "goal_minus_control_xyz": waypoint["delta_xyz"],
                "target_relative_to_gripper": waypoint["relative_direction"],
                "gripper_command": observation.state["gripper_command"],
                "gripper_opening": observation.state["gripper_opening"],
                **({"scene": observation.state["scene"]} if not observation.state.get("plan_only") else {}),
                "robot": observation.state["robot"],
            }
            if "image_annotations" in observation.state:
                model_state["image_annotations"] = observation.state["image_annotations"]
        request = {
            "model": MODEL,
            "state": model_state,
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": {action.value: description for action, description in ACTION_DESCRIPTIONS.items()},
                }
            },
        }
        image_base64, image_audit = self._encode_image(observation) if self.mode == "vision" else (None, {})
        if image_audit:
            image_audit["transport"] = self.vision_transport
        payload = self._vision_payload(request, image_base64) if image_base64 is not None else request
        result = self._post(payload)
        return self._decision_from_result(result, request, image_audit=image_audit)

    def _decide_atomic(self, observation: Observation, task: str) -> Decision:
        """Ask JEV to choose one bounded MetaWorld intent.

        The model still receives the same policy-visible state budget as the
        legacy path.  Only the finite Choice vocabulary changes; the selected
        intent is lowered to one legacy ``Action`` for the existing runner.
        """
        state = observation.state
        level = state.get("privilege_level")
        if state.get("information") == "nonprivileged" or level == 0:
            task_text = task_goal(state.get("task_name", "unknown"), information="nonprivileged")
            visible_state = _sensor_state(state, state.get("task_name", "unknown"))
        elif level in {1, 2, 3}:
            task_text = task
            visible_state = filter_policy_state(state, level,
                                                plan_only=bool(state.get("plan_only", False)))
        else:
            # Keep compatibility with hand-written observations that predate
            # privilege_level.  Such callers already own the state boundary.
            task_text = task
            visible_state = dict(state)
        model_state = {"task": task_text, **visible_state}
        if self._sensor_last_action is not None:
            model_state["previous_action"] = self._sensor_last_action

        # ``atomic_lift_steps`` is local controller history, never simulator
        # truth. It lets the bounded lift intent hand control back to JEV
        # after a fixed physical budget instead of repeating Z+ forever.
        candidate_state = dict(model_state)
        candidate_state["atomic_lift_steps"] = self._atomic_lift_steps
        candidates = metaworld_atomic_candidates(candidate_state)
        descriptions = {
            candidate.id: str(candidate.parameters.get("description", candidate.intent))
            for candidate in candidates
        }
        task_name = str(state.get("task_name", ""))
        gripper_command = str(visible_state.get("gripper_command", "")).lower()
        if task_name in {"pick-place-v3", "peg-insert-side-v3", "shelf-place-v3",
                         "bin-picking-v3", "assembly-v3"}:
            if gripper_command == "open":
                phase_instructions = (
                    "You are in the approach/lower part of the task. The gripper command is OPEN. "
                    "Use align_xy to reduce horizontal error to the visible object, or adjust_z to "
                    "reach its height. Choose grip_close only once the gripper is aligned around the "
                    "object. Do not select track_waypoint while OPEN: it is for carrying a held object."
                )
            else:
                phase_instructions = (
                    "The gripper command is CLOSED. You are in the grasp/lift/transfer part of the "
                    "task. Use lift to raise it clear before using track_waypoint to carry the object toward goal_xyz, preserving the "
                    "closed setting. Select grip_open only after the object is over the destination. "
                    "Use align_xy or adjust_z for final alignment when needed."
                )
        elif task_name in {"push-v3", "door-open-v3", "drawer-open-v3"}:
            if gripper_command == "open":
                phase_instructions = (
                    "This is a contact task. Approach the visible object or handle while OPEN, "
                    "then select grip_close only at the contact point. Do not track the final goal "
                    "with an empty gripper. After CLOSED, track_waypoint performs the bounded contact motion."
                )
            else:
                phase_instructions = (
                    "The gripper is CLOSED at the contact point. Select track_waypoint for the next "
                    "bounded motion toward the task goal and preserve contact."
                )
        else:
            phase_instructions = (
                "Use align_xy and adjust_z to approach the visible target, track_waypoint to move "
                "toward goal_xyz, and the gripper candidates only when their setting must change."
            )
        request = {
            "model": MODEL,
            "state": model_state,
            "candidate_protocol": ATOMIC_PROTOCOL,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "questions": {"action": {
                "type": "choice",
                "instructions": (
                    "Choose exactly one bounded MetaWorld intent ID from the candidate list. "
                    "Do not output coordinates, numbers, or a legacy action name. "
                    "The executor performs at most one existing environment.step and returns "
                    "control to JEV after that step. " + phase_instructions
                ),
                "criteria": descriptions,
            }},
        }
        image_base64, image_audit = self._encode_image(observation) if self.mode == "vision" else (None, {})
        if image_audit:
            image_audit["transport"] = self.vision_transport
        payload = self._vision_payload(request, image_base64) if image_base64 is not None else request
        result = self._post(payload)
        decision = self._atomic_decision_from_result(result, request, candidates,
                                                     model_state, image_audit=image_audit)
        selected_candidate = decision.selection.get("selected_candidate")
        if selected_candidate == "lift":
            self._atomic_lift_steps += 1
        elif selected_candidate in {"grip_close", "grip_open", "align_xy", "adjust_z"}:
            self._atomic_lift_steps = 0
        self._sensor_last_action = decision.action.value
        return decision

    def _atomic_decision_from_result(
        self, result: dict, request: dict,
        candidates: tuple[AtomicActionCandidate, ...],
        state: dict,
        *, image_audit: dict | None = None,
    ) -> Decision:
        """Parse a finite candidate choice and lower it to one legacy action."""
        by_id = {candidate.id: candidate for candidate in candidates}
        try:
            answer = result["answers"]["action"]
            selected_id = str(answer["choice"])
            candidate = by_id[selected_id]
            probabilities = {str(key): float(value)
                             for key, value in answer.get("probabilities", {}).items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid Jev atomic choice response: {result!r}") from exc

        selection = {
            "method": self.action_selection,
            "action_space": self.action_space,
            "candidate_protocol": ATOMIC_PROTOCOL,
            "service_choice": selected_id,
            "selected_candidate": selected_id,
            "policy_seed": self.policy_seed,
            "sampling_temperature": self.sampling_temperature,
        }
        if self.action_selection == "sample":
            keys = [candidate.id for candidate in candidates]
            if (set(probabilities) != set(keys)
                    or any(not math.isfinite(value) or value < 0 for value in probabilities.values())
                    or not any(value > 0 for value in probabilities.values())):
                raise RuntimeError(
                    "Sampling requires finite nonnegative probabilities for all atomic candidates and positive mass")
            log_mass = [math.log(probabilities[key]) if probabilities[key] > 0 else -math.inf
                        for key in keys]
            peak = max(log_mass)
            weights = [math.exp((value - peak) / self.sampling_temperature) for value in log_mass]
            total = math.fsum(weights)
            effective = [weight / total for weight in weights]
            selected_id = self._action_rng.choices(keys, weights=effective, k=1)[0]
            candidate = by_id[selected_id]
            selection["effective_probabilities"] = dict(zip(keys, effective))
            selection["selected_candidate"] = selected_id

        try:
            execution = self._atomic_executor.lower(candidate, state)
            action = execution.action
            selection["execution"] = execution.to_dict()
        except AtomicActionUnavailable as exc:
            # A visual L0 state can select a target-dependent intent before a
            # visual estimator has produced a 3-D target.  Preserve one-step
            # runner compatibility with an explicit, auditable HOLD fallback.
            action = Action.HOLD
            selection["execution"] = {
                "candidate_id": candidate.id, "intent": candidate.intent,
                "action": action.value, "steps": 1,
                "termination": "unavailable_target_fallback",
                "provenance": candidate.provenance,
                "error": str(exc),
            }
        selection["selected_action"] = action.value
        return Decision(action=action, probabilities=probabilities, request=request,
                        usage=self._response_usage(result), image_audit=image_audit or {},
                        selection=selection)

    def on_action_executed(self, action: Action) -> None:
        """Use actual actuator history when a recovery controller overrides JEV."""
        if self.action_granularity != "fixed" and action.value != self._sensor_last_action:
            self._adaptive_last_candidate = None
            self._adaptive_last_scale = None
            self._adaptive_action_repeats = 1
            self._adaptive_axis_reversals = 0
        self._sensor_last_action = action.value

    def _decide_adaptive(self, observation: Observation, task: str) -> Decision:
        state = observation.state
        level = state.get("privilege_level", 0 if state.get("information") == "nonprivileged" else None)
        if level not in {0, 1, 2} or state.get("plan_only") or "active_waypoint" in state:
            raise ValueError(f"{self.action_granularity} action granularity requires privilege level 0, 1, or 2 without oracle waypoints")
        task_name = str(state.get("task_name", "unknown"))
        schema = schema_for_task(task_name)
        # Apply the observation allowlist before constructing either model request.
        # _sensor_state also validates required proprioception and image annotations.
        sensors = _sensor_state(state, task_name)
        visible_state = sensors if level == 0 else filter_policy_state(state, level)
        task_text = (task if task and not any(name in task.lower() for name in ("goal_xyz", "object_xyz"))
                     else task_goal(task_name, information="nonprivileged"))
        model_state = {**visible_state, "task": task_text, "privilege_level": level,
                       "task_family": schema.family, "task_semantics": schema.description}
        step = int(state.get("control_steps", state.get("simulator_steps", 0)))
        if (self._sensor_last_task != task_name or step == 0
                or (self._sensor_last_step is not None and step <= self._sensor_last_step)):
            self._sensor_last_action = None
            self._sensor_last_phase = None
            self._sensor_last_tcp = None
            self._adaptive_last_candidate = None
            self._adaptive_last_scale = None
            self._adaptive_action_repeats = 0
            self._adaptive_phase_steps = 0
            self._adaptive_stall_steps = 0
            self._adaptive_axis_reversals = 0
        if self._sensor_last_action is not None:
            model_state.update(previous_action=self._sensor_last_action,
                               previous_candidate=self._adaptive_last_candidate,
                               previous_action_scale=self._adaptive_last_scale,
                               consecutive_same_action=self._adaptive_action_repeats,
                               consecutive_axis_reversals=self._adaptive_axis_reversals)
        if self._sensor_last_phase is not None:
            model_state.update(previous_inferred_phase=self._sensor_last_phase,
                               decisions_in_previous_phase=self._adaptive_phase_steps)
        if self._sensor_last_tcp is not None:
            motion = [float(now) - float(before)
                      for now, before in zip(model_state["control_xyz"], self._sensor_last_tcp)]
            model_state["last_tcp_motion_xyz"] = [round(value, 6) for value in motion]
            last_was_motion = self._sensor_last_action in {"x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg"}
            stalled = last_was_motion and math.sqrt(sum(value * value for value in motion)) < 0.0001
            self._adaptive_stall_steps = self._adaptive_stall_steps + 1 if stalled else 0
            model_state["motion_history"] = {
                "previous_movement_below_0_1mm": stalled,
                "consecutive_stalled_movements": self._adaptive_stall_steps,
                "evidence": "Robot TCP displacement only; this does not measure object or task progress.",
            }
        gripper_command = str(model_state["gripper_command"]).lower()
        adaptive = self.action_granularity == "adaptive"
        phase_choices, eligibility_reason = eligible_phases(schema, self._sensor_last_phase, gripper_command)
        image_base64, image_audit = self._encode_image(observation)
        image_audit["transport"] = self.vision_transport
        phase_request = {
            "model": MODEL, "state": model_state,
            "questions": {"phase": {
                "type": "choice", "criteria": phase_choices,
                "instructions": (
                    "Infer the current operation from the RGB image and permitted observation fields. "
                    + schema.description + " " + eligibility_reason + " "
                    "The previous inferred phase is a hypothesis. A gripper command alone cannot prove a grasp or contact. "
                    "Repeated actions or low TCP motion are evidence to reassess "
                    + ("alignment, scale, or recovery; " if adaptive else "alignment or recovery; ") +
                    "contact can legitimately constrain motion. Use object pose or contact facts only if explicitly provided. "
                    "Choose the contact operation only when current alignment and contact support it. "
                    "Recovery is temporary: return to approach or contact when their visual conditions apply. "
                    "A gripper setting change need not move the TCP and does not itself justify recovery. "
                    + SENSOR_PHASE_TASK.get(task_name, "")
                ),
            }},
        }
        phase_result = self._post(self._vision_payload(phase_request, image_base64))
        try:
            answer = phase_result["answers"]["phase"]
            phase = answer["choice"]
            if phase not in phase_choices:
                raise ValueError(f"Ineligible phase: {phase}")
            probabilities = {str(key): float(value) for key, value in answer.get("probabilities", {}).items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid Jev phase response: {phase_result!r}") from exc
        phase_evidence = {"request": phase_request, "choice": phase, "probabilities": probabilities,
                          "usage": self._response_usage(phase_result), "provenance": "model_inference"}
        candidates = phase_action_candidates(schema, phase, gripper_command, self.action_granularity)
        amplitude_instructions = (
            "Each candidate chooses a primitive AND its movement amplitude. "
            "Fine = 0.25, normal = 0.5, coarse = 1.0 times the configured movement amplitude, with the same duration. "
            "Use coarse for clear travel, fine near contact or the target, and normal for intermediate corrections. "
            if adaptive else
            "Each candidate chooses a primitive. Every motion uses a fixed 1.0 times the configured movement "
            "amplitude and the same duration; movement amplitude is not a model choice. "
        )
        instructions = (
            "Choose exactly one candidate ID. " + amplitude_instructions
            + f"Current inferred phase {phase}: {schema.phases[phase]} "
            "Every motion changes one signed world axis. Decide the useful sign from the image and permitted state; "
            "candidate order is not a recommendation. Gripper settings persist. Hold only to settle or if complete. "
            "If repeated movement stalls or overshoots, reassess "
            + ("direction and scale" if adaptive else "direction") + " from current evidence. "
            "Axis reversal history can indicate oscillation, but does not prove an overshoot. Recheck the "
            "current target direction before reversing. "
            + ("Use fine corrections near alignment. " if adaptive else "") +
            "action_screen_directions, when provided, maps world actions to image motion of the robot TCP. "
            "tcp_pixel, when provided, is the calibrated robot TCP location; u grows right and v down. "
            + SENSOR_PHASE_TASK.get(task_name, "")
        )
        if level >= 1:
            instructions += (
                " Use the permitted object, goal, and scene coordinates together with the image to identify the "
                "current contact or placement target. Compare that target with control_xyz to select the signed axis: "
                "target minus control positive means the positive action, and negative means the negative action. "
                "The previous action is history, not a direction instruction."
            )
        action_request = {
            "model": MODEL, "state": {**model_state, "inferred_phase": phase},
            "candidate_protocol": HIERARCHY_PROTOCOL if adaptive else PHASE_FIXED_PROTOCOL,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "questions": {"action": {"type": "choice", "instructions": instructions,
                                     "criteria": {candidate.id: candidate.description for candidate in candidates}}},
        }
        result = self._post(self._vision_payload(action_request, image_base64))
        decision = self._adaptive_decision_from_result(
            result, {**action_request, "phase_decision": phase_evidence}, candidates, image_audit=image_audit)
        self._adaptive_action_repeats = (self._adaptive_action_repeats + 1
                                         if self._sensor_last_action == decision.action.value else 1)
        opposite = {"x_pos": "x_neg", "x_neg": "x_pos", "y_pos": "y_neg", "y_neg": "y_pos",
                    "z_pos": "z_neg", "z_neg": "z_pos"}
        self._adaptive_axis_reversals = (self._adaptive_axis_reversals + 1
                                         if opposite.get(self._sensor_last_action) == decision.action.value else 0)
        self._adaptive_phase_steps = self._adaptive_phase_steps + 1 if self._sensor_last_phase == phase else 1
        self._sensor_last_action = decision.action.value
        self._sensor_last_phase = phase
        self._sensor_last_tcp = list(model_state["control_xyz"])
        self._sensor_last_task = task_name
        self._sensor_last_step = step
        self._adaptive_last_candidate = decision.selection["selected_candidate"]
        self._adaptive_last_scale = decision.action_scale
        return decision

    def _adaptive_decision_from_result(
        self, result: dict, request: dict, candidates: tuple[MotionCandidate, ...],
        *, image_audit: dict | None = None,
    ) -> Decision:
        by_id = {candidate.id: candidate for candidate in candidates}
        try:
            answer = result["answers"]["action"]
            selected_id = str(answer["choice"])
            candidate = by_id[selected_id]
            probabilities = {str(key): float(value) for key, value in answer.get("probabilities", {}).items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid Jev {self.action_granularity} choice response: {result!r}") from exc
        selection = {
            "method": self.action_selection, "action_space": "primitive", "action_granularity": self.action_granularity,
            "candidate_protocol": request["candidate_protocol"], "service_choice": selected_id,
            "policy_seed": self.policy_seed, "sampling_temperature": self.sampling_temperature,
            "phase_selection": "argmax", "inferred_phase": request["phase_decision"]["choice"],
        }
        if self.action_selection == "sample":
            keys = list(by_id)
            if (set(probabilities) != set(keys)
                    or any(not math.isfinite(value) or value < 0 for value in probabilities.values())
                    or not any(value > 0 for value in probabilities.values())):
                raise RuntimeError(f"Sampling requires finite nonnegative probabilities for every {self.action_granularity} candidate and positive mass")
            log_mass = [math.log(probabilities[key]) if probabilities[key] > 0 else -math.inf for key in keys]
            peak = max(log_mass)
            weights = [math.exp((value - peak) / self.sampling_temperature) for value in log_mass]
            total = math.fsum(weights)
            effective = [weight / total for weight in weights]
            selected_id = self._action_rng.choices(keys, weights=effective, k=1)[0]
            candidate = by_id[selected_id]
            selection["effective_probabilities"] = dict(zip(keys, effective))
        selection.update(selected_candidate=selected_id, selected_action=candidate.action.value,
                         action_scale=candidate.scale)
        return Decision(action=candidate.action, action_scale=candidate.scale, probabilities=probabilities,
                        request=request, usage=self._response_usage(result), image_audit=image_audit or {},
                        selection=selection)

    def _decide_visual(self, observation: Observation, task: str) -> Decision:
        if self.mode != "vision":
            raise ValueError("Nonprivileged policy requires a camera image and vision mode")
        image_base64, image_audit = self._encode_image(observation)
        image_audit["transport"] = self.vision_transport
        task_name = observation.state.get("task_name", "unknown")
        task_text = (task if task and not any(name in task.lower() for name in
                                             ("goal_xyz", "object_xyz"))
                     else task_goal(task_name, information="nonprivileged"))
        level = int(observation.state.get("privilege_level", 0))
        visible_state = (_sensor_state(observation.state, task_name) if level == 0
                         else filter_policy_state(observation.state, level))
        model_state = {"task": task_text, **visible_state}
        step = int(observation.state.get("control_steps", observation.state.get("simulator_steps", 0)))
        if (self._sensor_last_task != task_name or step == 0
                or (self._sensor_last_step is not None and step <= self._sensor_last_step)):
            self._sensor_last_action = None
            self._sensor_last_phase = None
            self._sensor_last_tcp = None
        if self._sensor_last_action is not None:
            model_state["previous_action"] = self._sensor_last_action
        if self._sensor_last_phase is not None:
            model_state["previous_inferred_phase"] = self._sensor_last_phase
        if self._sensor_last_tcp is not None:
            model_state["last_tcp_motion_xyz"] = [
                round(float(now) - float(before), 4)
                for now, before in zip(model_state["control_xyz"], self._sensor_last_tcp)
            ]

        phase_evidence = None
        if self.sensor_policy == "staged":
            eligible_phases, eligibility_reason = _eligible_sensor_phases(
                self._sensor_last_phase, str(model_state["gripper_command"]), task_name)
            phase_request = {
                "model": MODEL,
                "state": model_state,
                "questions": {"phase": {
                    "type": "choice",
                    "instructions": (
                        "Infer the CURRENT task phase from this RGB image, the provided observation, "
                        "and the robot's own previous decision. Use object coordinates or contact facts "
                        "only when they are explicitly present in the observation. The previous phase is a hypothesis, "
                        "not ground truth. " + eligibility_reason + " "
                        "Choose among the eligible phases from current visual evidence. "
                        "A closed gripper alone does not prove the object is held. "
                        "If an attempted grasp missed or a carried object was dropped, choose recover. "
                        + SENSOR_PHASE_TASK.get(task_name, "")
                    ),
                    "criteria": eligible_phases,
                }},
            }
            phase_result = self._post(self._vision_payload(phase_request, image_base64))
            try:
                phase_answer = phase_result["answers"]["phase"]
                phase = phase_answer["choice"]
                if phase not in eligible_phases:
                    raise ValueError(f"Ineligible phase: {phase}")
                phase_probabilities = {str(k): float(v) for k, v in phase_answer["probabilities"].items()}
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid Jev phase response: {phase_result!r}") from exc
            phase_evidence = {"request": phase_request, "choice": phase,
                              "probabilities": phase_probabilities,
                              "usage": self._response_usage(phase_result)}
            model_state = {**model_state, "inferred_phase": phase}
            descriptions = PRESS_PHASES if task_name in PRESS_TASKS else SENSOR_PHASES
            instructions = (
                "Choose exactly one robot primitive to make progress in the inferred phase. "
                f"Phase {phase}: {descriptions[phase]} "
                + SENSOR_PHASE_TASK.get(task_name, "") + " "
                "Use the image to locate the object and destination. Verify visually that an object "
                "moves with the gripper before carrying it; if a grasp failed, reacquire it. "
                "Movement names are world axes, not screen directions. Each move changes only one "
                "axis; the gripper command persists until changed, so do not repeat an already set "
                "gripper command. If the prior MOVEMENT action made no TCP progress, choose a "
                "different useful action."
            )
            if level >= 1:
                instructions += (
                    " Use the provided object and goal coordinates: approach/lower targets the object, "
                    "lift raises the held object, and transfer/place targets the destination. "
                    "Compare the relevant target with control_xyz; move along the needed signed world axis."
                )
            if task_name in PRESS_TASKS:
                model_state["task"] = f"Current operation: {phase}. {descriptions[phase]}"
                instructions = (
                    "Choose one of the nine primitives for the current model-inferred operation. "
                    f"{descriptions[phase]} Use the camera and robot state; a phase is a hypothesis. "
                    "The robot is pressing an object. Keep the closed gripper during alignment and pressing. "
                    "World +Z is upward; preserve height while aligning horizontally. "
                    "Each move changes only one world axis. Gripper commands persist."
                )
                if level >= 1:
                    instructions += (
                        " During align, compare goal_xyz X/Y with control_xyz X/Y. "
                        "A positive difference calls for the positive axis and a negative difference "
                        "for the negative axis. During press, move downward once aligned."
                    )
        else:
            instructions = (
                "Choose exactly one of the nine primitive actions using the current RGB image "
                "and the robot's own state. The image is an upright corner2 camera view. "
                "Movement names refer to world X, Y, Z, not screen left or right; each move "
                "changes only one axis, and the gripper command persists until changed; do not "
                "repeat an already set gripper command. "
                "Find the object and destination in the image. Approach with an open gripper, "
                "align before descending, close only when the fingers surround the object, "
                "then lift before carrying. Confirm visually that the object follows the gripper; "
                "a close command alone does not prove a grasp. Use hold only when waiting for "
                "the gripper to settle or when the task appears complete."
            )
            if task_name in ROBOTWIN_TASK_GUIDES:
                instructions = (
                    ROBOTWIN_TASK_GUIDES[task_name] + " Choose exactly one of the nine primitive actions. "
                    "Locate the object and destination in the upright RGB camera image. "
                    "Movement names refer to world X, Y and Z, not image coordinates. "
                    "Only the active arm moves; its orientation stays fixed. Each move changes one axis. "
                    "The gripper command persists; avoid repeating an already set open or close command."
                )
                if level >= 1:
                    instructions += (
                        " Use the permitted object and goal coordinates. For pressing, first align "
                        "control_xyz X/Y with goal_xyz X/Y, then descend; object_xyz is the body origin. "
                        "A positive target-minus-control error needs a positive world-axis move; "
                        "a negative error needs a negative move."
                    )
        if "previous_action" in model_state:
            instructions += (
                " previous_action and last_tcp_motion_xyz describe the robot's own previous command "
                "and measured displacement. If a movement produced nearly zero displacement, try "
                "a different useful axis instead of repeating the blocked movement."
            )
        if "action_screen_directions" in model_state:
            instructions += (
                " action_screen_directions gives the calibrated image displacement of the "
                "gripper TCP for each world-axis movement. Compare these directions with the "
                "visible object's position relative to the gripper, then choose the useful axis; "
                "the values describe robot motion only, not object or goal positions."
            )
        if "tcp_pixel" in model_state:
            instructions += (
                " tcp_pixel is the active gripper's calibrated image location from robot sensors. "
                "Its u coordinate grows to the right and v grows downward; it may be outside the image. "
                "Use it as the start point when comparing the visible object with action_screen_directions."
            )
        action_request = {
            "model": MODEL,
            "state": model_state,
            "questions": {"action": {
                "type": "choice", "instructions": instructions,
                "criteria": {action.value: description for action, description in ACTION_DESCRIPTIONS.items()},
            }},
        }
        result = self._post(self._vision_payload(action_request, image_base64))
        trace_request = ({**action_request, "phase_decision": phase_evidence}
                         if phase_evidence is not None else action_request)
        decision = self._decision_from_result(result, trace_request, image_audit=image_audit)
        self._sensor_last_action = decision.action.value
        self._sensor_last_phase = phase if self.sensor_policy == "staged" else None
        self._sensor_last_tcp = list(model_state["control_xyz"])
        self._sensor_last_task = task_name
        self._sensor_last_step = step
        return decision

    @staticmethod
    def _encode_image(observation: Observation) -> tuple[str, dict]:
        if observation.image is None:
            raise ValueError("Vision mode requires an image from the environment")
        image_bytes = io.BytesIO()
        observation.image.convert("RGB").save(image_bytes, format="JPEG", quality=85)
        jpeg = image_bytes.getvalue()
        image_audit = {
            "width": observation.image.width,
            "height": observation.image.height,
            "mime_type": "image/jpeg",
            "sha256": hashlib.sha256(jpeg).hexdigest(),
        }
        return base64.b64encode(jpeg).decode("ascii"), image_audit

    def _vision_payload(self, request: dict, image_base64: str) -> dict:
        if self.vision_transport == "native":
            return {**request, "image": "data:image/jpeg;base64," + image_base64}
        return {"request": request, "image_base64": image_base64}

    @staticmethod
    def _response_usage(result: dict) -> dict:
        usage = {key: result[key] for key in ("input_tokens", "image_tokens") if key in result}
        if isinstance(result.get("usage"), dict):
            usage.update(result["usage"])
        return usage

    def _decision_from_result(self, result: dict, request: dict, *, image_audit: dict | None = None) -> Decision:
        try:
            answer = result["answers"]["action"]
            action = Action(answer["choice"])
            probabilities = {str(k): float(v) for k, v in answer.get("probabilities", {}).items()}
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Invalid Jev choice response: {result!r}") from exc
        selection = {"method": self.action_selection, "service_choice": action.value,
                     "policy_seed": self.policy_seed,
                     "sampling_temperature": self.sampling_temperature,
                     "phase_selection": "argmax" if "phase_decision" in request else None}
        if self.action_selection == "sample":
            # Sample only the primitive action. The model-inferred phase remains
            # the service argmax, so this changes a single experimental factor.
            # No simulator truth, action masks, or hand-authored rescue rules.
            keys = [item.value for item in Action]
            if (set(probabilities) != set(keys)
                    or any(not math.isfinite(p) or p < 0 for p in probabilities.values())
                    or not any(p > 0 for p in probabilities.values())):
                raise RuntimeError("Sampling requires finite nonnegative probabilities for all nine actions and positive mass")
            log_mass = [math.log(probabilities[key]) if probabilities[key] > 0 else -math.inf
                        for key in keys]
            peak = max(log_mass)
            weights = [math.exp((value - peak) / self.sampling_temperature) for value in log_mass]
            total = math.fsum(weights)
            effective = [weight / total for weight in weights]
            action = Action(self._action_rng.choices(keys, weights=effective, k=1)[0])
            selection["effective_probabilities"] = dict(zip(keys, effective))
        selection["selected_action"] = action.value
        return Decision(action=action, probabilities=probabilities, request=request,
                        usage=self._response_usage(result), image_audit=image_audit or {},
                        selection=selection)
