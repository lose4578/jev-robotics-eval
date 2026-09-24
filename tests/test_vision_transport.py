"""The two HTTP transports must send identical image bytes and retain real usage."""

from __future__ import annotations

import base64
import hashlib
import json

from PIL import Image
import pytest

from jev_robo_eval.core import Action, Observation
from jev_robo_eval.jev import JevPolicy


def _observation() -> Observation:
    image = Image.new("RGB", (32, 24), (20, 100, 180))
    image.putpixel((7, 9), (240, 10, 20))
    return Observation(state={
        "information": "nonprivileged",
        "privilege_level": 0,
        "task_name": "shelf-place-v3",
        "control_xyz": [0.0, 0.6, 0.15],
        "gripper_opening": 1.0,
        "gripper_command": "open",
        "simulator_steps": 0,
    }, image=image)


@pytest.mark.parametrize("url,expected_transport", [
    ("http://127.0.0.1:8186/v1/decision", "native"),
    ("http://127.0.0.1:8081/v1/vision-decision", "bridge"),
])
def test_auto_vision_transport_and_response_usage(monkeypatch, url, expected_transport):
    policy = JevPolicy(mode="vision", url=url)
    sent = []

    def fake_post(payload):
        sent.append(payload)
        return {"answers": {"action": {"choice": "x_neg", "probabilities": {"x_neg": 0.7}}},
                "input_tokens": 501, "image_tokens": 96}

    monkeypatch.setattr(policy, "_post", fake_post)
    decision = policy.decide(_observation(), "Place the block on the shelf.")
    assert policy.vision_transport == expected_transport
    assert decision.action == Action.X_NEG
    assert decision.usage == {"input_tokens": 501, "image_tokens": 96}
    assert decision.image_audit["width"] == 32
    assert decision.image_audit["height"] == 24
    assert decision.image_audit["mime_type"] == "image/jpeg"
    assert decision.image_audit["transport"] == expected_transport
    if expected_transport == "native":
        assert set(sent[0]) == {"model", "state", "questions", "image"}
        assert sent[0]["image"].startswith("data:image/jpeg;base64,")
        raw = base64.b64decode(sent[0]["image"].split(",", 1)[1])
    else:
        assert set(sent[0]) == {"request", "image_base64"}
        raw = base64.b64decode(sent[0]["image_base64"])
    assert raw.startswith(b"\xff\xd8")
    assert hashlib.sha256(raw).hexdigest() == decision.image_audit["sha256"]
    assert "base64" not in json.dumps(decision.request)


def test_native_and_bridge_jpeg_bytes_are_identical(monkeypatch):
    observation = _observation()
    images = []
    for url in ("http://localhost:8186/v1/decision",
                "http://localhost:8081/v1/vision-decision"):
        policy = JevPolicy(mode="vision", url=url)

        def fake_post(payload):
            images.append(payload.get("image", payload.get("image_base64")))
            return {"answers": {"action": {"choice": "hold"}}}

        monkeypatch.setattr(policy, "_post", fake_post)
        decision = policy.decide(observation, "Place the block on the shelf.")
        assert decision.usage == {}  # A missing service field is not estimated.
    assert images[0].split(",", 1)[1] == images[1]


def test_staged_preserves_both_actual_usage_records(monkeypatch):
    policy = JevPolicy(mode="vision", url="http://localhost:8186/v1/decision",
                       sensor_policy="staged")
    images = []

    def fake_post(payload):
        images.append(payload["image"])
        if "phase" in payload["questions"]:
            return {"answers": {"phase": {"choice": "approach",
                                            "probabilities": {"approach": 0.8}}},
                    "input_tokens": 410, "image_tokens": 96}
        return {"answers": {"action": {"choice": "x_neg"}},
                "input_tokens": 425, "image_tokens": 96}

    monkeypatch.setattr(policy, "_post", fake_post)
    decision = policy.decide(_observation(), "Place the block on the shelf.")
    assert decision.action == Action.X_NEG
    assert images[0] == images[1]
    assert decision.request["phase_decision"]["usage"] == {
        "input_tokens": 410, "image_tokens": 96,
    }
    assert decision.usage == {"input_tokens": 425, "image_tokens": 96}
    assert decision.selection["phase_selection"] == "argmax"


def test_unknown_auto_path_requires_explicit_transport():
    with pytest.raises(ValueError, match="Cannot infer vision transport"):
        JevPolicy(mode="vision", url="http://localhost/custom")
    assert JevPolicy(mode="vision", url="http://localhost/custom",
                     vision_transport="native").vision_transport == "native"


def test_text_decision_keeps_only_reported_usage(monkeypatch):
    policy = JevPolicy(mode="text")

    def fake_post(payload):
        assert "image" not in payload
        return {"answers": {"action": {"choice": "hold"}}, "input_tokens": 72}

    monkeypatch.setattr(policy, "_post", fake_post)
    decision = policy.decide(Observation(state={"task_name": "reach-v3",
                                                "control_xyz": [0, 0, 0],
                                                "goal_xyz": [0, 0, 0]}), "Reach the target")
    assert decision.usage == {"input_tokens": 72}
    assert decision.image_audit == {}
