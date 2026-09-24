#!/usr/bin/env python3
"""Serve a read-only, polling replay dashboard using only the Python standard library.

Only three bundled dashboard files and GIF/PNG media under runs/gifs are public.
Run manifests are projected into an allowlisted API; raw manifests and traces are
never served. Each opened path component rejects symlinks, including during races.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import time
from urllib.parse import quote, unquote, urlsplit

CONFIG_FIELDS = (
    "environment", "mode", "information", "privilege_level", "sensor_policy", "action_space", "control_mode", "benchmark", "guidance", "plan_only", "condition",
    "stuck_recovery",
    "recovery_window", "recovery_displacement_m", "recovery_cooldown", "recovery_max_interventions", "recovery_steps",
    "image_size", "annotate_vision", "proprio_projection", "action_repeat",
    "move_scale", "max_decisions", "task_index", "action_selection", "sampling_temperature",
)
ROLLOUT_FIELDS = ("episode_index", "policy_seed", "recovery_seed")
RECOVERY_DEFAULTS = {"recovery_window": 6, "recovery_displacement_m": 0.003, "recovery_cooldown": 6,
                     "recovery_max_interventions": 3, "recovery_steps": 2}
SKIP_DIRS = {"source", "__pycache__", ".git", ".venv"}
LEVEL_LABELS = {0: "L0 · 无特权", 1: "L1 · 位置真值", 2: "L2 · 交互 / 进度真值", 3: "L3 · 阶段 / 路标"}
MAX_JSON_BYTES = 16 * 1024 * 1024


def utc_timestamp(value: float | None = None) -> str:
    return datetime.fromtimestamp(value if value is not None else time.time(), timezone.utc).isoformat()


def safe_parts(relative: str) -> tuple[str, ...]:
    if not relative or relative.startswith("/") or "\\" in relative or "\x00" in relative:
        raise ValueError("Invalid relative path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Invalid path component")
    return tuple(parts)


def open_beneath(root: Path, relative: str):
    """Open a regular file with no symlink traversal, including check/open races."""
    parts = safe_parts(relative)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("Not a regular file")
    return os.fdopen(fd, "rb")


def discover(root: Path, *, names: set[str] | None = None, suffixes: set[str] | None = None,
             exclude_gifs: bool = False):
    if root.is_symlink() or not root.is_dir():
        return
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories
                                if name not in SKIP_DIRS and not name.startswith(".")
                                and not (exclude_gifs and Path(current) == root and name == "gifs")
                                and not (Path(current) / name).is_symlink())
        for name in sorted(files):
            path = Path(current) / name
            if not path.is_symlink() and ((names is not None and name in names)
                                         or (suffixes is not None and path.suffix.lower() in suffixes)):
                yield path.relative_to(root).as_posix()


def as_dict(value):
    return value if isinstance(value, dict) else {}


def as_list(value):
    return value if isinstance(value, list) else []


def public_scalar(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value is None or isinstance(value, (str, int, float, bool)) else None


def text_scalar(value, fallback=""):
    return str(value)[:400] if isinstance(value, (str, int, float)) else fallback


def positive_int(value, fallback=1):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else fallback


def environment_name(value):
    return text_scalar(value, "metaworld").strip().lower() or "metaworld"


def privilege(config: dict, *, legacy_default: bool = True) -> tuple[int | None, str]:
    value = config.get("privilege_level")
    if value is not None and not isinstance(value, bool):
        raw = str(value).lower().removeprefix("l")
        if raw in {"0", "1", "2", "3"}:
            return int(raw), "explicit"
    if config.get("information") == "nonprivileged":
        return 0, "legacy"
    if config.get("guidance") == "waypoints":
        return 3, "legacy"
    if legacy_default or config.get("information") == "privileged" or config.get("guidance") == "direct":
        return 2, "legacy"
    return None, "unknown"


def control_mode(config: dict, level: int | None) -> str:
    """Classify controller branches independently of observation privilege.

    Atomic actions bypass the staged planner. L3 oracle waypoints bypass JEV
    phase selection even with sensor_policy=staged. Missing metadata is unknown.
    """
    action_space = config.get("action_space")
    if action_space in ("metaworld_atomic", "atomic"):
        return "atomic"
    if action_space not in (None, "primitive", "legacy"):
        return "unknown"
    # resolve_access makes waypoint guidance mandatory for L3 (including L3-P).
    if level == 3:
        return "oracle_waypoints"
    if config.get("sensor_policy") == "staged":
        return "hierarchical_staged"
    if config.get("sensor_policy") == "direct" or config.get("guidance") == "direct":
        return "direct"
    return "unknown"


def episode_status(result: dict) -> tuple[str, str | None]:
    raw = text_scalar(result.get("status")).lower()
    if raw in {"error", "failed_to_start", "cancelled", "canceled", "timeout"} or result.get("error"):
        return "error", "execution_error"
    if raw in {"running", "started", "active"}:
        return "running", None
    if raw in {"queued", "pending", "planned"}:
        return "pending", None
    if result.get("decisions") == 0 or result.get("simulator_steps") == 0:
        return "error", "zero_actions"
    success = result.get("success")
    if isinstance(success, bool):
        return ("success" if success else "failed"), None
    return "unknown", None


class Snapshot:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.runs_root = self.project_root / "runs"
        self.media_root = self.runs_root / "gifs"
        self.lock = threading.Lock()
        self.json_cache: dict[tuple[str, str], tuple[dict, float, tuple]] = {}
        self.snapshot: dict | None = None
        self.checked_at = 0.0
        self.stale_files: set[str] = set()
        self.read_count = 0

    def read_json(self, root: Path, relative: str) -> tuple[dict | None, float]:
        key = (str(root), relative)
        old = self.json_cache.get(key)
        try:
            with open_beneath(root, relative) as handle:
                before = os.fstat(handle.fileno())
                signature = (before.st_ino, before.st_size, before.st_mtime_ns)
                if old and old[2] == signature:
                    return old[0], old[1]
                if before.st_size > MAX_JSON_BYTES:
                    raise ValueError("Manifest exceeds size limit")
                raw = handle.read(MAX_JSON_BYTES + 1)
                after = os.fstat(handle.fileno())
                if signature != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError("Manifest changed while reading")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("Manifest must be an object")
                self.json_cache[key] = (value, before.st_mtime, signature)
                self.read_count += 1
                return value, before.st_mtime
        except (OSError, ValueError, UnicodeError, RecursionError):
            self.stale_files.add(relative)
            if old:
                return old[0], old[1]
            return None, 0.0

    def relative_run_path(self, value, base: str = "") -> str | None:
        if not isinstance(value, str) or "\\" in value:
            return None
        path = Path(value)
        if path.is_absolute():
            try:
                relative = path.relative_to(self.runs_root).as_posix()
            except ValueError:
                return None
        elif value.startswith("runs/"):
            relative = value[5:]
        else:
            relative = (PurePosixPath(base) / path).as_posix()
        try:
            safe_parts(relative)
            return relative
        except ValueError:
            return None

    def media_path(self, value) -> str | None:
        relative = self.relative_run_path(value)
        return relative[5:] if relative and relative.startswith("gifs/") else None

    def build(self) -> dict:
        self.stale_files = set()
        manifests = {}
        matrices = {}
        for path in discover(self.runs_root, names={"summary.json", "matrix.json"}, exclude_gifs=True):
            value, modified = self.read_json(self.runs_root, path)
            if value is None:
                continue
            target = matrices if path.endswith("/matrix.json") or path == "matrix.json" else manifests
            target[path] = (value, modified)

        media = {}
        for relative in discover(self.media_root, suffixes={".gif"}):
            try:
                with open_beneath(self.media_root, relative) as handle:
                    modified = os.fstat(handle.fileno()).st_mtime
            except (OSError, ValueError):
                continue
            sidecar_path = str(PurePosixPath(relative).with_suffix(".json"))
            metadata, metadata_modified = ({}, 0.0)
            if (self.media_root / sidecar_path).is_file():
                metadata, metadata_modified = self.read_json(self.media_root, sidecar_path)
            preview = str(PurePosixPath(relative).with_suffix(".png"))
            try:
                with open_beneath(self.media_root, preview):
                    pass
            except (OSError, ValueError):
                preview = None
            media[relative] = {"metadata": metadata or {}, "modified": max(modified, metadata_modified),
                               "url": "/runs/gifs/" + quote(relative, safe="/"),
                               "preview": "/runs/gifs/" + quote(preview, safe="/") if preview else None}

        # A matrix owns its nested jobs, preventing parent/child double counting.
        matrix_jobs = {}
        for path, (matrix, modified) in matrices.items():
            parent = str(PurePosixPath(path).parent)
            for index, value in enumerate(as_list(matrix.get("jobs"))):
                job = as_dict(value)
                summary = self.relative_run_path(job.get("summary") or job.get("summary_path"), parent)
                if not summary:
                    output = self.relative_run_path(job.get("output_dir"), parent)
                    summary = output + "/summary.json" if output else None
                if summary:
                    matrix_jobs[summary] = (parent, matrix, job, index, modified)

        entries = []
        used_media = set()
        consumed_jobs = set()

        def make_entry(experiment, identity, result, config, modified, *, media_only=False, job=None):
            gif = self.media_path(result.get("gif"))
            item_media = media.get(gif)
            media_config = {key: value for key, value in item_media["metadata"].items()
                            if key in CONFIG_FIELDS} if item_media else {}
            merged = {**media_config, **config, **{key: result[key] for key in CONFIG_FIELDS if key in result}}
            merged["environment"] = environment_name(merged.get("environment"))
            merged.setdefault("action_selection", "argmax" if not media_only else "unknown")
            rollout = {**config, **(item_media["metadata"] if item_media else {}),
                       **{key: job[key] for key in ROLLOUT_FIELDS if job and key in job}, **result}
            episode_index = positive_int(rollout.get("episode_index"))
            level, source = privilege(merged, legacy_default=not media_only)
            # A restricted oracle plan remains a privileged L3 condition.
            merged["plan_only"] = merged.get("plan_only") is True
            merged["stuck_recovery"] = text_scalar(merged.get("stuck_recovery"), "none").strip().lower() or "none"
            for key, default in RECOVERY_DEFAULTS.items():
                if merged["stuck_recovery"] == "jitter":
                    merged.setdefault(key, default)
                else:
                    merged.pop(key, None)
            merged["condition"] = ("L3-P" if level == 3 and merged["plan_only"]
                                   else f"L{level}" if level is not None else "unknown")
            merged["control_mode"] = control_mode(merged, level)
            intervention_count = rollout.get("intervention_count", 0)
            if not isinstance(intervention_count, int) or isinstance(intervention_count, bool) or intervention_count < 0:
                intervention_count = 0
            state, issue = episode_status(result)
            if item_media:
                used_media.add(gif)
                modified = max(modified, item_media["modified"])
            # Scheduler errors classify missing outcomes, without erasing completed child episodes.
            if state not in {"success", "failed", "error"} and job and (job.get("status") == "error" or job.get("error")):
                state, issue = "error", "execution_error"
            public_config = {key: public_scalar(merged[key]) for key in CONFIG_FIELDS if key in merged}
            public_config["privilege_level"] = level
            public_config["information"] = "nonprivileged" if level == 0 else "privileged" if level is not None else "unknown"
            # Hash all experimental settings, including unknown future settings, without publishing secrets.
            fingerprint_config = {key: value for key, value in merged.items()
                                  if key not in {"tasks", "seeds", "output_dir", "gif_dir", "workers", "jobs",
                                                 "max_workers", "jev_url", "jev_urls", "started_at", "finished_at",
                                                 "task", "seed", "trace", "gif", "episode_index", "episodes",
                                                 "policy_seed", "policy_seed_base", "base_policy_seed",
                                                 "recovery_seed", "intervention_count", "interventions"}}
            fingerprint_config["privilege_level"] = level
            fingerprint = hashlib.sha256(json.dumps(fingerprint_config, sort_keys=True, default=str).encode()).hexdigest()[:10]
            entry = {
                "id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                "experiment": experiment, "configuration": fingerprint,
                "environment": merged["environment"],
                "control_mode": merged["control_mode"],
                "plan_only": merged["plan_only"], "condition": merged["condition"],
                "stuck_recovery": merged["stuck_recovery"], "intervention_count": intervention_count,
                "task": text_scalar(result.get("task"), "未标注任务"), "seed": public_scalar(result.get("seed")),
                "episode_index": episode_index, "episodes": positive_int(config.get("episodes")),
                "policy_seed": public_scalar(rollout.get("policy_seed")),
                "recovery_seed": public_scalar(rollout.get("recovery_seed")),
                "action_selection": text_scalar(merged.get("action_selection"), "unknown"),
                "sampling_temperature": public_scalar(merged.get("sampling_temperature")),
                "mode": text_scalar(merged.get("mode"), "unknown"), "privilege_level": level,
                "privilege_source": source, "information": public_config["information"], "status": state,
                "issue": issue, "eligible": state in {"success", "failed"} and not media_only,
                "media_only": media_only, "config": public_config,
                "decisions": public_scalar(result.get("decisions")),
                "simulator_steps": public_scalar(result.get("simulator_steps")),
                "wall_seconds": public_scalar(result.get("wall_seconds")),
                "stopped_by": text_scalar(result.get("stopped_by")),
                "artifact_error": bool(result.get("artifact_error")),
                "updated_at": utc_timestamp(modified), "updated_epoch": modified,
                "gif_url": item_media["url"] if item_media else None,
                "preview_url": item_media["preview"] if item_media else None,
            }
            entries.append(entry)

        for path, (manifest, modified) in manifests.items():
            parent = str(PurePosixPath(path).parent)
            owner = matrix_jobs.get(path)
            config = as_dict(manifest.get("config"))
            experiment = owner[0] if owner else parent
            job = owner[2] if owner else None
            if owner:
                config = {**as_dict(owner[1].get("config")), **config}
                if as_dict(owner[1].get("config")).get("episodes"):
                    config["episodes"] = owner[1]["config"]["episodes"]
                modified = max(modified, owner[4])
            results = [as_dict(item) for item in as_list(manifest.get("results")) if isinstance(item, dict)]
            if owner and results:
                consumed_jobs.add(path)
            seen = set()
            occurrences = Counter()
            for index, result in enumerate(results):
                result = dict(result)
                task_seed = (environment_name(result.get("environment", config.get("environment"))),
                             text_scalar(result.get("task")), text_scalar(result.get("seed")))
                occurrences[task_seed] += 1
                result.setdefault("episode_index", (job or {}).get("episode_index")
                                  or config.get("episode_index") or occurrences[task_seed])
                seen.add((*task_seed, positive_int(result.get("episode_index"))))
                if not self.media_path(result.get("gif")):
                    # Older manifests omit GIF links. Match their trace identity only within the experiment.
                    stem = PurePosixPath(text_scalar(result.get("trace"))).stem
                    candidate_prefixes = [parent, PurePosixPath(parent).name]
                    for prefix in candidate_prefixes:
                        for outcome in ("success", "failed"):
                            candidate = f"{prefix}/{stem}-{outcome}.gif"
                            if candidate in media:
                                result["gif"] = "runs/gifs/" + candidate
                make_entry(experiment, f"{path}:{index}", result, config, modified, job=job)
            if not owner:
                for task in as_list(config.get("tasks")):
                    for seed in as_list(config.get("seeds")):
                        episodes = [positive_int(config["episode_index"])] if config.get("episode_index") else range(
                            1, positive_int(config.get("episodes")) + 1)
                        for episode_index in episodes:
                            if (environment_name(config.get("environment")), text_scalar(task),
                                    text_scalar(seed), episode_index) not in seen:
                                pending = {"task": task, "seed": seed, "episode_index": episode_index, "status": "pending"}
                                if isinstance(config.get("policy_seed"), int) and not config.get("episode_index"):
                                    pending["policy_seed"] = config["policy_seed"] + episode_index - 1
                                make_entry(experiment, f"{path}:pending:{task}:{seed}:{episode_index}",
                                           pending, config, modified)

        for path, (parent, matrix, job, index, modified) in matrix_jobs.items():
            if path in consumed_jobs:
                continue
            config = {**as_dict(matrix.get("config")), **as_dict(job.get("config")),
                      **{key: job[key] for key in CONFIG_FIELDS if key in job}}
            result = {**job}
            if result.get("status") == "completed":
                result["status"] = "pending"  # Wait for the child outcome and action evidence.
            make_entry(parent, f"matrix:{parent}:{index}", result, config, modified, job=job)

        for relative, item_media in media.items():
            if relative in used_media:
                continue
            metadata = item_media["metadata"]
            stem = PurePosixPath(relative).stem
            match = re.match(r"(.+)-(text|vision)-seed(-?\d+)(?:-(?:ep|episode)(\d+))?-(success|failed)$", stem)
            result = {**metadata, "gif": "runs/gifs/" + relative, "status": "completed"}
            if match:
                result.setdefault("task", match[1])
                result.setdefault("mode", match[2])
                result.setdefault("seed", int(match[3]))
                if match[4]:
                    result.setdefault("episode_index", int(match[4]))
                result["success"] = match[5] == "success"
            elif stem.endswith(("-success", "-failed")):
                result["success"] = stem.endswith("-success")
            parent = str(PurePosixPath(relative).parent)
            # Sidecars may point into a batch child; use that matrix's experiment when available.
            trace = self.relative_run_path(metadata.get("trace"))
            if trace:
                owner = matrix_jobs.get(str(PurePosixPath(trace).parent / "summary.json"))
                if owner:
                    parent = owner[0]
            make_entry(parent, f"media:{relative}", result, metadata, item_media["modified"], media_only=True)

        groups = {}
        task_seed_groups = {}
        for entry in entries:
            key = (entry["experiment"], entry["privilege_level"], entry["configuration"])
            if key not in groups:
                groups[key] = {"experiment": key[0], "privilege_level": key[1], "configuration": key[2],
                               "environment": entry["environment"],
                               "control_mode": entry["control_mode"],
                               "plan_only": entry["plan_only"], "condition": entry["condition"],
                               "stuck_recovery": entry["stuck_recovery"], "intervention_count": 0,
                               "label": ("L3-P · 仅计划/路标（oracle 辅助）" if entry["condition"] == "L3-P"
                                         else LEVEL_LABELS.get(key[1], "等级未标注")), "config": entry["config"],
                               "total": 0, "success": 0, "failed": 0, "error": 0, "running": 0,
                               "pending": 0, "unknown": 0, "media_only": 0, "updated_epoch": 0.0}
            group = groups[key]
            if entry["media_only"]:
                group["media_only"] += 1
            else:
                group["total"] += 1
                group[entry["status"]] += 1
                group["intervention_count"] += entry["intervention_count"]
                task_seed_key = (*key, entry["task"], text_scalar(entry["seed"]))
                if task_seed_key not in task_seed_groups:
                    task_seed_groups[task_seed_key] = {
                        "environment": entry["environment"], "task": entry["task"], "seed": entry["seed"], "total": 0,
                        "control_mode": entry["control_mode"],
                        "plan_only": entry["plan_only"], "condition": entry["condition"],
                        "stuck_recovery": entry["stuck_recovery"], "intervention_count": 0,
                        "success": 0, "failed": 0, "error": 0, "running": 0, "pending": 0, "unknown": 0,
                    }
                task_seed_groups[task_seed_key]["total"] += 1
                task_seed_groups[task_seed_key][entry["status"]] += 1
                task_seed_groups[task_seed_key]["intervention_count"] += entry["intervention_count"]
            group["updated_epoch"] = max(group["updated_epoch"], entry["updated_epoch"])
        for key, group in groups.items():
            group["denominator"] = group["success"] + group["failed"]
            group["success_rate"] = group["success"] / group["denominator"] if group["denominator"] else None
            group["updated_at"] = utc_timestamp(group["updated_epoch"])
            group["task_seeds"] = []
            for task_seed_key, task_seed in task_seed_groups.items():
                if task_seed_key[:3] != key:
                    continue
                task_seed["denominator"] = task_seed["success"] + task_seed["failed"]
                task_seed["success_rate"] = task_seed["success"] / task_seed["denominator"] if task_seed["denominator"] else None
                group["task_seeds"].append(task_seed)
            group["task_seeds"].sort(key=lambda item: (item["task"], text_scalar(item["seed"])))
        entries.sort(key=lambda entry: (-entry["updated_epoch"], entry["experiment"], entry["task"],
                                        str(entry["seed"]), entry["episode_index"]))
        group_list = sorted(groups.values(), key=lambda group: (-group["updated_epoch"], group["experiment"]))
        last_update = max((entry["updated_epoch"] for entry in entries), default=0.0)
        counts = Counter(entry["status"] for entry in entries if not entry["media_only"])
        return {"generated_at": utc_timestamp(), "updated_at": utc_timestamp(last_update) if last_update else None,
                "poll_seconds": 5, "benchmark": any(
                    bool(entry.get("config", {}).get("benchmark")) for entry in entries),
                "scope_note": ("benchmark 运行：按环境、动作空间、信息等级和配置分别统计成功率；"
                               "不支持任务、运行异常和仅有 GIF 的记录不进入成功率分母。"
                               if any(bool(entry.get("config", {}).get("benchmark")) for entry in entries)
                               else "调试实验，非 benchmark；同一环境、实验、配置与条件内统计成功率；L3-P 与完整 L3、不同脱困辅助分别统计。"),
                "denominator_note": "分母只含完成且有有效结果的回合；运行异常、零动作、待运行与仅有 GIF 的旧记录不计入。",
                "stale_file_count": len(self.stale_files), "stale_snapshot": False,
                "counts": dict(counts), "experiments": sorted({entry["experiment"] for entry in entries}),
                "groups": group_list, "runs": entries}

    def get(self) -> dict:
        with self.lock:
            if self.snapshot is not None and time.monotonic() - self.checked_at < 1.0:
                return self.snapshot
            try:
                fresh = self.build()
            except Exception:
                # Preserve the last complete response if a directory changes during scanning.
                if self.snapshot is None:
                    raise
                fresh = {**self.snapshot, "stale_snapshot": True, "generated_at": utc_timestamp()}
            self.snapshot = fresh
            self.checked_at = time.monotonic()
            return fresh


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "JEVReplay/1.0"
    sys_version = ""

    def do_GET(self):
        self.respond(head=False)

    def do_HEAD(self):
        self.respond(head=True)

    def send_bytes(self, content: bytes, content_type: str, head: bool, status=200):
        self.send_response(status)
        self.headers_common(content_type, len(content), cache="no-store")
        self.end_headers()
        if not head:
            self.wfile.write(content)

    def headers_common(self, content_type: str, length: int, cache="no-cache"):
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")

    def respond(self, head: bool):
        try:
            path = unquote(urlsplit(self.path).path, errors="strict")
            if "\\" in path or "\x00" in path or any(part in {".", ".."} for part in path.split("/")):
                raise ValueError("Invalid path")
            if path == "/api/runs":
                content = json.dumps(self.server.snapshot.get(), ensure_ascii=False, allow_nan=False).encode()
                return self.send_bytes(content, "application/json; charset=utf-8", head)
            static = {"/": "index.html", "/index.html": "index.html", "/dashboard/": "index.html",
                      "/dashboard.js": "dashboard.js", "/dashboard.css": "dashboard.css"}
            if path in static:
                root, relative = self.server.dashboard_root, static[path]
            elif path.startswith("/runs/gifs/"):
                root, relative = self.server.snapshot.media_root, path[len("/runs/gifs/"):]
                if PurePosixPath(relative).suffix.lower() not in {".gif", ".png"}:
                    raise ValueError("Media type not public")
            else:
                raise FileNotFoundError
            with open_beneath(root, relative) as handle:
                info = os.fstat(handle.fileno())
                content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
                if content_type.startswith("text/") or relative.endswith(".js"):
                    content_type += "; charset=utf-8"
                self.send_response(200)
                self.headers_common(content_type, info.st_size)
                self.end_headers()
                if not head:
                    while chunk := handle.read(128 * 1024):
                        self.wfile.write(chunk)
        except (ValueError, UnicodeError, OSError) as exc:
            if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                return
            self.send_bytes(b"Not found\n", "text/plain; charset=utf-8", head, status=404)
        except Exception:
            self.send_bytes(b"Dashboard snapshot temporarily unavailable\n", "text/plain; charset=utf-8", head, status=503)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    server = DashboardServer((args.host, args.port), DashboardHandler)
    server.snapshot = Snapshot(args.project_root)
    server.dashboard_root = args.project_root.resolve() / "dashboard"
    print(f"Replay dashboard listening on http://{args.host}:{args.port} (PID {os.getpid()})", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
