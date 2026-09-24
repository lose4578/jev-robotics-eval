"""Validated GIF exports and one shared browsing index for every outcome."""

from html import escape
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def _rebuild_index(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    gifs = sorted(root.rglob("*.gif"))
    rows = ["# 全部任务回放", "", "成功与失败统一保存；按实验文件夹分组，文件名标明结果。", "",
            "| 实验 | 信息条件 | 回放 | 结果 |", "| --- | --- | --- | --- |"]
    cards = []
    for gif in gifs:
        path = gif.relative_to(root).as_posix()
        metadata_path = gif.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        result = ("执行异常" if metadata.get("status") == "error" or metadata.get("outcome") == "error"
                  or gif.stem.endswith("-error") else "成功" if gif.stem.endswith("-success") else "失败")
        information = metadata.get("information", "unknown")
        label = {"privileged": "有特权", "nonprivileged": "无特权", "unknown": "未标注"}[information]
        rows.append(f"| {gif.parent.name} | {label} | [{gif.stem}]({path}) | {result} |")
        preview = escape(Path(path).with_suffix(".png").as_posix(), quote=True)
        cards.append(f'<article data-status="{result}" data-information="{information}" data-search="{escape(path.lower(), quote=True)}"><a href="{escape(path, quote=True)}"><img loading="lazy" src="{preview}" alt="{escape(gif.stem)}"></a>'
                     f'<p>{escape(gif.parent.name)}<br><b>{label} · {result}</b> · <a href="{escape(path, quote=True)}">{escape(gif.stem)}</a></p></article>')
    (root / "README.md").write_text("\n".join(rows) + "\n")
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>JEV 仿真回放</title>
<style>body{font:16px system-ui;margin:24px;background:#111827;color:#eee}a{color:#93c5fd}main{display:flex;flex-wrap:wrap;gap:18px}article{width:300px;background:#1f2937;padding:12px;border-radius:8px;overflow-wrap:anywhere}img{width:100%}button,input{margin:0 10px 20px 0;padding:8px 18px}input{width:min(440px,80%)}label{display:block}</style>
<h1>JEV 仿真回放</h1><p>成功、失败统一收集。点击预览图或文件名播放 GIF；预览图是最后一帧。</p>
<label for="search">按任务、模式或实验名称搜索</label><input id="search" type="search" placeholder="例如 hard-vision、assembly、vision seed2" oninput="applyFilters()">
<div><button onclick="filter('全部')">全部结果</button><button onclick="filter('成功')">成功</button><button onclick="filter('失败')">失败</button><button onclick="filter('执行异常')">执行异常</button>
<label for="information">信息条件</label><select id="information" onchange="applyFilters()"><option value="all">全部条件</option><option value="privileged">有特权</option><option value="nonprivileged">无特权</option></select> <span id="count" aria-live="polite"></span></div><main>'''
    html += "\n".join(cards) + '''</main><script>
let selectedStatus='全部';
function filter(s){selectedStatus=s;applyFilters()}
function applyFilters(){
  const terms=document.getElementById('search').value.trim().toLowerCase().split(/\\s+/).filter(Boolean);
  const information=document.getElementById('information').value;
  let count=0;
  document.querySelectorAll('article').forEach(x=>{
    x.hidden=(selectedStatus!=='全部'&&x.dataset.status!==selectedStatus)||(information!=='all'&&x.dataset.information!==information)||!terms.every(t=>x.dataset.search.includes(t));
    if(!x.hidden) count++;
  });
  document.getElementById('count').textContent=count+' 个回放';
}
applyFilters();</script></html>'''
    (root / "index.html").write_text(html)


def rebuild_index(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Text and vision evaluations can finish episodes concurrently.
    with (root / ".index.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _rebuild_index(root)


def export_episode(trace: Path, output: Path, *, task: str, seed: int,
                   task_index: int = 0, action_repeat: int = 3,
                   move_scale: float = 1.0, guidance: str = "direct", annotate_vision: bool = False,
                   information: str = "privileged", proprio_projection: bool = False,
                   privilege_level: int | None = None, plan_only: bool = False,
                   metadata: dict | None = None) -> Path:
    if not output.exists():
        env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
        command = [
            sys.executable, "-m", "jev_robo_eval.replay", "--trace", str(trace),
            "--output", str(output), "--task", task, "--seed", str(seed),
            "--task-index", str(task_index), "--action-repeat", str(action_repeat),
            "--move-scale", str(move_scale), "--guidance", guidance,
            "--information", information,
        ]
        if plan_only:
            command.append("--plan-only")
        if annotate_vision:
            command.append("--annotate-vision")
        if proprio_projection:
            command.append("--proprio-projection")
        if privilege_level is not None:
            command.extend(["--privilege-level", str(privilege_level)])
        subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)
    output.with_suffix(".json").write_text(json.dumps({
        "information": information, "task": task, "seed": seed, "guidance": guidance,
        "trace": str(trace), "task_index": task_index, "action_repeat": action_repeat,
        "move_scale": move_scale, "annotate_vision": annotate_vision,
        "proprio_projection": proprio_projection, "plan_only": plan_only,
        "privilege_level": privilege_level if privilege_level is not None else (
            0 if information == "nonprivileged" else 3 if guidance == "waypoints" else 2),
        **(metadata or {}),
    }, indent=2) + "\n")
    return output
