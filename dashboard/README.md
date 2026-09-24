# 动态实验回放网页

在项目根目录启动，无需安装额外依赖：

```bash
python3 scripts/serve_dashboard.py --host 0.0.0.0 --port 8787
```

访问 `http://127.0.0.1:8787/`，或使用服务器内网 IP。网页每 5 秒读取
`/api/runs`，也可立即刷新。可按实验、任务 / seed 搜索、信息条件、L0–L3、
成功 / 任务失败 / 执行异常 / 运行中、视觉 / 文本以及是否有 GIF 筛选。
也可按 `argmax` / `sample` 筛选，回合卡片与 GIF 播放器展示 `episode_index`、
环境 seed、`policy_seed` 和采样温度。
环境筛选支持 MetaWorld 与 RoboTwin，卡片、实验分组和 GIF 播放标题均标明环境。
实验条件、仅计划/路标和脱困辅助可分别筛选。L3-P 显示为“仅计划/路标（oracle 辅助）”，
仍属于 `privilege_level=3` / 有特权，独立于完整 L3 统计。
点击预览打开 GIF 播放器。浏览器页面保留滚动与筛选条件，无变化时不重建卡片。

## 控制模式筛选

“控制模式”可与环境、L0–L3、视觉 / 文本、结果等筛选组合，统计行、回放卡片和
GIF 播放器都显示类别：

- **分层控制 · JEV 阶段 → 动作**：`sensor_policy=staged` 的阶段选择与动作选择。
- **原子动作 · 意图 / 视觉方向**：`metaworld_atomic`，含规则候选约束；L0 使用视觉方向，
  L1/L2 使用意图候选，不代表全部由 JEV 自主规划阶段。
- **直接控制 · 基础动作**：直接选择基础动作。
- **Oracle 路标辅助**：基础动作控制使用 L3 / L3-P 的 oracle 阶段与路标，
  即使配置也写了 `staged`，仍单独归类。
- **未标注**：历史记录没有足够的控制配置，或使用尚未识别的动作空间。

“分层 / 原子动作（全部）”同时查看前两类；信息等级继续独立筛选。
分类由服务端按控制器分支优先级推导，API 的 run、group 和 config 提供 `control_mode`。
自动同步保留选择，控制模式也保存在 URL 中，可收藏或分享：

- [只看 JEV 分层控制](http://127.0.0.1:8787/?control-mode=hierarchical_staged)
- [只看原子动作](http://127.0.0.1:8787/?control-mode=atomic)
- [分层 / 原子动作汇总](http://127.0.0.1:8787/?control-mode=hierarchical)

## 结果与统计

- 递归读取 `runs/**/summary.json` 和 `runs/**/matrix.json`，跳过 source、隐藏目录和符号链接。
- 并行 matrix 的 `jobs[].summary` / `output_dir` 定位子任务 summary；
  以 matrix 所在目录作为实验身份，每个 job 只计算一次。子 summary 尚未写出时，
  展示 matrix 中的 running / queued / error；completed 但尚无子结果时等待同步。
- `config.environment` / `results[].environment` 支持 `metaworld` 与 `robotwin`，
  结果字段优先于配置；旧记录缺少该字段时默认 MetaWorld。环境参与配置散列，
  即使实验名、任务名、seed、等级与策略相同，也不会跨环境合并成功率或重复回合汇总。
- 每个环境与实验按特权等级和配置分别计算成功率；不同实验从不合并成一个成功率。
  计划数包括已完成、待运行和异常，成功率分母只包括有效完成的成功 / 任务失败回合。
- `plan_only`、`condition` 与 `stuck_recovery` 参与配置分组。旧记录默认
  `plan_only=false`、`stuck_recovery=none`，条件从等级推导为 L0–L3。
  仅显式 `plan_only=true` 且等级 L3 才是 L3-P：保留既有 oracle 阶段与路标，
  去掉环境物体位置等额外真值，仍属于 oracle 规划辅助。
- `stuck_recovery=none|jitter` 分别统计，每回合展示 `intervention_count`，
  分组及 task / env seed 汇总累加该次数；次数属于运行结果，不拆分策略组。
- jitter 配置的签名包含 `recovery_window`、`recovery_displacement_m`、
  `recovery_cooldown`、`recovery_max_interventions`、`recovery_steps`，
  缺失时分别使用默认值 `6`、`0.003`、`6`、`3`、`2`；不同参数分别统计。
  none 模式忽略这些参数差异。`recovery_seed` 为每回合信息，不拆分策略组。
- 相同 task / env seed 的 `episodes=5` 会保留五条独立回合；不以 task / seed 去重。
  `action_selection` 与 `sampling_temperature` 区分策略组，`policy_seed` 和
  `episode_index` 是回合信息，不拆分策略组。矩阵父配置的 episodes 表示计划重复次数，
  子 summary 的 episodes=1 不会减少矩阵计划数。串行 summary 未完成的重复回合显示待运行。
- 每组可展开 task / env seed 汇总：完整五回合显示例如 `3/5 成功 · 5/5 有效完成`；
  尚有异常 / 待运行时显示有效分母及计划数，例如 `2/3 成功 · 3/5 有效完成`，
  并列出异常 / 运行中 / 待运行数量。汇总不会跨等级或采样策略合并。
- `status=error`、含错误字段或者显式零决策 / 零仿真步的回合属于执行异常，
  不计入任务失败与成功率分母。页面展示的是调试实验，非 benchmark 结论。
- `runs/gifs/**/*.gif` 与同名 JSON sidecar 补充回放与等级信息。
  只有 GIF、没有 summary 的历史记录仍可浏览，但不计成功率。
- 有显式 `privilege_level` 时以其为准；旧配置中 nonprivileged 为 L0，
  waypoints 为 L3，privileged/direct 以及旧 summary 的默认配置为 L2。
  缺少任何配置的历史 GIF 等级显示未标注。
- JSON 暂时半写或不可读时继续使用该文件上次有效快照并显示提示。
  目录扫描整体失败时保留上次完整 API 响应。
- 运行状态来自调度文件；页面显示最后更新时间，不把长期不更新的 running 记录
  当作进程仍存活的证明。旧 summary 未完成的任务显示待运行 / 待同步。

等级定义：L0 是 RGB、本体、自身历史与相机标定；L1 加物体 / 目标 / 场景真实位置；
L2 加接触、抬升、距离与完成几何指标；L3 再加 oracle 阶段与 waypoint 路标。

API 返回 `runs`（每条回合 / 回放）、`groups`（每个实验 / 配置 / 等级的完整分母）、
`counts`、`experiments`、`updated_at`、`generated_at` 和快照状态。
`eligible` 指是否计入成功率；`media_only` 指只存在回放记录。
配置只发布固定白名单字段，配置标识是散列值，不发布 API key、服务 URL、原始错误或 trace。

## 文件访问边界

HTTP 仅公开 `/`、`/index.html`、`/dashboard/`、`/dashboard.js`、`/dashboard.css`、
`/api/runs` 和 `runs/gifs` 下的 GIF / PNG。没有目录列表功能，不能访问原始 summary、
JSONL、sidecar、源码、日志或 Git 文件。

每个路径分量通过 Linux `openat` / `O_NOFOLLOW` 打开，阻止文件和目录符号链接，
并避免单纯 resolve 检查与读取之间的替换竞态。HTTP 拒绝点路径、反斜杠和 NUL；
非普通文件不可读取。界面通过 DOM `textContent` 展示不受信任的任务名和实验名，
并设置 CSP 与 `nosniff`。服务只读实验产物，不修改仿真、trace 或结果目录。

## 验证

```bash
python3 -m unittest discover -s dashboard -p 'test_*.py' -v
node --check dashboard/dashboard.js
```

测试使用临时目录，覆盖同 task / seed 五回合计数、策略与温度隔离、重复回合进度、
等级兼容、异常分母、配置隔离、半写恢复、嵌套任务去重、
sidecar 合并、敏感字段过滤，以及 HTTP 路径 / 符号链接隔离。

## 部署

默认以 `--host 127.0.0.1` 供本机访问；需要其他机器访问时配置实际监听地址。
长期运行可用进程管理器管理服务和日志。修改 Python 服务代码后需要重启进程，
静态页面与实验结果在下次请求时自动读取。
