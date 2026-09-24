# JEV Robotics Eval

A framework for evaluating JEV robot control in **MetaWorld and RoboTwin**. Both simulators are integrated through working environment adapters and share the policy interface, evaluation loop, experiment records, and replay dashboard.

JEV chooses discrete actions from text or text plus images. The simulator executes each action and returns a new observation. The framework supports comparisons across observation privilege levels, model-selected stages, bounded action intents, oracle waypoint guidance, and optional recovery assistance.

## Supported environments

| Environment | Integration | Registered benchmark tasks |
| --- | --- | --- |
| **MetaWorld** | MT1 adapter, text and vision decisions, primitive actions and bounded intents; tested with MetaWorld 3.1.1 | 9 tasks: `reach-v3`, `push-v3`, `door-open-v3`, `drawer-open-v3`, `pick-place-v3`, `peg-insert-side-v3`, `shelf-place-v3`, `bin-picking-v3`, `assembly-v3` |
| **RoboTwin** | Adapter using the upstream planner, head-camera RGB, robot proprioception, and task success checks | 3 tasks: `click_bell`, `press_stapler`, `move_pillbottle_pad` |

RoboTwin currently controls one fixed arm with Cartesian translations, gripper commands, and a fixed end-effector orientation. Rotation commands and coordinated dual-arm tasks are outside the current adapter's scope. The arm is selected by experiment configuration, independently of object positions.

`--benchmark` runs the **task subset registered in this project**. It does not cover all MetaWorld or RoboTwin tasks. Physics steps and control budgets differ between the two simulators, so their results are reported separately.

## Installation

Use Python 3.10+ and a working simulator installation. The MetaWorld examples below require MuJoCo rendering; headless EGL rendering also requires compatible graphics drivers.

```bash
git clone https://github.com/lose4578/jev-robotics-eval.git
cd jev-robotics-eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[metaworld]'
```

The package and command are named `jev-robo-eval`; the Python module is `jev_robo_eval`.

Connect a separately deployed decision service that implements the request and response protocol used by the [JEV client](src/jev_robo_eval/jev.py). Pass its full endpoint URL with `--jev-url` or `--jev-urls`. Vision mode requires an image-capable service. The client supports `/v1/decision` and `/v1/vision-decision` endpoints. Set `JEV_MODEL` if the service requires a model identifier other than the default `jev`. If the service requires authentication, set `JEV_API_KEY` in the evaluation process.

Model weights, the JEV inference runtime, RoboTwin assets, and historical GIFs and traces are external to this repository. For RoboTwin, install its dependencies and assets according to the upstream instructions, then install this project in that Python environment with `python -m pip install -e .` from the project root.

## Run an evaluation

Run commands from the repository root. Replace the example endpoint URLs with your deployed services. Always select an observation level explicitly; the default configuration uses privileged state.

### MetaWorld: text or vision

Run a text episode using L2 geometric and interaction state:

```bash
MUJOCO_GL=egl jev-robo-eval \
  --task reach-v3 --mode text --privilege-level 2 \
  --sensor-policy direct --action-space primitive \
  --jev-url http://127.0.0.1:8186/v1/decision \
  --seed 2 --task-index 0 --action-repeat 3 --max-decisions 120
```

Run a vision episode using L0 observations, with JEV selecting a stage and then an action:

```bash
MUJOCO_GL=egl jev-robo-eval \
  --task shelf-place-v3 --mode vision --privilege-level 0 \
  --sensor-policy staged --action-space primitive --proprio-projection \
  --jev-url http://127.0.0.1:8186/v1/decision \
  --seed 2 --image-size 384 --move-scale 0.5 \
  --action-repeat 3 --max-decisions 170
```

These commands save a JSONL trace and a GIF under `runs/`. The single-episode command targets MetaWorld; use the batch scripts for either environment.

### MetaWorld: parallel benchmark and repeated rollouts

This example runs each registered task five times at environment seed 2, using two independently deployed JEV endpoints:

```bash
MUJOCO_GL=egl python scripts/evaluate_parallel.py \
  --benchmark --environment metaworld \
  --mode vision --privilege-level 0 --sensor-policy staged \
  --action-space primitive --proprio-projection \
  --jev-urls http://127.0.0.1:8186/v1/decision \
             http://127.0.0.1:8191/v1/decision \
  --seeds 2 --episodes 5 --image-size 384 \
  --move-scale 0.5 --action-repeat 3 --max-decisions 170 \
  --output-dir runs/metaworld-L0-staged
```

One endpoint is sufficient; each endpoint handles its assigned episodes sequentially, while different endpoints run in parallel. Replace `--benchmark` with `--tasks shelf-place-v3 bin-picking-v3 assembly-v3` to select tasks explicitly. Use a new output directory for each experiment.

When `--action-space` is omitted, the MetaWorld benchmark defaults to `metaworld_atomic`. The explicit `primitive` setting above selects JEV stage/action control. For L0 atomic evaluation, `--proprio-projection` is required. Some MetaWorld tasks define their target only through a virtual marker; hiding that marker at L0 removes the visible goal, which limits how their L0 results can be interpreted.

### RoboTwin: parallel benchmark

Run this command in the Python environment containing RoboTwin and this project. Set the checkout path explicitly and use the upstream `demo_clean` configuration:

```bash
python scripts/evaluate_parallel.py \
  --benchmark --environment robotwin \
  --robotwin-root /path/to/RoboTwin --robotwin-config demo_clean \
  --active-arm left --robotwin-initial-pose topdown \
  --mode vision --privilege-level 0 --sensor-policy staged \
  --action-space primitive --proprio-projection \
  --jev-urls http://127.0.0.1:8186/v1/decision \
             http://127.0.0.1:8191/v1/decision \
  --seeds 2 --episodes 5 --image-size 384 \
  --move-scale 0.5 --action-repeat 1 --max-decisions 80 \
  --output-dir runs/robotwin-L0-staged
```

`topdown` initializes the active arm at a fixed pose using robot geometry and constants; `home` retains the upstream initial pose. RoboTwin preserves the head camera's aspect ratio, so `--image-size 384` produces 384 × 288 images. `--proprio-projection` derives screen directions from camera calibration and the robot's own pose.

For multiple simulator GPUs, add `--simulator-gpus 0 1`, supplying one GPU index per endpoint.

## Observation levels and control modes

Observation access and control mode are separate experiment choices. Policy inputs use explicit field allowlists; full evaluation state is retained for logging and scoring.

| Condition | Available information |
| --- | --- |
| **L0** | RGB, robot proprioception, action history, and camera calibration; requires vision mode |
| **L1** | L0 sensor information plus object, target, and scene positions |
| **L2** | L1 plus interaction and progress facts such as contact, lift, and relative distances |
| **L3** | Oracle stages, waypoints, and desired gripper state generated by handwritten rules using simulator truth |
| **L3-P** | `--privilege-level 3 --plan-only`: retains oracle plan/waypoint information, proprioception, and RGB while removing additional scene truth |

L1–L3 can omit RGB when a compatible text policy branch is used; the quickstart demonstrates MetaWorld L2 text. L0 hides MetaWorld virtual sites and rejects oracle waypoint annotations. L3-P remains oracle-assisted because its plan generator still reads simulator truth.

| Control mode | Selection | Behavior |
| --- | --- | --- |
| Direct primitives | `--action-space primitive --sensor-policy direct` | JEV chooses one primitive action from the current observation. |
| JEV stage/action control | `--action-space primitive --sensor-policy staged`, L0–L2 vision | JEV infers a stage from a constrained set, then chooses a primitive action. Both requests are recorded. |
| MetaWorld bounded intents | `--action-space metaworld_atomic` | JEV chooses an intent from rule-constrained candidates; an executor maps it to at most one environment step. L0 uses calibrated visual directions; L1/L2 can use privileged position coordinates. |
| Oracle waypoint assistance | `--action-space primitive --privilege-level 3` | Handwritten rules supply the stage and waypoint; JEV chooses the next primitive action. `--plan-only` selects L3-P. |

The nine primitives are `x_pos`, `x_neg`, `y_pos`, `y_neg`, `z_pos`, `z_neg`, `grip_open`, `grip_close`, and `hold`. Gripper commands persist during later moves. `--move-scale` and `--action-repeat` set the execution budget of each primitive; one environment step can include several simulator steps.

Bounded intents include alignment, height adjustment, lifting, waypoint tracking, and gripper operations. Candidate eligibility and coordinate-based execution add controller assistance, so their success rates measure the combined policy and controller. They are reported separately from JEV stage selection and oracle waypoint guidance.

## Sampling, recovery, and evidence

`--episodes 5` resets each task/seed pair five times. The default `--action-selection argmax` can produce identical trajectories. To sample from JEV's returned action probabilities, add:

```bash
--action-selection sample --policy-seed 100 --sampling-temperature 1.0
```

The five policy seeds are then 100–104, independently of the environment seed. Stage selection remains argmax. Five rollouts of one scene are repeated trials of that scene; an “at least one success in five” statistic uses five attempts and must be reported with that budget.

Optional `--stuck-recovery jitter` detects stalls from robot proprioception and action history, then inserts bounded random translations. It uses the same decision budget and records the proposed and executed actions plus each intervention. The default is `none`. Recovery-assisted results and oracle-assisted results have separate labels.

Batch evaluation saves `summary.json`, per-episode traces, source snapshots, and success/failure GIFs. Parallel batches also maintain `matrix.json` with queued, running, completed, and error states. Vision batches and RoboTwin record actual observed frames, including partial error episodes when frames are available. MetaWorld text batches and the single-episode CLI export GIFs by replaying actions and checking the numerical outcomes.

Traces retain model requests, probabilities, execution details, and image/token audit metadata when supplied by the service. API keys are not written to these records. Success comes from the upstream simulator's task criterion. RoboTwin's stapler task primarily checks contact at a designated point; success does not establish completion of a physical stapling mechanism.

## Replay dashboard

```bash
python scripts/serve_dashboard.py --host 127.0.0.1 --port 8787
```

Open [http://127.0.0.1:8787/](http://127.0.0.1:8787/). The dashboard refreshes every five seconds and filters by environment, task, seed, experiment, observation level, control mode, sampling method, recovery setting, and outcome. Successes, task failures, execution errors, and pending episodes remain distinguishable; execution errors are excluded from the valid success-rate denominator.

Control filters distinguish JEV stage/action control, bounded intents, direct primitives, and oracle assistance. For example, [JEV stage/action runs](http://127.0.0.1:8787/?control-mode=hierarchical_staged) and [atomic runs](http://127.0.0.1:8787/?control-mode=atomic) have separate views. The control-mode selection is stored in the URL; automatic refresh also preserves the other active filters. The current dashboard interface and its [detailed documentation](dashboard/README.md) are in Chinese.

The server exposes dashboard assets, aggregate results, and GIF/PNG replays; raw traces and configuration files are not served. Each experiment and configuration retains its own statistics.

## Performance summary

The latest completed registered-task evaluation used the **JEV** model in vision-plus-text mode. Each task was evaluated once with a single environment seed and argmax action selection. Results are separated by environment, observation level, and controller.

| Model | Environment | Observation level | Controller | Successes / episodes | Success rate |
| --- | --- | --- | --- | ---: | ---: |
| JEV | MetaWorld | L0 | Bounded intents / visual directions | 0 / 9 | 0.0% |
| JEV | MetaWorld | L1 | Bounded intents with position truth | 7 / 9 | 77.8% |
| JEV | MetaWorld | L2 | Bounded intents with interaction truth | 8 / 9 | 88.9% |
| JEV | RoboTwin | L0 | JEV stage selection + primitives | 0 / 3 | 0.0% |
| JEV | RoboTwin | L1 | JEV stage selection + primitives | 0 / 3 | 0.0% |
| JEV | RoboTwin | L2 | JEV stage selection + primitives | 0 / 3 | 0.0% |
| JEV | RoboTwin | L3 | Oracle waypoints + primitives | 1 / 3 | 33.3% |

These scores cover the 9 MetaWorld and 3 RoboTwin tasks listed above. They are development results from a single seed, with some settings tuned on these scenes; they do not estimate performance across either simulator's full task distribution. MetaWorld bounded intents include rule-constrained candidates and coordinate-based execution. RoboTwin L3 uses oracle guidance. These assistance conditions are part of the evaluated system, and the scores should be interpreted accordingly.

The table counts completed episodes with valid outcomes; execution errors are excluded. Raw experiment records, per-episode results, audit reports, deployment notes, traces, and GIFs are private and are not distributed with this repository. The framework lets users generate their own records and replays locally.

## Development

Install the test dependencies and run the suite in an environment with working EGL rendering:

```bash
python -m pip install -e '.[metaworld,test]'
MUJOCO_GL=egl python -m pytest tests dashboard/test_dashboard.py -q
node --check dashboard/dashboard.js
```

Node.js is only needed for the JavaScript syntax check. Tests cover observation boundaries, action selection, intents, recovery, recording, batch configuration, and dashboard behavior. Some tests render MetaWorld scenes; the suite does not replace end-to-end experiments with a live JEV service and installed RoboTwin assets.

The shared interfaces live in [core.py](src/jev_robo_eval/core.py). An environment implements `reset(seed=...)`, `step(Action)`, and `close()`; a policy implements `decide(observation, task)`. The existing [MetaWorld adapter](src/jev_robo_eval/metaworld_env.py), [RoboTwin adapter](src/jev_robo_eval/robotwin_env.py), [JEV client](src/jev_robo_eval/jev.py), and [runner](src/jev_robo_eval/runner.py) provide the integration pattern.

## License

Project code is available under the [MIT License](LICENSE). External decision services, MetaWorld, MuJoCo, RoboTwin, and their models and assets retain their respective licenses.
