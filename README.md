# JEV Robotics Eval

A framework for evaluating JEV robot control in **MetaWorld and RoboTwin**. Both simulators are integrated through working environment adapters and share the policy interface, evaluation loop, experiment records, and replay dashboard.

JEV chooses discrete actions from text or text plus images. The simulator executes each action and returns a new observation. The framework supports comparisons across observation privilege levels, model-selected stages and movement amplitudes, bounded action intents, oracle waypoint guidance, and optional recovery assistance.

## Supported environments

| Environment | Integration | Registered benchmark tasks |
| --- | --- | --- |
| **MetaWorld** | MT1 adapter, text and vision decisions, primitive actions and a separate bounded-intent controller; tested with MetaWorld 3.1.1 | **16 tasks** covering reaching, pushing, pressing, doors, drawers, windows, and object transfer/insertion |
| **RoboTwin** | Adapter using the upstream planner, head-camera RGB, robot proprioception, and task success checks | **6 tasks** covering pressing and object placement |

The [central task registry](src/jev_robo_eval/task_registry.py) defines the benchmark task sets:

- **MetaWorld, original nine:** `reach-v3`, `push-v3`, `door-open-v3`, `drawer-open-v3`, `pick-place-v3`, `peg-insert-side-v3`, `shelf-place-v3`, `bin-picking-v3`, `assembly-v3`.
- **MetaWorld, seven additions:** `button-press-v3`, `button-press-topdown-v3`, `drawer-close-v3`, `window-open-v3`, `window-close-v3`, `plate-slide-v3`, `plate-slide-side-v3`. These add front/top button pressing, drawer closing, window sliding, and plate sliding.
- **RoboTwin, original three:** `click_bell`, `press_stapler`, `move_pillbottle_pad`.
- **RoboTwin, three additions:** `click_alarmclock`, `place_container_plate`, `place_object_scale`: press an alarm-clock button, place a cup or bowl on a plate, and place an object on an electronic scale.

Registration means the primitive-action adapter and task definitions are available. Reset, rendering, and action-interface smoke checks establish environment compatibility; they do not establish task success or calibration of a controller. The expanded task list does not imply task-specific support in the separate MetaWorld bounded-intent executor.

RoboTwin currently controls one fixed arm with Cartesian translations, gripper commands, and a fixed end-effector orientation. Rotation commands and coordinated dual-arm tasks are outside the current adapter's scope. The arm is selected by experiment configuration, independently of object positions.

`--benchmark` runs the **task subset registered in this project**, using `--action-space primitive` by default. It does not cover all MetaWorld or RoboTwin tasks. Physics steps and control budgets differ between the two simulators, so their results are reported separately.

The new tasks support L0–L2 evaluation and have no L3 oracle guide. The implemented L3 guides cover MetaWorld `push-v3`, `door-open-v3`, `pick-place-v3`, `peg-insert-side-v3`, `shelf-place-v3`, `bin-picking-v3`, `assembly-v3`, and the original three RoboTwin tasks. Select those explicitly with `--tasks` for L3; a full `--benchmark --privilege-level 3` is rejected because it includes tasks without guides.

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

Connect a decision service that implements the request and response protocol used by the [JEV client](src/jev_robo_eval/jev.py). Pass its full endpoint URL with `--jev-url` or `--jev-urls`. Vision mode requires an image-capable service. The client supports `/v1/decision` and `/v1/vision-decision` endpoints. Set `JEV_MODEL` to the identifier accepted by your service; the examples use the generic identifier `jev`:

```bash
export JEV_MODEL=jev
```

If the service requires authentication, set `JEV_API_KEY` in the evaluation process.

Model weights, the JEV inference runtime, RoboTwin assets, and historical GIFs and traces are external to this repository. For RoboTwin, install its dependencies and assets according to the upstream instructions, then install this project in that Python environment with `python -m pip install -e .` from the project root.

## Run an evaluation

Run commands from the repository root. Replace the example endpoint URLs with your deployed services. Always select an observation level explicitly; the default configuration uses privileged state.

### MetaWorld: text or vision

Run a text episode using L2 geometric and interaction state:

```bash
MUJOCO_GL=egl jev-robo-eval \
  --task reach-v3 --mode text --privilege-level 2 \
  --sensor-policy direct --action-space primitive \
  --action-granularity fixed \
  --jev-url http://127.0.0.1:8186/v1/decision \
  --seed 2 --task-index 0 --action-repeat 3 --max-decisions 120
```

Run a vision episode using L0 observations, with JEV selecting a task-family stage and then a primitive direction and movement amplitude:

```bash
MUJOCO_GL=egl jev-robo-eval \
  --task shelf-place-v3 --mode vision --privilege-level 0 \
  --sensor-policy staged --action-space primitive --proprio-projection \
  --action-granularity adaptive \
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
  --action-granularity adaptive \
  --jev-urls http://127.0.0.1:8186/v1/decision \
             http://127.0.0.1:8191/v1/decision \
  --seeds 2 --episodes 5 --image-size 384 \
  --move-scale 0.5 --action-repeat 3 --max-decisions 170 \
  --output-dir runs/metaworld-L0-staged-adaptive
```

One endpoint is sufficient; each endpoint handles its assigned episodes sequentially, while different endpoints run in parallel. Replace `--benchmark` with `--tasks shelf-place-v3 bin-picking-v3 assembly-v3` to select tasks explicitly. Use a new output directory for each experiment.

Both environment benchmarks default to `primitive` when `--action-space` is omitted. Select `--action-space metaworld_atomic` explicitly for the separate MetaWorld bounded-intent controller; L0 atomic evaluation also requires `--proprio-projection`.

L0 observations hide MetaWorld virtual sites. In `reach-v3`, `push-v3`, and `pick-place-v3`, this removes the only target marker, leaving no physical destination visible in RGB. The upstream `plate-slide-side-v3` environment can also place its randomized success target slightly away from the visible goal fixture. Report these observation limitations when interpreting L0 results; successful environment initialization does not resolve them.

### RoboTwin: parallel benchmark

Run this command in the Python environment containing RoboTwin and this project. Set the checkout path explicitly and use the upstream `demo_clean` configuration:

```bash
python scripts/evaluate_parallel.py \
  --benchmark --environment robotwin \
  --robotwin-root /path/to/RoboTwin --robotwin-config demo_clean \
  --active-arm left --robotwin-initial-pose topdown \
  --mode vision --privilege-level 0 --sensor-policy staged \
  --action-space primitive --proprio-projection \
  --action-granularity adaptive \
  --jev-urls http://127.0.0.1:8186/v1/decision \
             http://127.0.0.1:8191/v1/decision \
  --seeds 2 --episodes 5 --image-size 384 \
  --move-scale 0.5 --action-repeat 1 --max-decisions 80 \
  --output-dir runs/robotwin-L0-staged-adaptive
```

`topdown` initializes the active arm at a fixed pose using robot geometry and constants; `home` retains the upstream initial pose. RoboTwin preserves the head camera's aspect ratio, so `--image-size 384` produces 384 × 288 images. `--proprio-projection` derives screen directions from camera calibration and the robot's own pose.

For multiple simulator GPUs, add `--simulator-gpus 0 1`, supplying one GPU index per endpoint.

## Observation levels and control modes

Observation access and control mode are separate experiment choices. Policy inputs use explicit field allowlists; full evaluation state is retained for logging and scoring.

| Condition | Available information |
| --- | --- |
| **L0** | RGB, robot proprioception, action history, and optional camera-derived action projections; requires vision mode |
| **L1** | L0 sensor information plus object, target, and scene positions |
| **L2** | L1 plus available task-specific interaction and progress facts such as contact, lift, and relative distances |
| **L3** | For tasks with an implemented guide: oracle stages, waypoints, and desired gripper state generated by handwritten rules using simulator truth |
| **L3-P** | `--privilege-level 3 --plan-only`: retains oracle plan/waypoint information, proprioception, and RGB while removing additional scene truth |

L1–L3 can omit RGB when a compatible text policy branch is used; the quickstart demonstrates MetaWorld L2 text. L0 hides MetaWorld virtual sites and rejects oracle waypoint annotations. L3-P remains oracle-assisted because its plan generator still reads simulator truth.

| Control mode | Selection | Behavior |
| --- | --- | --- |
| Direct primitives | `--action-space primitive --sensor-policy direct` | JEV chooses one primitive action from the current observation. |
| JEV stage/action control | `--action-space primitive --sensor-policy staged`, L0–L2 vision | JEV infers a stage from a constrained set, then chooses an action. `--action-granularity` selects the stage/candidate protocol and whether movement amplitude is a model choice. Both requests are recorded. |
| MetaWorld bounded intents | `--action-space metaworld_atomic` | JEV chooses an intent from rule-constrained candidates; an executor maps it to at most one environment step. L0 uses calibrated visual directions; L1/L2 can use privileged position coordinates. |
| Oracle waypoint assistance | `--action-space primitive --privilege-level 3` | Handwritten rules supply the stage and waypoint; JEV chooses the next primitive action. `--plan-only` selects L3-P. |

The nine primitives are `x_pos`, `x_neg`, `y_pos`, `y_neg`, `z_pos`, `z_neg`, `grip_open`, `grip_close`, and `hold`. Gripper commands persist during later moves. `--move-scale` and `--action-repeat` set the execution budget of each primitive; one environment step can include several simulator steps.

Bounded intents include alignment, height adjustment, lifting, waypoint tracking, and gripper operations. Candidate eligibility and coordinate-based execution add controller assistance, so their success rates measure the combined policy and controller. They are reported separately from JEV stage selection and oracle waypoint guidance.

### Action granularity

`--action-granularity` defaults to `fixed`. The other two choices require all of `--mode vision --sensor-policy staged --action-space primitive` and observation level 0, 1, or 2. Select the level explicitly when defining an experiment. They support both registered environments.

| Granularity | Stage/candidate protocol | Movement amplitude |
| --- | --- | --- |
| `fixed` | Existing direct or staged primitive policy; staged mode uses the original phase vocabulary | Every translation uses the configured amplitude |
| `phase-fixed` | JEV selects a task-family stage, then a signed primitive or gripper command | Every translation uses 1.0 × the configured amplitude |
| `adaptive` | The same task-family stage protocol, with joint direction/size candidates | JEV chooses fine (0.25 ×), normal (0.5 ×), or coarse (1.0 ×) for every signed translation |

The task-family stages distinguish pressing, horizontal pushing, handle pulling/sliding, reaching, and object transfer/insertion. Stage criteria describe the current physical evidence. JEV can reconsider every stage on each decision; an open gripper command excludes stages that require carrying an object. Compact stage observations separate current physical state from the task and movement instructions used for action selection. Both requests use the current RGB image.

At L1/L2, the controller distinguishes a current contact point from a final destination and describes the directional error computed from the already permitted coordinates. These arithmetic hints are marked `derived_from_permitted_pose`: they add representation assistance, not additional simulator measurements. They do not automatically choose a stage, direction, or amplitude. L0 receives no such target coordinates or computed target errors. Every level may receive the configured nominal movement distance, a robot control calibration; actual movement can differ because of contact, clipping, and tracking. The selected hierarchy protocol version is retained in manifests, traces, and the dashboard.

Adaptive scaling changes translation amplitude; it leaves `--action-repeat` unchanged and executes one primitive per decision. Fine motion is not a multi-step macro or a separate autonomous controller. Phase vocabularies and eligibility rules still provide assistance and belong in the experiment description.

To compare granularity, rerun the same task/seed pairs and observation level with `fixed`, `phase-fixed`, and `adaptive`, using separate output directories. Compare `phase-fixed` with `adaptive` to assess model-selected motion size under the shared task-family stage protocol; comparing either against `fixed` also changes that protocol. Keep action repeat, move scale, decision budget, sampling, and recovery settings matched. A staged decision makes a phase request and an action request, so also compare recorded request counts, latency, and token usage when available. Joint direction/size candidates also change the classification distribution: a performance difference is evidence about this controller design, not an isolated measurement of movement precision. Keep protocol versions separate when comparing results.

## Sampling, recovery, and evidence

`--episodes 5` resets each task/seed pair five times. The default `--action-selection argmax` can produce identical trajectories. To sample from JEV's returned action probabilities, add:

```bash
--action-selection sample --policy-seed 100 --sampling-temperature 1.0
```

The five policy seeds are then 100–104, independently of the environment seed. Stage selection remains argmax. Five rollouts of one scene are repeated trials of that scene; an “at least one success in five” statistic uses five attempts and must be reported with that budget.

Optional `--stuck-recovery jitter` detects stalls from robot proprioception and action history, then inserts bounded random translations. It uses the same decision budget and records the proposed and executed actions plus each intervention. The default is `none`. Recovery-assisted results and oracle-assisted results have separate labels.

Batch evaluation saves `summary.json`, per-episode traces, source snapshots, and success/failure GIFs. Parallel batches also maintain `matrix.json` with queued, running, completed, and error states. Vision batches and RoboTwin record actual observed frames, including partial error episodes when frames are available. MetaWorld text batches and the single-episode CLI export GIFs by replaying actions and checking the numerical outcomes.

Traces retain model requests, probabilities, selected phases/candidates, requested and executed action scales, execution details, and image/token audit metadata when supplied by the service. Replay applies the recorded action scale. API keys are not written to these records. Success comes from the upstream simulator's task criterion. RoboTwin's stapler task primarily checks contact at a designated point; success does not establish completion of a physical stapling mechanism.

## Replay dashboard

```bash
python scripts/serve_dashboard.py --host 127.0.0.1 --port 8787
```

Open [http://127.0.0.1:8787/](http://127.0.0.1:8787/). The dashboard refreshes every five seconds and filters by environment, task, seed, experiment, observation level, control mode, action granularity, sampling method, recovery setting, and outcome. Successes, task failures, execution errors, and pending episodes remain distinguishable; execution errors are excluded from the valid success-rate denominator.

Control filters distinguish JEV stage/action control, bounded intents, direct primitives, and oracle assistance. For example, [JEV stage/action runs](http://127.0.0.1:8787/?control-mode=hierarchical_staged) and [atomic runs](http://127.0.0.1:8787/?control-mode=atomic) have separate views. The control-mode selection is stored in the URL; automatic refresh also preserves the other active filters. The current dashboard interface and its [detailed documentation](dashboard/README.md) are in Chinese.

The server exposes dashboard assets, aggregate results, and GIF/PNG replays; raw traces and configuration files are not served. Each experiment and configuration retains its own statistics.

## Earlier subset performance

The following historical evaluation used the **JEV** model in vision-plus-text mode on the **earlier subset of 9 MetaWorld and 3 RoboTwin tasks**. Each task was evaluated once with a single environment seed and argmax action selection. These are not results for the current 16/6 task sets, `phase-fixed`, or `adaptive`. Results are separated by environment, observation level, and controller.

| Model | Environment | Observation level | Controller | Successes / episodes | Success rate |
| --- | --- | --- | --- | ---: | ---: |
| JEV | MetaWorld | L0 | Bounded intents / visual directions | 0 / 9 | 0.0% |
| JEV | MetaWorld | L1 | Bounded intents with position truth | 7 / 9 | 77.8% |
| JEV | MetaWorld | L2 | Bounded intents with interaction truth | 8 / 9 | 88.9% |
| JEV | RoboTwin | L0 | JEV stage selection + primitives | 0 / 3 | 0.0% |
| JEV | RoboTwin | L1 | JEV stage selection + primitives | 0 / 3 | 0.0% |
| JEV | RoboTwin | L2 | JEV stage selection + primitives | 0 / 3 | 0.0% |
| JEV | RoboTwin | L3 | Oracle waypoints + primitives | 1 / 3 | 33.3% |

These scores cover the tasks labeled **original nine** and **original three** in the task registry list above. They are development results from a single seed, with some settings tuned on these scenes; they do not estimate performance across either simulator's full task distribution or the expanded benchmark. MetaWorld bounded intents include rule-constrained candidates and coordinate-based execution. RoboTwin L3 uses oracle guidance. These assistance conditions are part of the evaluated system, and the scores should be interpreted accordingly. Environment smoke checks are not episodes in this success-rate table.

The table counts completed episodes with valid outcomes; execution errors are excluded. Raw experiment records, per-episode results, audit reports, deployment notes, traces, and GIFs are private and are not distributed with this repository. The framework lets users generate their own records and replays locally.

## Development

Install the test dependencies and run the suite in an environment with working EGL rendering:

```bash
python -m pip install -e '.[metaworld,test]'
MUJOCO_GL=egl python -m pytest tests dashboard/test_dashboard.py -q
node --check dashboard/dashboard.js
```

Node.js is only needed for the JavaScript syntax check. Tests cover observation boundaries, task registration, hierarchy candidates and action scales, action selection, intents, recovery, recording, batch configuration, and dashboard behavior. Some tests render MetaWorld scenes; the suite does not replace end-to-end experiments with a live JEV service and installed RoboTwin assets.

The shared interfaces live in [core.py](src/jev_robo_eval/core.py). An environment implements `reset(seed=...)`, `step(Action, scale=1.0)`, and `close()`; a policy implements `decide(observation, task)` and returns a primitive with its amplitude scale. The existing [MetaWorld adapter](src/jev_robo_eval/metaworld_env.py), [RoboTwin adapter](src/jev_robo_eval/robotwin_env.py), [JEV client](src/jev_robo_eval/jev.py), and [runner](src/jev_robo_eval/runner.py) provide the integration pattern.

## License

Project code is available under the [MIT License](LICENSE). External decision services, MetaWorld, MuJoCo, RoboTwin, and their models and assets retain their respective licenses.
