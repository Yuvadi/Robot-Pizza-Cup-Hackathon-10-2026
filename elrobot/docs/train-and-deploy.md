# ElRobot — Train, deploy, and evaluate

There are two dataset formats: converted **LeRobot v2.1 datasets** use the eight-joint workflow below; raw **Station Parquet** uses the legacy `smolvla_py` workflow later in this document. Do not interchange their trainers or checkpoints.

**Prerequisites:** CUDA GPU and a compatible dataset. Live follower tests additionally require Station running and calibrated buses. Raw Station exports come from [data collection](data-collection.md). Setup: [ElRobot README](../README.md).

Run commands from `elrobot` unless a step says `cd norma-core/...`.

## Contents

- [Fine-tune (SmolVLA)](#fine-tune)
- [Eight-joint Hugging Face dataset workflow](#eight-joint-hub-dataset)
- [Deploy on the arm](#deploy-on-arm)
- [Evaluation](#evaluation)
- [Limits and troubleshooting](#limits)

---

<a id="eight-joint-hub-dataset"></a>

## Eight-joint Hugging Face dataset workflow (WSL)

`PeteVLA/cheese2` is LeRobot **v2.1**, not raw Station Parquet. Its metadata specifies
8 episodes, 3,193 frames at 10 FPS, eight normalized state/action values, and
two 224x224 video cameras (`camera_0`, `camera_1`). The recorded task is exactly
**`spreading cheese on pizza`**, not `place cheese on the pizza`.

Use the isolated [training environment](../training/pyproject.toml), not the
SO-101 environment or upstream `smolvla_py`. It pins LeRobot 0.3.3 for v2.1
compatibility and PyTorch 2.7.1/CUDA 12.8 for the RTX 5090. Do not upgrade
LeRobot independently: later dataset/processor APIs are different.

### Environment and preflight

Run these commands **inside WSL**, from `elrobot/training`:

```bash
# First-time system dependencies (evdev needs the Linux input headers).
sudo apt-get update
sudo apt-get install -y build-essential linux-libc-dev ffmpeg

# Install uv if missing, then create the dedicated Python 3.12 environment.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
uv run python train.py --check-only
```

Preflight downloads the dataset, decodes its first pair of camera frames,
checks dimensions/timing, and executes a CUDA matrix multiplication. It does not
start training or connect to the robot. Model weights download separately on
the first training run. No native Windows Python or Conda activation is needed.

### Fine-tune

```bash
uv run python train.py --steps 5000 --batch-size 8 --output outputs/cheese-smolvla
```

The wrapper uses `lerobot/smolvla_base`, infers eight-joint features/statistics
from the dataset, and disables automatic Hub uploads and W&B logging. If memory
is insufficient, retry with `--batch-size 4` and a new output path. Existing output
directories are refused. Checkpoints include policy normalization buffers:

```text
outputs/cheese-smolvla/checkpoints/last/pretrained_model/
```

Eight demonstrations are a small smoke-test dataset, not evidence of robust
autonomous performance. Inspect the videos and hold out complete demonstrations
when comparing models; the command above trains on all eight and does not report
a held-out success score.

The recorder's new collection uses 10 FPS / 224x224 after synchronization issues
with the experimental 15 FPS setup. Preflight
validates 1..30 FPS and decoded RGB shapes against dataset metadata rather than
requiring 10 FPS / 224x224. Policy creation derives input features from that
metadata. Keep the full workflow consistent; see the
[new collection profile and quality checklist](episode-recorder.md#new-place-toppings-on-pizza-collection-10-fps--224x224).
Models trained on this collection should use `follow.py --control-fps 10`.
The default remains 10 for existing models. The rate is an upper bound, not a
real-time guarantee: inference, telemetry and the per-command pause reduce
achieved cadence. Increasing this setting does not speed up a 10 FPS-trained
trajectory correctly. Live images must match the checkpoint dimensions exactly.

### Corrected training loss and existing checkpoints

All runs launched through `train.py` use a process-local correction for the pinned
LeRobot 0.3.3 SmolVLA loss. The upstream implementation reads `actions_id_pad`
instead of the dataset's `action_is_pad` and averages over all 32 padded action
dimensions. The correction averages only real joint dimensions and valid
within-episode steps. Fully padded batches contribute zero loss and gradient.
Installed packages, inference, calibration, and existing checkpoints are unchanged.
Running `python -m lerobot.scripts.train` directly bypasses this correction.

The sibling `<output>.training.json` records `imitation_loss` version
`valid-action-mean-v1`. Loss values are not directly comparable with earlier runs:
excluding 24 padded dimensions and dividing by valid targets changes their scale.
Old checkpoints are not repaired automatically. Use a new output directory and
prefer a fresh fine-tune from `lerobot/smolvla_base` for a controlled comparison.
Keep the export, seed, batch size, and step budget fixed; evaluate on whole
held-out episodes before supervised hardware tests. The wrapper disables robot
evaluation (`eval_freq=0`); its training loss is not a held-out success metric.

### Experimental range-penalized fine-tuning

For an existing Station-normalized checkpoint, `train.py` accepts `--policy-path`
and the opt-in `--range-penalty-weight` (default 0 uses corrected imitation loss only).
For example, from `elrobot/training`, with a verified local toppings export:

```bash
.venv/bin/python train.py \
  --dataset yaditya/place-toppings-on-pizza \
  --dataset-root /absolute/path/to/toppings-export \
  --policy-path ../../toppings-smolvla-policy \
  --output outputs/toppings-range-penalty \
  --steps 5000 --batch-size 8 --range-penalty-weight 10
```

This is additional fine-tuning with a new optimizer, not a resume of an interrupted
optimizer state. The original checkpoint is not overwritten. A sibling
`<output>.training.json` records the input checkpoint, dataset, weight and command.
The opt-in entrypoint changes only the training process, not installed LeRobot
files or checkpoint inference behavior.

The added loss is the weighted mean squared distance outside `[0, 1]` of the
estimated clean action `x_t - t * predicted_velocity`, decoded using the
checkpoint's action mean/std. Only real action dimensions and non-padded episode
steps contribute. The imitation loss uses the padding correction above. This supports Station actions
with MEAN_STD normalization, not ALOHA conversion. Weight 10 is an experimental
starting value, not a validated setting. Loss components are logged every 20
updates.

This penalizes a single-time flow endpoint estimate, not the full multi-step
inference trajectory. It does not guarantee bounded sampled outputs, collision
avoidance, or successful picking. Compare checkpoints offline using fixed seeds
and held-out episodes, checking both range violations and imitation error before
supervised hardware evaluation. All runtime guards remain required.

### Demo cheese dataset (local v3 conversion)

`n4chi/7-robopizzeria-cheese-v1` contains 10 demonstrations / 2,293 frames
at 10 FPS, with embedded top and front/wrist images. It is LeRobot v3 and
cannot be passed directly to the pinned v2.1 trainer. Convert a pinned local
copy without changing the source dataset or upgrading the environment:

```bash
# Inside WSL, from elrobot/training
.venv/bin/python prepare_demo_cheese.py
.venv/bin/python train.py \
  --dataset local/demo-cheese-data \
  --dataset-root local-datasets/demo-cheese-data --check-only
.venv/bin/python train.py \
  --dataset local/demo-cheese-data \
  --dataset-root local-datasets/demo-cheese-data \
  --output outputs/demo-cheese-data --steps 5000 --batch-size 8
```

Conversion preserves episode boundaries, normalized state/action slots, task
labels and the uniform timeline. It maps **top to camera_0** and **front/wrist to
camera_1**, encoding videos with the existing LeRobot writer. Source provenance
and the pinned revision are saved with the converted dataset. Existing
conversion/output directories are never overwritten. Original source timing
fields remain available in the pinned source rather than in training features.

After training, test without connecting to Station:

```bash
.venv/bin/python follow.py \
  --checkpoint outputs/demo-cheese-data/checkpoints/last/pretrained_model \
  --dataset local/demo-cheese-data \
  --dataset-root local-datasets/demo-cheese-data \
  --task "place cheese on pizza" --offline
```

Offline mode compares predictions with recorded actions on `--samples`
evenly spaced frames (default 5). It prints the per-joint error and flags
predictions outside `[0, 1]`, which live mode would refuse. These frames come
from the training data, so a low error shows fit, not generalisation.

The dataset does not identify physical motor IDs for its eight joint slots.
Matching dimensions do not establish compatibility with your follower or its
new calibration. Do not execute the model on hardware without verifying the
joint mapping, calibration, framing and starting scene. This run trains on all
10 episodes; it does not measure held-out task success or certify safe control.

### Original cheese checkpoint offline test

```bash
uv run python follow.py \
  --checkpoint outputs/cheese-smolvla/checkpoints/last/pretrained_model \
  --offline
```

This predicts from evenly spaced recorded observations and prints each prediction
beside its recorded action. It is a wiring check, not a task-success evaluation.

### Live follower dry run

#### ACT checkpoints with external processors

The bridge also recognizes `"type": "act"` checkpoints using the external
MEAN_STD processors supported by `act_checkpoint.py`. It validates processor
ordering, statistics, and supported configuration fields, then loads every model
weight strictly into the installed ACT architecture. It does not modify the
checkpoint or download backbone weights. Unsupported formats fail explicitly.
The same prediction, calibration, freshness, movement and confirmation guards
apply. Rollouts preserve the checkpoint's configured action horizon rather than
truncating it to 10 actions (ACT temporal ensembling retains its native behavior).

ACT is not language-conditioned: `--task pick_and_place_cheese` labels/checks the
requested dataset task but does not teach or select cheese behavior in ACT.
The exception is checkpoints that include `task_vocab.json` (see "Training ACT on
the qual dataset"): there `--task` must be one of the listed tasks and selects
the task one-hot input.
Camera framing, calibration, joint mapping and training behavior must match.
Test without hardware first, quoting the checkpoint path if it contains spaces:

```bash
.venv/bin/python follow.py --checkpoint "../../ACT Model" \
  --dataset yaditya/place-toppings-on-pizza \
  --dataset-root /absolute/path/to/toppings-export \
  --task pick_and_place_cheese --offline --samples 20
```

#### Optional bounded policy rollout

The default `--execute` path still sends only one target. To test a policy over
time, additionally pass `--rollout-seconds 20`. Any explicitly supplied positive,
finite duration is accepted, including `100` or `150`; there is no fixed upper
cap. Omitting the option (or specifying `0`) retains the single-send behavior.
The confirmation phrase includes the requested duration. Longer runs require
continuous supervision and do not add a hardware watchdog. For example,
from `elrobot/training`:

```bash
.venv/bin/python follow.py \
  --checkpoint ../../cheese-smolvla-v2-policy \
  --task "place cheese on pizza" \
  --bus-serial 5B61032837 \
  --motor-ids 1,2,3,4,5,6,7,8 \
  --camera-order 0,1 \
  --max-delta-ticks 1500 --execute --rollout-seconds 20
```

The movement limit defaults to 100 ticks and can be raised explicitly to at most
1500 ticks per command for testing. A larger limit allows substantially larger
physical movements; verify calibration and clear the workspace before execution.

This requires the separate confirmation `ROLLOUT 5B61032837 20`. Verify the
actual mappings and calibration, stop recording, leader mirroring, browser
motor adjustments, and other policies, clear the workspace, and keep a physical
stop accessible. The application does not establish exclusive control.

The rollout preserves the policy action queue between ticks and consumes the
checkpoint's configured `n_action_steps` before replanning with current images.
It no longer truncates that horizon to 10 actions. For example, a 50-action
SmolVLA horizon or a 100-action ACT horizon is retained. This is the checkpoint's
execution horizon, which can be shorter than its predicted `chunk_size`.
ACT temporal ensembling, when enabled, retains its native per-tick prediction.
For queued policies, new observations do not affect predictions until the queue
is exhausted; longer horizons delay visual feedback and can execute larger
later movements. Start with a short supervised trial and keep the physical stop
accessible. Every send still uses fresh telemetry for safety validation.
The bridge sends at most
10 targets per second without catch-up bursts. The duration includes inference
and reads. Before the timed rollout, one no-motion inference warms up the policy;
its prediction and cached actions are discarded. The bridge then fetches fresh
observations for execution. Warm-up time is printed and excluded from the rollout
duration. Warm-up inference runs in a worker thread so Station network reads and
keepalive messages can continue. Cancellation waits for that no-motion worker
before clearing policy state; no warm-up targets are sent. A Station fetch timeout
still aborts rather than retrying or sending from cached observations.
No warm-up action is sent, even if its input becomes stale during
inference. Execution observations must still pass the unchanged two-second
freshness checks before any command is sent.
Every target is checked against input and refreshed telemetry. Invalid/non-finite
predictions, excessive movement, stale input, calibration changes,
and read/send errors abort the run without retrying or clipping.
Station publishes about every 100 ms, so a fetch can return the frame that was
already used. The rollout then polls (20 ms) for up to 1 s for a newer frame and
never acts twice on the same observation; only a 1 s stall aborts with
`Rollout input did not advance`.

To opt into bounded retries for finite but out-of-range predictions, add
`--max-invalid-retries 3` to the rollout command (allowed: 0..20; default: 0,
immediate abort). Each rejection discards the entire eight-joint action and
cached chunk, logs the values, and sends no command. After a rate-limited pause,
the policy replans from a fresh, advancing observation. Three consecutive
rejections allow three retries; a fourth consecutive rejection aborts.
A successfully sent command resets the rejection counter. The original rollout
deadline remains in force, and all other validation failures still abort.
There is no clipping or mixing of joint actions. Previously sent goals remain
active while retrying; a numerically valid prediction is not proof of safe motion.

Each rollout prediction logs its inference duration before checking freshness.
If freshness still fails after warm-up, compare the inference time with the
reported sensor age: slow inference and already-old telemetry are distinct
possibilities. Verify Station sensor updates; do not raise the freshness limit
just to accept an old observation.

Movement-limit aborts print the prediction number, whether a new action chunk
started, inference duration, and a per-motor diagnostic table. `gap` is proposed
target minus observed position along the calibrated arc; `target_change` compares
the proposed target to the previous command from this rollout; `tracking_gap`
compares that previous target to observed position. The previous target is a
commanded value, not proof of motor receipt or exclusive control. These values
help distinguish policy jumps from tracking lag without weakening the limit.

An observed position outside its calibrated arc stops execution even if Station
reports a normalized endpoint of `0` or `1`. The error identifies the zero-based
joint slot, raw position, calibration endpoints, and distance outside the arc.
Do not widen calibration or clamp telemetry merely to suppress this error.
Check the physical pose and calibration through the supported manual workflow
before another policy attempt.

Policy outputs are absolute targets in the dataset's `[0, 1]` calibrated-arc
coordinates, not signed movement deltas or degrees. The checkpoint's mean/std
action normalization is reversed inside `select_action` before the bridge
receives the prediction. Negative returned values therefore fall below the
calibrated minimum; they are not requests to rotate backward. Encoder wraparound
is already supported when the calibrated arc crosses raw encoder zero, but this
does not make the normalized endpoints interchangeable or permit crossing a
mechanical stop. Do not modulo-wrap an out-of-range normalized prediction.

Ctrl+C stops new commands once control returns from any in-flight inference.
There is no independent hardware watchdog or collision detection. The process
does **not** enable/disable torque or retract an already sent goal; timeout,
abort, or exit can leave motors moving toward or holding the last goal.
Use the physical stop for immediate intervention, not Ctrl+C alone.

#### Pizza kitchen web UI (order -> recipe -> supervised steps)

`kitchen.py` serves a local page where an order (0-20 pieces of cheese, mystery
item, and salami per quadrant, or the Margherita / Salami / Half & half presets)
becomes one recipe step per piece: cheese first, then the mystery item, then salami.
Each step is a trained task built from `--task-template`; the default matches
`place cheese in quadrant 1`. For checkpoints trained on
`yaditya/place-toppings-on-pizza-with-quads` use
`--task-template "pick-n-put-{topping} put-in-q{quadrant}"`. The mystery item has
no trained task, so it must sit in the cheese or salami bin; choose which in the
UI and the step reuses that bin's task.

```bash
# Inside WSL, from elrobot/training, with Station running and follower torque on
.venv/bin/python kitchen.py --checkpoint ../../toppings-quads-aditya-policy \
  --task-template "pick-n-put-{topping} put-in-q{quadrant}" \
  --bus-serial 5B61032837 --motor-ids 1,2,3,4,5,6,7,8 --camera-order 1,0 \
  --max-delta-ticks 1500
# open http://localhost:8893/
```

- **Authorization:** each order requires typing `COOK <bus serial>`. STOP cancels
  new commands only; the last goals stay active and torque is never changed.
- **Step completion:** each step runs `follow.rollout` (all safety checks unchanged)
  until the arm has moved more than `--move-ticks` and then held within
  `--still-ticks` for `--settle-seconds`, or `--segment-seconds` elapse. The
  verifier then checks the result; if it fails, the same task runs again.
- **Verifier:** drag a box around the pizza on one camera, map Q1-Q4 to corners
  exactly as the training data was labeled, and save. A step passes when the
  target quadrant's mean pixel change since the step started is at least
  `--verify-change` (default 0.03). SmolVLM2-500M yes/no probabilities before
  and after are logged. Add `--verify-strict` to also require a gain of
  `--verify-gain`. On 224 px crops SmolVLM is weak, and a false "not done"
  makes the robot re-place toppings. Without calibration, steps finish on settle
  and are marked unverified.
- **Inspect now (colour-based, no SmolVLM):** SmolVLM mistook red sauce for
  salami and the dough for cheese, so Inspect compares against a reference
  instead. With **no toppings** on the pizza, press **Bare pizza now → capture
  reference**. Also save both bin boxes while the bins hold toppings, because
  their median colours define "cheese" and "salami". Inspect counts a pixel as a
  topping only if it changed since the bare reference and now matches that
  topping's colour better than what was there before. Sauce and dough are part
  of the reference, so they never count. A quadrant reports a topping when at
  least `--inspect-cover` (default 0.03) of it qualifies. Moving the pizza box
  or lighting changes requires a new reference. Each step's verification also
  logs `topping_cover`. With `--verify-colour X` (default 0, report only), a
  step also needs `X` of the target quadrant to newly look like the topping.
  The mystery item is never colour-checked.
- **Ingredient bins:** set *Draw* to cheese bin or salami bin, drag a box around
  that bin on either camera, and save. The policy does not see these boxes; it
  still reaches for the bins it was trained on. They are checks only: a step
  also needs the bin region to change by `--verify-pick` (default 0.02; 0
  disables). With the bin empty, press **Bin is empty now** to store a
  reference. Steps from that bin then send no commands and wait while the bin
  differs from the reference by less than `--empty-match` (default 0.03), and
  resume once it is refilled. Moving or re-saving a bin box clears its reference.
- **Recovery:** any stop (stale or repeated frames, out-of-range retries
  exhausted, movement limit, Station read/send errors) is logged and the same
  step is retried after 2-10 s back-off, re-warming the policy and reconnecting
  for non-validation errors. This repeats until the order finishes or STOP is
  pressed, so stay at the physical stop while cooking.

#### Free-placement kitchen (policies without quadrant input)

`kitchen_free.py` is the same UI on port 8896 for checkpoints trained on plain
`pick-n-put-cheese` / `pick-n-put-salami` tasks, such as
`toppings-smolvla-multitask-policy`. It shares the pizza and bin calibration with
`kitchen.py`. Run only one kitchen at a time, because both drive the same robot.

```bash
.venv/bin/python kitchen_free.py --checkpoint ../../toppings-smolvla-multitask-policy \
  --bus-serial 5B61032837 --motor-ids 1,2,3,4,5,6,7,8 --camera-order 1,0 \
  --max-delta-ticks 1500
# open http://localhost:8896/
```

The order is still per quadrant, but the policy cannot be steered. After each
piece, the quadrant whose pixels changed most (at least `--verify-change`) is
recorded as where it landed. The step shows whether that matches the requested
quadrant, and the page keeps an on-target/off-target tally. A step is retried
only if nothing landed, or if a bin box shows nothing was picked. A
wrong-quadrant piece is counted and never re-placed. SmolVLM is not loaded in
this mode.

#### Quadrant kitchen for `qual-cheese-salami-final-model` policies

`kitchen_qual.py` (port 8898) is the standard kitchen with the task template
`pick-n-put-{topping} put-in-q{quadrant}`, matching the two labels in
`yaditya/qual-cheese-salami-final-model`. It accepts SmolVLA checkpoints and ACT
checkpoints from `train_act_qual_colab.ipynb`. For ACT, it refuses to start
unless all 8 recipe tasks are in the checkpoint's `task_vocab.json`.

```bash
.venv/bin/python kitchen_qual.py --checkpoint ../../act-qual-policy \
  --bus-serial 5B61032837 --motor-ids 1,2,3,4,5,6,7,8 --camera-order 1,0 \
  --max-delta-ticks 1500
# open http://localhost:8898/
```

#### Training ACT on the qual dataset (Colab)

Open `elrobot/training/train_act_qual_colab.ipynb` in Google Colab with a GPU
runtime and run all cells. No login is needed because the dataset is public. It
pins `lerobot==0.3.3`, matching the local loader and the v2.1 dataset.

ACT has no language input, so the notebook adds a one-hot of the task string
as `observation.environment_state` (an `ENV` feature, which is not normalized).
It writes the order to `task_vocab.json` beside `config.json` and
`model.safetensors`. `act_checkpoint.load_act` (used by `follow.py` and every
kitchen) wraps such checkpoints and builds the one-hot from each batch's
`task`. An unknown task raises an error and sends nothing.

Defaults: chunk 20 actions (2 s at 10 fps), re-plan every 10, batch 8, lr 1e-5,
30k steps, and image colour/sharpness augmentation. A checkpoint is saved every
5k steps. An open-loop cell plots predicted vs recorded joints, with error in %
of joint range, because actions are calibrated positions in [0, 1]. Download
`act-qual-policy.zip` and unzip it into the repo root.

#### Preconditions and single-target preview

Before connecting a policy, verify all of the following:

- Training joints refer to this follower, with the same calibration and ascending
  motor-ID ordering. The dataset labels `joint_0` through `joint_7` do not encode
  anatomical identity. Eight matching dimensions alone are not sufficient.
- The physical cameras and framing match training. Station orders cameras by
  device identity; changing USB attachment can change their order. Verify it
  visually. `--camera-order 0,1` maps Station images 0/1 to training camera_0/1;
  use `1,0` only if their order is reversed.
- Stop leader teleoperation and any other policy process.
- Station runs in the **same WSL instance** with fresh inference frames. The
  bridge rejects stale joint/camera timestamps and does not support remote clocks.
- Pin `st3215-bus` under the `inference/normvla` entry in the **active**
  `norma-core/bin/station.yaml` to the follower's exact serial, replacing `auto`.
  Restart Station after changing it. The frame format contains no bus/motor
  identity, so the bridge relies on this configuration and your explicit mapping.
- Station may stop producing inference frames when all torque is disabled.
  The bridge never enables torque; use the existing supervised Station workflow
  to establish valid telemetry. Do not disable guards to get past missing frames.
  A stale-frame error reports the sample age; an old frame can remain in history
  even while the web UI is connected. Verify that the selected follower is online,
  calibrated, and producing valid telemetry, and that both cameras are streaming.
  Check Station's terminal for inference skip reasons. The Linux bridge compares
  timestamps using Station's `CLOCK_BOOTTIME`; do not increase the age limit to
  accept an old observation.

From `elrobot/training` (replace placeholders; IDs 1..8 are only an example):

```bash
uv run python follow.py \
  --checkpoint outputs/cheese-smolvla/checkpoints/last/pretrained_model \
  --bus-serial FOLLOWER_SERIAL \
  --motor-ids 1,2,3,4,5,6,7,8 \
  --camera-order 0,1
```

Default behavior is **prediction only**, with no robot writes. It rejects wrong
dimensions, non-finite actions, and calibration mismatches. Out-of-range
predictions are reported per motor as `OUT OF RANGE` and blocked for execution,
without clipping or converting them to motor targets. Execution rejects them.
For predictions within [0, 1], it prints
all eight current positions, predicted targets, and calibrated-arc deltas in ticks,
marking movements over 100 ticks as `OVER LIMIT` and blocked for execution.
Over-limit previews never send commands; execution still rejects these movements
(the limit cannot be disabled).
Prediction-only mode refreshes telemetry after inference for its movement-bound
preview, allowing slow initial predictions without accepting stale live telemetry.
Execution additionally requires that the original prediction input is still
within the two-second freshness limit; a slow prediction cannot authorize motion.

Only after verifying the printed predictions, calibration, mappings and clear
workspace, rerun that command with `--execute`. It requires typing
`MOVE FOLLOWER_SERIAL` and sends **one** bounded sync-write, then exits.
It refreshes telemetry and rechecks joint limits immediately before sending.
It does not enable torque, run an autonomous episode, or automatically disable
holding torque on exit. Keep a physical power stop accessible; a model prediction
and these software guards do not guarantee collision avoidance.

Hardware-free safety tests:

```bash
uv run python -m unittest -v test_safety test_follow
```

---

<a id="fine-tune"></a>

## 1. Legacy raw Station Parquet fine-tuning (six-joint default)

Trainer: [`software/ai/smolvla_py`](https://github.com/norma-core/norma-core/tree/v0.1.0-beta.10/software/ai/smolvla_py). It fine-tunes `lerobot/smolvla_base` on Station parquet files. Requires **CUDA**.

The bundled script hardcodes **6** state values and **6** actions (`--motor-ids` defaults to `1,2,3,4,5,6`). ElRobot has **8** servos. A checkpoint from the default settings is SO-101-sized — do not treat it as a full 8-DoF ElRobot policy until motor dimensions match the arm.

```bash
cd norma-core/software/ai/smolvla_py
uv sync
uv run python scripts/train.py \
  --steps 30000 --batch-size 64 \
  --parquets ../../../../data/elrobot-pizza.parquet \
  --output checkpoints/run
```

| Output | Location |
| --- | --- |
| Step checkpoints | `checkpoints/run/step-NNNNNN/` |
| Final | `checkpoints/run/final/` |

Pass several parquet files to `--parquets`. Cloud notes: `norma-core/software/ai/smolvla_py/nebius.md`.

`TASK` in training data must match the sentence in `elrobot.env` and at deploy.

---

<a id="deploy-on-arm"></a>

## 2. Deploy on the arm

There is no separate offline eval harness in this tag — the policy runs against a **live** Station instance.

Same motor limit as training: `run_policy.py` commands motors `1` through `6` unless you change `--motor-ids`.

```bash
cd norma-core/software/ai/smolvla_py
uv run python scripts/run_policy.py \
  --checkpoint checkpoints/run/final \
  --task "place pepperoni on pizza" \
  --server localhost \
  --bus-serial <FOLLOWER_SERIAL>
```

Use the follower bus serial from `elrobot.env`. `--task` must match the parquet / `TASK` in `elrobot.env`.

| Flag | Behavior |
| --- | --- |
| `--auto --max-ticks 200` | Run without pressing space each tick |
| (default interactive) | Space = one tick; `q` or Ctrl+C to stop |

Station must already be running (`bash scripts/run.sh` from `elrobot`).

Optional: `bash scripts/python_env.sh examples/monitor_inference.py` for live ST3215 frame prints (debug only).

---

<a id="evaluation"></a>

## 3. Evaluation

Automated scoring scripts are not in this repo yet.

Until then, use physical success rate on the **follower** with the same scene reset every episode:

1. Reset the pizza scene the same way every time.
2. Deploy with `run_policy.py` (above) for **10** episodes (`--auto` or interactive).
3. Score success / fail by hand against the challenge rubric.
4. Compare to a teleoperated baseline (`bash scripts/teleop.sh`).

If success is far below what you expect, fix **data** (tags, episode length, `TASK`, calibration, camera framing in Station) before changing the model. Remember the **6 vs 8 DoF** caveat when interpreting policy behavior.

---

<a id="limits"></a>

## 4. Limits and troubleshooting

| Issue | Likely cause | Fix |
| --- | --- | --- |
| Policy moves wrong joints | Default `--motor-ids` 1–6 | Align motor ids with your bus layout or accept SO-101-sized action space |
| Arm barely moves | Task / scene mismatch vs training parquet | Match `TASK`, cameras, and object poses |
| `run_policy.py` cannot connect | Station down or wrong serial | `bash scripts/run.sh`; check `FOLLOWER_SERIAL` in `elrobot.env` |
| Training OOM / slow | Laptop without CUDA | Use GPU machine; see `nebius.md` in `smolvla_py` |

Hardware and Station UI: [ElRobot README](../README.md).
