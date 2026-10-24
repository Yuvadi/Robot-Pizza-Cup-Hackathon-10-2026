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

### Experimental range-penalized fine-tuning

For an existing Station-normalized checkpoint, `train.py` accepts `--policy-path`
and the opt-in `--range-penalty-weight` (default 0 preserves the original loss).
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
steps contribute. The imitation loss is unchanged. This supports Station actions
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

The rollout preserves the policy action queue between ticks, consumes at most
10 actions per chunk (then replans with current images), and sends at most
10 targets per second without catch-up bursts. The duration includes inference
and reads. Before the timed rollout, one no-motion inference warms up the policy;
its prediction and cached actions are discarded. The bridge then fetches fresh
observations for execution. Warm-up time is printed and excluded from the rollout
duration. No warm-up action is sent, even if its input becomes stale during
inference. Execution observations must still pass the unchanged two-second
freshness checks before any command is sent.
Every target is checked against input and refreshed telemetry. Invalid/non-finite
predictions, excessive movement, stale or non-advancing input, calibration changes,
and read/send errors abort the run without retrying or clipping.

To opt into bounded retries for finite but out-of-range predictions, add
`--max-invalid-retries 3` to the rollout command (allowed: 0..3; default: 0,
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
