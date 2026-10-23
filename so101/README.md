# SO-101 + LeRobot

This directory is the LeRobot environment for the SO-101 leader/follower pair. It has its own `uv` project. ElRobot uses [NormaCore Station](../elrobot/README.md) and does not share this environment.

Teach the follower arm a pizza task in this order:

```text
Install → find ports → (one-time) motor IDs → calibrate → teleoperate → record → train → rollout
```

Official references:

- LeRobot install: https://huggingface.co/docs/lerobot/installation
- SO-101 hardware: https://huggingface.co/docs/lerobot/so101
- Imitation learning on real robots: https://huggingface.co/docs/lerobot/il_robots
- SmolVLA: https://huggingface.co/docs/lerobot/smolvla
- Cloud fine-tune pattern: [lerobot-finetune-job](https://github.com/mnrozhkov/serverless-ai-cookbook/tree/clawbio/robotics/lerobot-finetune-job) and [smolva-ft-norma-core](https://github.com/mnrozhkov/serverless-ai-cookbook/tree/clawbio/robotics/smolva-ft-norma-core)

Robot control and data collection run in a local `uv` venv (MPS on Apple Silicon). Real training runs in Docker / Nebius GPU jobs (CUDA).

---

## 0. What you need

| Item | Notes |
| --- | --- |
| SO-101 follower + leader | Powered, daisy-chained, matching voltage PSU |
| 1–2 USB webcams | Fixed mounts, 640×480 @ 30 FPS. Front is required; overhead (`top`) is optional |
| Hugging Face account **(yours)** | `hf auth login`; set `HF_USER` in the repo-root `.env` |
| Weights & Biases **(yours)** | `WANDB_API_KEY` + `WANDB_ENTITY`; optional shared project `robotpizzacup` |
| `uv` | Python package runner for this directory |
| `ffmpeg` | `brew install ffmpeg` on macOS |
| Docker (optional) | Local CPU smoke test of the cloud image |
| Nebius CLI (optional) | GPU fine-tune |

---

## 1. Install (local uv environment)

From **this directory** (`so101`):

```bash
cd so101
uv python install 3.12
uv sync
hf auth login
```

This pulls LeRobot **0.6.1** with:

| Extra | Why |
| --- | --- |
| `core_scripts` | `lerobot-calibrate`, `lerobot-record`, `lerobot-replay`, `lerobot-teleoperate` |
| `training` | `lerobot-train`, W&B, datasets |
| `feetech` | SO-101 motors |
| `smolvla` | headline VLA |
| `diffusion` | second visuomotor baseline |
| `pi` | π0 / π0.5 deps |
| `pusht` | tiny public dataset for a no-robot smoke test |

ACT has no extra — it ships in the core package.

Check:

```bash
uv run python -c "import torch, lerobot; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
uv run lerobot-calibrate --help
uv run lerobot-train --help
```

On Apple Silicon, `mps` should be `True`. CUDA is for the cloud image.

Copy station + secrets templates. Secrets stay at the **repo root**. Ports stay here:

```bash
cp station.env.example station.env
cp ../.env.example ../.env
```

| File | Purpose |
| --- | --- |
| `so101/station.env` | USB ports, cameras, `TASK`, episode timing |
| repo-root `.env` | `HF_USER`, `DATASET_NAME`, `HF_TOKEN`, `WANDB_*`, `FILESYSTEM_ID`, `IMAGE` / `PARENT_ID` |

Scripts load `../.env`, then `station.env`.

Full train → deploy runbook: **[Train and deploy](docs/train-and-deploy.md)**.

---

## 2. Find USB ports and set motor IDs (once per arm)

Connect **one** MotorBus USB cable to the computer (and power the arm), then from `so101`:

```bash
bash scripts/list_ports.sh
```

On macOS the MotorBus is `/dev/tty.usbmodem...`. Plug one arm, copy the path into `FOLLOWER_PORT` or `LEADER_PORT` in `station.env`, then swap cables and repeat.

`lerobot-find-port` is optional. It only works if you unplug the **USB cable from the computer** (not the PSU) and wait ~2 s before Enter. If it errors with `No difference was found`, ignore it and use `list_ports.sh`.

**New motors only** (default id `1` on every servo). Connect **one motor at a time** as instructed:

```bash
set -a && source station.env && set +a
uv run lerobot-setup-motors --robot.type=so101_follower --robot.port="$FOLLOWER_PORT"
uv run lerobot-setup-motors --teleop.type=so101_leader --teleop.port="$LEADER_PORT"
```

Skip this if the kit was already ID'd. Pre-assembled event arms already have motor IDs.

Set `CAMERA_INDEX` (front) and optionally `CAMERA_INDEX_TOP` (overhead) in `station.env` — see [Camera setup](#camera-setup) (next section).

---

## Camera setup

Hackathon V1 scoring uses a **fixed front RGB camera**. You can also record a second **overhead (`top`)** stream at the same time; both become part of the dataset schema.

| Setting | Value |
| --- | --- |
| Resolution | 640 × 480 minimum |
| FPS | 30 |
| Mount | fixed, same viewpoint on every station |
| Front camera | required (`observation.images.front`) |
| Overhead camera | optional (`observation.images.top`) |
| Wrist camera | not part of V1 scoring |

### Find camera indexes

```bash
uv run python scripts/list_cameras.py
```

This prints every OpenCV index that opens and writes a JPEG to `/tmp/lerobot-cameras/index-N.jpg`. Open those stills to tell FaceTime vs workspace vs overhead.

On macOS, OpenCV indexes are **0 … N−1** for the N devices in `system_profiler SPCameraDataType`. There is no index `2` until a third camera actually appears. ffmpeg’s `[2] Capture screen 0` is a screen grab, not a webcam.

Index `0` is often the built-in FaceTime camera, but it can swap with a USB cam. Always check the preview JPEGs. Prefer USB webcams for `front` and `top`; never record FaceTime.

Set indexes from the preview JPEGs, not by guessing the next integer:

```bash
CAMERA_INDEX=0          # USB workspace (example)
CAMERA_INDEX_TOP=       # empty until a second USB camera appears in system_profiler
```

Leave `CAMERA_INDEX_TOP` empty for a single front camera.

Quit Zoom, Photo Booth, and Chrome before listing or recording — they hold the webcam exclusive.

### Framing checklist

You should be able to do the task **from the camera images alone**:

**Front**

- pizza disk fully in frame
- topping pickup zone fully in frame
- gripper visible throughout the motion

**Top** (if used)

- full table / pizza / pickup zone from above
- arm does not occlude the pizza for the whole place motion

Shared:

- no motion blur at 30 FPS
- lighting is diffuse and stable (no moving shadows)
- do not move either camera after recording starts

### LeRobot camera dict

`scripts/record.sh`, `teleoperate.sh`, and `rollout.sh` expand `station.env` into:

```text
{ front: {type: opencv, index_or_path: <CAMERA_INDEX>, width: 640, height: 480, fps: 30},
  top:   {type: opencv, index_or_path: <CAMERA_INDEX_TOP>, width: 640, height: 480, fps: 30} }
```

Camera **names** (`front`, `top`) are part of the dataset schema. Record, train, and rollout must use the same keys.

- Adding `top` to an existing front-only dataset will fail — bump `DATASET_NAME` in `.env` (e.g. `pizza_hack_v2`) and start a new dataset.
- A policy trained on `{front, top}` must see **both** cameras at rollout.

Confirm both feeds in Rerun before recording:

```bash
bash scripts/teleoperate.sh
```

---

## 3. Calibrate

The `id` is the calibration key. Reuse it everywhere.

LeRobot stores SO-101 calibration under `~/.cache/huggingface/lerobot/calibration/`, keyed by `--robot.id` / `--teleop.id`.

Reuse the same ids (`FOLLOWER_ID`, `LEADER_ID` in `station.env`) for:

- `lerobot-calibrate`
- `lerobot-teleoperate`
- `lerobot-record`
- `lerobot-replay`
- `lerobot-rollout`

If you change the id, you are starting a new robot identity and must recalibrate.

```bash
bash scripts/calibrate.sh follower
bash scripts/calibrate.sh leader
```

**Procedure**

1. Power both arms. Confirm every motor LED is a steady red from gripper to base.
2. Source `station.env`.
3. Run `bash scripts/calibrate.sh follower`, then `bash scripts/calibrate.sh leader`.
4. Move the arm to the mid-range pose, press Enter, then sweep **each joint through its full range**.

If teleop later looks skewed, recalibrate — do not keep recording.

Hardware faults: [troubleshooting.md](docs/troubleshooting.md).

---

## 4. Teleoperate (no recording)

```bash
bash scripts/teleoperate.sh
```

Confirm:

- follower tracks leader smoothly
- gripper opens and closes
- Rerun window shows the **workspace** camera, not FaceTime (and the overhead view if `CAMERA_INDEX_TOP` is set)
- you can complete the task looking only at the camera view(s)

Practice 5–10 demos before recording.

---

## 5. Collect data (leader / follower)

Full guide: **[Data collection](docs/data-collection.md)** (format, storage, H.264 video, LeRobot doc links). ElRobot: [ElRobot data collection](../elrobot/docs/data-collection.md).

Log in if you later want to push datasets (`hf upload`). Recording itself stays local:

```bash
hf auth login
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
```

Put `HF_USER` and `DATASET_NAME` in the repo-root `.env`. Put `TASK` in `station.env`. The task sentence is part of the dataset — **use the same words at train and rollout**.

```bash
bash scripts/record.sh
```

`record.sh` records **H.264** MP4s (QuickTime-friendly on Mac) with streaming encoding. See [streaming video encoding](https://huggingface.co/docs/lerobot/streaming_video_encoding).

Keys while recording (terminal must stay focused on macOS without Accessibility permission):

| Key | Action |
| --- | --- |
| `n` or `→` | save episode, go to next |
| `r` or `←` | redo this episode |
| `q` or `Esc` | finish session, encode locally (no Hub upload) |

Press **`n` after each demo you want to keep** — quitting with `q` mid-episode does not save it.

Defaults in `station.env`: 50 episodes × 30 s + 10 s reset. That is the SmolVLA starting point.

**Data quality beats model choice.** First 50 episodes: one object, fixed pizza pose, fixed camera, one operator, deliberate grasps.

After recording, `record.sh` prints a Rerun command. Preview the saved episodes (video + joints):

```bash
uv run rerun ~/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}_<timestamp>
# or: bash scripts/viz_dataset.sh
```

Live Rerun during record does **not** play video (images are logged static). Use the commands above. Hub visualizer: https://huggingface.co/spaces/lerobot/visualize_dataset

Replay episode 0 on the follower:

```bash
bash scripts/replay.sh 0
```

Resume more episodes later with `--resume=true` and `--dataset.num_episodes` set to the **additional** count (see [il_robots — checkpointing](https://huggingface.co/docs/lerobot/il_robots#2-checkpointing-and-resuming)).

---

## 6. Train

Local = **20-step `lerobot-train`**. GPU = pull the published image + `run_serverless.sh`. Do not train 30k steps on a Mac.

### 6a. Local smoke test (no robot, public dataset)

```bash
bash training/scripts/train_local.sh act lerobot/pusht 20
```

Or Docker CPU (amd64 — same image family as cloud):

```bash
bash training/scripts/run_docker.sh --rebuild act lerobot/pusht 20
```

`Using device: cpu` in Docker on macOS is expected.

### 6b. Pizza YAML on the laptop (still a smoke)

Default dataset: **`mnrozhkov/pizza_hack_v1_fake_ingredients`**. Use **your** Hub dataset via `DATASET_REPO_ID` / `HF_USER` after you record.

```bash
bash training/scripts/train_pizza.sh act 20
```

Or:

```bash
uv run lerobot-train \
  --config_path=training/policy/act/config.yaml \
  --policy.device=mps \
  --steps=20 \
  --batch_size=2 \
  --policy.push_to_hub=false
```

W&B uses **your** `WANDB_ENTITY`; shared project name `robotpizzacup`. Chart `train/loss` under Charts.

### 6c. Cloud GPU (the real policy)

See **[Train in the cloud](docs/train-and-deploy.md#cloud-training)**. `PARENT_ID` / `IMAGE` in `.env.example` are organizer defaults — switch them if you use your own Nebius project.

```bash
bash training/scripts/run_serverless.sh --debug act     # 500 steps, 1h, CUDA + Hub + FS
bash training/scripts/run_serverless.sh act 30000       # ACT
bash training/scripts/run_serverless.sh smolvla 20000   # SmolVLA
```

Vision-encoder flags for SmolVLA are already in `training/policy/smolvla/config.yaml`.

| Profile | Policy | Batch | Steps | Where | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| Install check | ACT | 2 | 20 | local MPS / Docker CPU | throwaway local |
| Cloud canary | ACT | 8 | 500 | 1× L40S `--debug` | shared FS |
| First robot policy | ACT | 8 | 30k–80k | 1× L40S | FS / Hub → rollout |
| Headline VLA | SmolVLA | 8 | 20k | 1× L40S | same |
| Optional | Diffusion | 8 | 20k+ | 1× L40S | same |

---

## 7. Deploy on the arm

Camera keys, resolution, `FOLLOWER_ID`, and **task sentence** must match the dataset. Details: [Deploy on the arm](docs/train-and-deploy.md#deploy-on-arm).

```bash
# Local training output (smoke)
bash scripts/rollout.sh \
  outputs/train/<run>/checkpoints/last/pretrained_model

# After cloud job — FS mount or Hub
bash training/scripts/download_checkpoint.sh <job-name>
bash scripts/rollout.sh lerobot-outputs/<job-name>/checkpoints/last/pretrained_model
```

Record 10 eval episodes (policy drives the follower, no leader) — see [Train and deploy](docs/train-and-deploy.md#record-eval-episodes). Scoring: [Evaluation](docs/train-and-deploy.md#evaluation).

If ACT is below ~70% on the **same** scene you trained on, collect more consistent demos. Do not jump to π0 first.

---

## Policy cheat-sheet

| Policy | Extra | VRAM (indicative) | Use |
| --- | --- | --- | --- |
| ACT | core | ~1 GB | First success, laptops, single task |
| Diffusion | `diffusion` | ~5 GB | Multimodal actions |
| SmolVLA | `smolvla` | ~4 GB+ | Language / multi-task, hackathon headline |
| π0 / π0.5 | `pi` | ~16 GB | After ACT works, 24 GB+ GPU |

---

## Directory map

```text
station.env.example   USB ports, cameras, TASK
scripts/               calibrate, teleop, record, replay, rollout
training/              policy YAML, scripts/, docker/ (CUDA images)
docs/                  data collection, train/deploy (+ evaluation), troubleshooting
```
