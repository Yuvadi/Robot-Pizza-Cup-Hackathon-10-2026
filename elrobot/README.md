# ElRobot + NormaCore Station

This directory is the ElRobot setup. It does not use the LeRobot `uv` environment in [so101](../so101/README.md).

Teach the follower arm a pizza task in this order:

```text
Install → connect → calibrate → teleoperate → export a dataset → fine-tune → deploy
```

Pinned release: **v0.1.0-beta.10**.

Upstream: [norma-core](https://github.com/norma-core/norma-core) · [releases](https://github.com/norma-core/norma-core/releases) · [dataset export](https://github.com/norma-core/norma-core/blob/v0.1.0-beta.10/docs/datasets/export-parquet/datasets-export.md)

---

## 0. What you need

| Item | Notes |
| --- | --- |
| ElRobot follower + leader | 7 joints + gripper (8 ST3215 servos each). Pre-assembled. |
| Power | **12 V follower, 5 V leader.** Never swap supplies. |
| Laptop | **macOS (Apple Silicon)** or **Linux (x86_64 or ARM64)**. Windows is not supported. |
| USB | Data cables, one controller board per arm |
| `git` | `install.sh` shallow-clones the norma-core tag |
| `uv` | Only for the fine-tune step. Teleop and calibration examples use the Python standard library |

Do not build Station from source. `install.sh` downloads the release binary and clones the matching tag.

---

## 1. Install

From `elrobot`:

```bash
bash scripts/install.sh
```

This fills `./norma-core/` with two things:

1. **norma-core at v0.1.0-beta.10** — a depth-1 clone of the whole repo. Upstream examples import `software.station...`, `target.gen_python...`, and `shared.gremlin_py`, so a checkout of `software/station` alone does not run.
2. **Station server** — release CLI at `./norma-core/bin/station` (skipped if this version is already there).

Hackathon settings stay in this directory, not inside the clone:

```bash
cp elrobot.env.example elrobot.env
```

Put the leader and follower serials in `elrobot.env` after you identify the buses (section 2).

Use `--force` to re-clone and re-download. Flags: `--binary-only`, `--python-only`.

| OS | File (v0.1.0-beta.10) | What it is |
| --- | --- | --- |
| macOS Apple Silicon | `station-macos-arm64.zip` | Command-line archive |
| Linux ARM64 | `station-linux-arm64.tar.gz` | Command-line archive (Raspberry Pi and other ARM64) |
| Linux x86_64 | `station-linux-amd64.tar.gz` | Command-line archive (Intel/AMD) |

### Linux USB access

Station needs your user in the `dialout` group to open servo boards without sudo. `install.sh` prints a reminder if you are not in `dialout`:

```bash
sudo usermod -a -G dialout $USER
```

Log out and back in before you plug in the arms.

Start Station (after `install.sh`):

```bash
bash scripts/run.sh
```

Stop Station:

```bash
bash scripts/run.sh stop
```

Python clients only stop with **Ctrl+C**. They do not shut down the server.

| Flag | Listens on |
| --- | --- |
| `--web` | `0.0.0.0:8889` (browser UI) |
| `--tcp` | `0.0.0.0:8888` (normfs TCP) |

Open http://localhost:8889. You should see the Station web interface.

`run.sh` starts the binary from `norma-core/bin/`, so `station.yaml` and `station_data/` live in that folder, next to the binary. A commented starter is [station.yaml.example](station.yaml.example).

`install.sh` already checks that the teleop and auto-cal **Python examples import** (no Station process required). If that step fails, fix the clone before `teleop.sh` / `calibrate.sh`.

---

## 2. Connect the arms

1. Clamp the leader and follower to the table.
2. Wire each arm's servo cable to its own controller board. Label the boards EL-LEADER and EL-FOLLOWER.
3. Connect each arm's own power supply — **12 V follower, 5 V leader** — and leave it off until the data cables are in.
4. Connect both boards with USB data cables, then power the arms. USB does not power the motors.
5. Start Station and open http://localhost:8889. Each arm is a servo bus named by the controller board serial number. Each bus should show **8 servos**. Unplug one USB cable to see which serial is the leader and which is the follower. Write both serials into `elrobot.env` as `LEADER_SERIAL` and `FOLLOWER_SERIAL`.

### Cameras

ElRobot cameras are UVC devices inside Station (`usb-video` in `norma-core/bin/station.yaml`). They do **not** appear in `lerobot-find-cameras` or SO-101 `station.env` camera indexes.

| Setting | Notes |
| --- | --- |
| Driver | `usb-video` entries in `norma-core/bin/station.yaml` (see [station.yaml.example](station.yaml.example)) |
| `resolution: auto` | Station picks a format the camera offers |
| `1280x720` | Used only if the device lists that mode |
| `resize_target` | Stored frame size — separate from capture resolution |
| `frame-skip` | Thins both history recording and live preview |

**Framing (hackathon)**

- Confirm **wrist** and **side** views in the web UI at http://localhost:8889 before you tag history.
- Keep mounts fixed; do not move cameras mid-session.
- You should be able to judge grasp and place quality from the Station preview alone.

Tweak drivers in `norma-core/bin/station.yaml` only when you understand the upstream schema — copy ideas from [station.yaml.example](station.yaml.example) first. No preview: [troubleshooting](docs/troubleshooting.md#no-camera-stream-in-station).

---

## 3. Calibrate

Click the arm's bus to open **Calibrating Bus: [serial]**. Opening this page resets that arm's calibration.

1. **Move every joint by hand through its full range,** including the gripper. Keep moving until Station has seen each joint move far enough.
2. **Follower (12 V): click Auto Calibrate.** The arm moves each joint to its mechanical stop, one joint at a time. Clear the workspace, keep hands out, and keep one person at the power switch. Click **Stop Calibration** to abort.
3. **Leader (5 V): skip Auto Calibrate.** Station disables it below 7 V and shows a low-voltage warning. That is expected. The ranges from step 1 are the calibration.
4. **Click Save.** The 3D view should follow your moves, and every motor should travel from 0% to 100%. Then click **Done**. Station locks the calibration.
5. If calibration fails: click **Reset**, then repeat, or calibrate by hand only (step 1, then Save).

Use the web UI for the event. The wrapper below runs the upstream SO-101 auto-cal sequence on the **follower** serial from `elrobot.env`. It drives motors to their mechanical stops. Clear the workspace first. Calibrate the 5 V leader by hand (step 3), not with this script.

```bash
bash scripts/calibrate.sh
```

---

## 4. Collect data

Full guide: **[Data collection](docs/data-collection.md)** (tag/export workflow, parquet schema, storage). SO-101 LeRobot datasets: [SO-101 data collection](../so101/docs/data-collection.md).

Station must be running (`bash scripts/run.sh`) with both buses calibrated. Set `TASK` and `DATASET_OUTPUT` in `elrobot.env` before export.

```bash
bash scripts/teleop.sh
# Tag start/end in the web UI, then:
bash scripts/export_dataset.sh <START_FRAME> <END_FRAME>
```

Practice teleop until the follower tracks smoothly (including gripper) before tagging. Default export keeps episodes ≥ **45 s** (`EPISODE_DURATION`); wait **~8 s** between demos. Details: [data collection](docs/data-collection.md).

---

## 5. Train and deploy

Full runbook: **[Train and deploy](docs/train-and-deploy.md)** (SmolVLA fine-tune on CUDA, `run_policy.py`, evaluation).

Quick reference — after you have `${DATASET_OUTPUT}.parquet`:

```bash
cd norma-core/software/ai/smolvla_py
uv sync
uv run python scripts/train.py \
  --steps 30000 --batch-size 64 \
  --parquets ../../../../data/elrobot-pizza.parquet \
  --output checkpoints/run

uv run python scripts/run_policy.py \
  --checkpoint checkpoints/run/final \
  --task "place pepperoni on pizza" \
  --server localhost \
  --bus-serial <FOLLOWER_SERIAL>
```

The bundled trainer defaults to **6** motor dimensions (SO-101). ElRobot has **8** servos — read the limits in [train and deploy](docs/train-and-deploy.md). Scoring: [evaluation](docs/train-and-deploy.md#evaluation).

---

## Directory map

```text
elrobot.env.example         leader/follower serials, TASK, episode timing
scripts/install.sh          shallow-clone into norma-core/ + release binary
scripts/run.sh              start or stop Station
scripts/calibrate.sh        follower auto-cal (upstream example)
scripts/teleop.sh           leader → follower (upstream example)
scripts/export_dataset.sh   history range → parquet + mp4
scripts/python_env.sh       optional PYTHONPATH for elrobot/examples/
docs/                       data collection, train/deploy, troubleshooting
norma-core/                 gitignored checkout (v0.1.0-beta.10); do not edit
norma-core/bin/station      release CLI, station.yaml, station_data/
station.yaml.example        commented driver and camera starter
```

`examples/monitor_inference.py` prints live ST3215 frames. It is optional. The teleop and calibration commands above do not use it:

```bash
bash scripts/python_env.sh examples/monitor_inference.py
```
