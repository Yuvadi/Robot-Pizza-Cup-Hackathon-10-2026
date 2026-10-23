Commands below run from `elrobot`. SO-101 recordings use a different format: [SO-101 data collection](../../so101/docs/data-collection.md).

# ElRobot (NormaCore Station)

Station **always logs** motor and camera data into on-disk history (normfs under `norma-core/bin/station_data/`). You do **not** run `lerobot-record`. For ML, you **tag** a time range in the web UI and **export** a `.parquet` file with `dataset-generator`.

Upstream export walkthrough: [datasets-export.md](https://github.com/norma-core/norma-core/blob/v0.1.0-beta.10/docs/datasets/export-parquet/datasets-export.md).

## Prerequisites

1. `bash scripts/install.sh` — NormaCore **v0.1.0-beta.10** under `norma-core/`
2. `bash scripts/run.sh` — web UI at http://localhost:8889
3. Both ST3215 buses calibrated — [ElRobot README — Calibrate](../README.md#3-calibrate)

Cameras are configured in Station (`usb-video` in `norma-core/bin/station.yaml`). They do **not** use `lerobot-find-cameras` or SO-101 `station.env` indexes. Framing notes: [ElRobot README — Connect](../README.md#2-connect-the-arms).

## Configure `elrobot.env`

```bash
cd elrobot
cp elrobot.env.example elrobot.env
```

| Variable | Purpose |
| --- | --- |
| `LEADER_SERIAL` / `FOLLOWER_SERIAL` | Bus serials from the Station UI (8 servos each) |
| `STATION_HOST` | Where Station runs (`localhost` on the laptop wired to the arms) |
| `TASK` | Task sentence on **every exported frame** — match train/deploy verbatim |
| `EPISODE_DURATION` | Minimum episode length in seconds for export (default **45**) |
| `DATASET_OUTPUT` | Path **prefix**; export writes `${DATASET_OUTPUT}.parquet`. Default `../data/elrobot-pizza` → repo-root `data/` |

## Teleoperate before you tag

Station must be running. Leader mirrors to follower via the upstream teleop example:

```bash
bash scripts/teleop.sh
```

Ctrl+C stops teleop and disables follower torque. Practice smooth demos before tagging history.

Timing for export (defaults):

- Each kept episode should be at least **`EPISODE_DURATION`** seconds (45 by default). Longer demos are kept; shorter ones may be dropped.
- Wait **at least 5 seconds** between episodes; **~8 seconds** reset is comfortable.

## Tag, export, preview

Recording is **tagged slices** of Station history, not a separate recorder.

1. In http://localhost:8889, press **Tag** and note the **start** tag name.
2. Run `bash scripts/teleop.sh` and perform one or more episodes (respect duration and reset gaps).
3. Press **Tag** again and note the **end** tag name.
4. Open **History**, find both tags, and copy their **frame pointers** (start and end).
5. Export:

```bash
bash scripts/export_dataset.sh <START_FRAME> <END_FRAME>
```

The script runs `dataset-generator` (reads the live Station at `STATION_HOST`) then `dataset-mp4` for a QuickTime-friendly preview next to the parquet.

Manual equivalent:

```bash
norma-core/bin/dataset-generator \
  -episode.duration 45 \
  -from <START_FRAME> \
  -to <END_FRAME> \
  -robot localhost \
  -output ../data/elrobot-pizza \
  -task "place pepperoni on pizza"
```

| Export flag | Meaning |
| --- | --- |
| `-episode.duration` | Minimum episode length (seconds) |
| `-from` / `-to` | Frame pointers from History tags |
| `-robot` | Station host (`STATION_HOST`) |
| `-output` | Parquet path prefix |
| `-task` | Same string as `TASK` in `elrobot.env` |
| `-episode.min-commands` | Drop episodes with too few commands (default **100**) |

Read the CLI log: it lists kept and discarded episodes. If you see `insufficient commands`, re-run `dataset-generator` with a lower `-episode.min-commands`.

## Where data is stored

| Location | Contents |
| --- | --- |
| `elrobot/norma-core/bin/station_data/` | Raw Station history (normfs). Grows while Station runs; bounded by Station disk/memory limits. |
| `${DATASET_OUTPUT}.parquet` | Exported ML dataset (e.g. `data/elrobot-pizza.parquet` at repo root) |
| `${DATASET_OUTPUT}.mp4` | Optional preview from `export_dataset.sh` |

Back up Parquet files before wiping `station_data`. This repository includes small sample Parquet exports and MP4 previews under `data/`, alongside the converted dataset under `newdata/`.

Do not commit large episode exports to this repo. Keep larger datasets on the Hub or team object storage.

## Dataset format (Station parquet)

One row = one frame. Schema (v0.1.0-beta.10 export):

| Column | Meaning |
| --- | --- |
| `episode_start_ns` | Wall-clock ns of the episode’s first frame (episode id) |
| `global_frame_id` | Frame id (binary, little-endian) |
| `timestamp_ns_since_episode_start` | Time within episode |
| `joints` | Per-servo telemetry: raw position/goal, **`position_norm` / `goal_norm`**, current, velocity, calibrated ranges, … |
| `images` | List of `{jpeg: binary}` — one embedded JPEG per camera |
| `task` | Your `-task` / `TASK` string |

ElRobot has **8** servos per bus; exported `joints` lists reflect what Station recorded on the follower/leader buses during teleop.

## After export — training and deploy

Fine-tune and run policies: **[Train and deploy](train-and-deploy.md)** (SmolVLA in `smolvla_py`, CUDA, 6 vs 8 DoF limits).

For SO-101 + `lerobot-train`, use [SO-101 data collection](../../so101/docs/data-collection.md).

## Data quality checklist

Before fine-tuning or sharing a parquet:

- [ ] Wrist and side cameras look correct in the Station UI before tagging
- [ ] Same `TASK` string in `elrobot.env`, export, and deploy
- [ ] Export log shows the expected number of episodes kept (not mostly “too short” or “insufficient commands”)
- [ ] Preview MP4 motion matches what you demonstrated
- [ ] Leader and follower were calibrated; follower tracked during `teleop.sh`

Setup: [ElRobot setup](../README.md).
