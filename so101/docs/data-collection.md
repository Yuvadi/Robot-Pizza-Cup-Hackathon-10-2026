Commands below run from `so101`. ElRobot recordings use a different format: [ElRobot data collection](../../elrobot/docs/data-collection.md).

# SO-101 (LeRobot)

Official LeRobot references:

| Topic | Link |
| --- | --- |
| Real-robot imitation learning (record / replay / train) | [il_robots](https://huggingface.co/docs/lerobot/il_robots) |
| SO-101 hardware | [so101](https://huggingface.co/docs/lerobot/so101) |
| Streaming video encoding | [streaming_video_encoding](https://huggingface.co/docs/lerobot/streaming_video_encoding) |
| Video encoder parameters | [video_encoding_parameters](https://huggingface.co/docs/lerobot/video_encoding_parameters) |
| Dataset visualizer (web) | [visualize_dataset](https://huggingface.co/spaces/lerobot/visualize_dataset) |
| What makes a good dataset | [LeRobot community datasets](https://huggingface.co/blog/lerobot-datasets) |
| LeRobotDataset v3.0 format | [lerobot-datasets-v3](https://huggingface.co/blog/lerobot-datasets-v3) |

## SO-101 — Configure env files

```bash
cd so101
cp station.env.example station.env
cp ../.env.example ../.env
```

| File | Contains |
| --- | --- |
| `so101/station.env` | Ports, camera indexes, calibration ids, `TASK`, episode timing |
| repo-root `.env` | `HF_USER`, `DATASET_NAME`, `HF_TOKEN`, W&B, Nebius / training |

Set these before recording:

| Variable | File | Purpose |
| --- | --- | --- |
| `HF_USER` | `.env` | Hub **namespace**: your username *or* an org (e.g. `RobotPizzaCup`) |
| `DATASET_NAME` | `.env` | Dataset slug, e.g. `pizza_topping_v1` |
| `TASK` | `station.env` | Natural-language task sentence — **must match train and rollout verbatim** |
| `NUM_EPISODES` | `station.env` | Target episode count (default 50) |
| `EPISODE_TIME_S` | `station.env` | Seconds of teleop per episode (default 30) |
| `RESET_TIME_S` | `station.env` | Seconds to reset the scene between episodes (default 10) |
| `FOLLOWER_ID` / `LEADER_ID` | `station.env` | Calibration keys — same ids as calibrate/teleop |
| `CAMERA_INDEX` | `station.env` | Front / workspace OpenCV index |
| `CAMERA_INDEX_TOP` | `station.env` | Overhead OpenCV index; leave empty for front-only |

Log in only if you will push to the Hub later (token needs **write** access). Recording does not upload:

```bash
hf auth login
# Personal account — copy your username into .env:
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
```

**Personal account** — set `HF_USER` to your username (e.g. `mnrozhkov`). Datasets appear at `https://huggingface.co/datasets/mnrozhkov/<name>`.

**Organization** — set `HF_USER` to the org slug (e.g. `RobotPizzaCup` for [Robot Pizza Cup](https://huggingface.co/RobotPizzaCup)). You must be an org member with permission to create datasets. Datasets appear at `https://huggingface.co/datasets/RobotPizzaCup/<name>`.

## SO-101 — Record

```bash
bash scripts/record.sh
```

`scripts/record.sh` wraps `lerobot-record` with station settings and **H.264 video encoding** so episode MP4s play in QuickTime and Finder on macOS. LeRobot’s default codec is AV1, which many Mac players cannot decode.

Encoding flags (already set in the script):

| Flag | Value | Why |
| --- | --- | --- |
| `--dataset.rgb_encoder.vcodec` | `h264_videotoolbox` (macOS) / `h264` (Linux) | QuickTime-compatible MP4 |
| `--dataset.streaming_encoding` | `true` | Encode during capture; faster `save_episode()` |
| `--dataset.encoder_threads` | `2` per camera | `record.sh` sets `2` (front) or `4` (front + top) |

See [video encoding parameters](https://huggingface.co/docs/lerobot/video_encoding_parameters) to tune quality (`crf`, `preset`, etc.).

### Keyboard controls

Keep the **terminal focused** on macOS unless Accessibility / Input Monitoring is granted to the terminal app.

| Key | Action |
| --- | --- |
| `n` or `→` | Finish this episode, **save it**, start reset / next |
| `r` or `←` | Discard this episode and re-record |
| `q` or `Esc` | Stop session, encode remaining work (local only) |

**Important:** `q` / `Esc` quits immediately. An in-progress episode that was never confirmed with `n` is **not saved**. Press `n` after each demo you want to keep.

### Resume more episodes later

Re-run with `--resume=true` and set `--dataset.num_episodes` to the **additional** count (not the new total). See [checkpointing in il_robots](https://huggingface.co/docs/lerobot/il_robots#2-checkpointing-and-resuming).

```bash
uv run lerobot-record \
  --resume=true \
  --dataset.root="$HOME/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}_<timestamp>" \
  --dataset.num_episodes=10 \
  ... # same robot/teleop/dataset flags as the original run
```

Each new recording session gets a **timestamp suffix** on `repo_id` (e.g. `pizza_topping_v1_20260919_145422`). Use `--dataset.no_stamp=true` on the CLI to keep a fixed name, or pass `--dataset.root` when resuming.

## SO-101 — Where data is stored (local)

Demonstration datasets on disk:

```text
~/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}_<timestamp>/
```

Large episode files are **not** committed to this git repo.

### Sample and smoke-test datasets

Do not commit large episode files to git. Use Hugging Face Hub or object storage.

| Dataset | Use |
| --- | --- |
| [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) | Tiny ACT smoke test (local CPU / Docker) |
| [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) | SmolVLA paper-style pick-place reference |

After `bash scripts/record.sh`, local files land in:

```text
~/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}_<timestamp>/
```

LeRobot appends a timestamp to each new recording session unless you pass `--dataset.no_stamp=true`. Example:

```text
~/.cache/huggingface/lerobot/your-hf-username/pizza_topping_v1_20260919_145422/
```

Hub URL (after upload):

```text
https://huggingface.co/datasets/${HF_USER}/${DATASET_NAME}_<timestamp>
```

Videos are encoded as **H.264** by `scripts/record.sh` so they play in QuickTime on macOS. LeRobot’s default AV1 codec is smaller but often won’t open in Finder — see [streaming video encoding](https://huggingface.co/docs/lerobot/streaming_video_encoding).

Visualize any LeRobot dataset: https://huggingface.co/spaces/lerobot/visualize_dataset

### Calibration (separate from episodes)

Motor calibration for each arm id:

```text
~/.cache/huggingface/lerobot/calibration/
```

Keyed by `FOLLOWER_ID` and `LEADER_ID`. Setup procedure: [SO-101 README — Calibrate](../README.md#3-calibrate).

## SO-101 — Upload to Hugging Face Hub

`scripts/record.sh` saves **locally only** (`--dataset.push_to_hub=false`). LeRobot’s default is to upload; this repo turns that off. Push later with `hf upload` when you choose. See [il_robots — dataset upload](https://huggingface.co/docs/lerobot/il_robots#dataset-upload).

### Choose destination in `.env`

| Destination | `HF_USER` value | Example Hub URL |
| --- | --- | --- |
| Your account | `mnrozhkov` | `https://huggingface.co/datasets/mnrozhkov/pizza_hack_v1_<timestamp>` |
| Event org | `RobotPizzaCup` | `https://huggingface.co/datasets/RobotPizzaCup/pizza_hack_v1_<timestamp>` |

Example — upload to your account:

```bash
# .env
HF_USER=mnrozhkov
DATASET_NAME=pizza_hack_v1
```

Example — upload to the hackathon org:

```bash
# .env
HF_USER=RobotPizzaCup
DATASET_NAME=pizza_hack_v1
```

Then record as usual:

```bash
cd so101
set -a && source ../.env && source station.env && set +a
bash scripts/record.sh
```

Press `n` after each saved episode; press `q` when done to encode locally. Upload is optional (see below).

### Upload an existing local dataset (no re-record)

**Find the real local path first** — it depends on what `HF_USER` was *when you recorded*, not what it is now. If you recorded with a placeholder username, data lives there even after you change `.env`.

```bash
bash scripts/list_datasets.sh
```

Example output:

```text
  your-hf-username/pizza_topping_v1_20260919_145422  (2 episodes)
    path: /Users/.../.cache/huggingface/lerobot/your-hf-username/pizza_topping_v1_20260919_145422
    upload: uv run hf upload <namespace>/<repo> "..." --repo-type dataset
```

Then upload (use the **path** line from the script, not a guessed path):

```bash
# To your account
uv run hf upload mnrozhkov/pizza_hack_v1 \
  ~/.cache/huggingface/lerobot/your-hf-username/pizza_topping_v1_20260919_145422 \
  --repo-type dataset

# To the org
uv run hf upload RobotPizzaCup/pizza_hack_v1 \
  ~/.cache/huggingface/lerobot/your-hf-username/pizza_topping_v1_20260919_145422 \
  --repo-type dataset
```

Use the **exact local folder path** (namespace folder + timestamp suffix from the recording session). Add `--private` to keep the dataset private until you publish it.

### Recording does not upload

`scripts/record.sh` already passes `--dataset.push_to_hub=false`. Use `hf upload` above when you want the Hub copy.

### Use the dataset for training

Pass the full `namespace/dataset_name` (with timestamp if that is how it was recorded):

```bash
uv run lerobot-train \
  --config_path=training/policy/act/config.yaml \
  --dataset.repo_id="mnrozhkov/pizza_hack_v1_20260919_145422" \
  --policy.device=mps \
  --steps=20 \
  --batch_size=2 \
  --policy.push_to_hub=false
```

Full GPU training: `bash training/scripts/run_serverless.sh act 30000` (see [Train and deploy](train-and-deploy.md)).

Works the same for `RobotPizzaCup/...`.

### Common upload issues

| Symptom | Fix |
| --- | --- |
| `403 Forbidden` | Token lacks write access, or you are not a member of the org |
| `No episodes saved — skipping push to hub` | No episode was confirmed with `n` before quit |
| Dataset on Hub under wrong namespace | Set `HF_USER` in `.env` before recording, or `hf upload` to the correct `org/name` |
| Replay/train can't find data | Use full repo id including `_<timestamp>`, or `--dataset.root` pointing at the local folder |

## SO-101 — Dataset format (LeRobot v3.0)

Each recording session produces a **LeRobotDataset** (codebase version `v3.0`). Layout on disk:

```text
.
├── meta/
│   ├── info.json          # schema: fps, feature shapes, video codec
│   ├── stats.json         # normalization stats (after episodes are saved)
│   ├── tasks.parquet      # task strings
│   └── episodes/          # per-episode metadata (length, indices, …)
├── data/                  # tabular frames (parquet, chunked)
│   └── chunk-000/file-000.parquet
└── videos/
    ├── observation.images.front/
    │   └── chunk-000/file-000.mp4   # one MP4 per episode
    └── observation.images.top/      # only if CAMERA_INDEX_TOP is set
        └── chunk-000/file-000.mp4
```

### Per-frame fields (this station)

Recorded at **30 FPS** (`640×480`):

| Feature | Dtype | Content |
| --- | --- | --- |
| `observation.state` | `float32[6]` | Follower joint positions (degrees): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| `observation.images.front` | video | RGB workspace camera |
| `observation.images.top` | video | RGB overhead camera (only if `CAMERA_INDEX_TOP` is set) |
| `action` | `float32[6]` | Leader joint targets (same six joints) |
| `task` | string | e.g. `"Pick a pepperoni token and place it on the pizza."` |
| `timestamp` | float | `frame_index / fps` |
| `frame_index`, `episode_index`, `index`, `task_index` | int | Dataset indexing |

Joint positions live in **parquet**; camera frames are stored as **H.264 MP4** with frame references in parquet. Full format description: [LeRobot dataset docs](https://huggingface.co/docs/lerobot/il_robots#1-data-storage).

## SO-101 — Inspect and replay

Live Rerun during `record.sh` shows joints moving but **freezes the camera**. LeRobot logs images as `static=True` (RAM). The MP4s on disk are the real video. Preview them after the session.

`scripts/record.sh` prints a copy-paste command when it exits. Open the stamped folder in the Rerun app (video + joints stay open until you quit):

```bash
uv run rerun "$HOME/.cache/huggingface/lerobot/mnrozhkov/pizza_hack_v1_YYYYMMDD_HHMMSS"
```

Or the newest local dataset:

```bash
bash scripts/viz_dataset.sh
```

Web visualizer (Hub): [visualize_dataset](https://huggingface.co/spaces/lerobot/visualize_dataset).

### Play a single episode video

After recording with H.264, QuickTime should work. For older AV1 datasets, use VLC or convert:

```bash
ffmpeg -i path/to/file-000.mp4 -c:v h264_videotoolbox -b:v 2M ~/Desktop/episode0.mp4
```

### Replay on the robot

Replays saved joint actions on the follower (no leader):

```bash
bash scripts/replay.sh 0
```

Uses the dataset at `~/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}` — if your run has a timestamp suffix, pass the full path via `--dataset.root` or set `DATASET_NAME` to the stamped folder name.

LeRobot replay docs: [Replay an episode](https://huggingface.co/docs/lerobot/il_robots#replay-an-episode).

## SO-101 — Data quality checklist

Before training, confirm:

- [ ] 50+ episodes with consistent scene setup (object pose, pizza position, camera mount)
- [ ] Same `TASK` string everywhere
- [ ] Episodes saved with `n` (check local `meta/episodes/` or Hub upload)
- [ ] Visualizer shows smooth motion and correct camera framing (`bash scripts/viz_dataset.sh`)
- [ ] `bash scripts/replay.sh 0` looks like your demonstration

Collect more consistent data before switching models if ACT success is below ~70% on the same scene.

Troubleshooting: [troubleshooting.md](troubleshooting.md).
