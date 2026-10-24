# Robotics Pizza Hackathon 2026

**Teach a Robot to Make Pizza in 48 Hours**

October 30 – November 1, 2026 · [Venue / City] · 8–12 teams · 2–4 participants per team

Each team has two kits: a LeRobot SO-101 pair and an ElRobot pair. Pick the arm directory for the stack you are on. They do not share a Python environment.

| Arm | Directory | Stack |
| --- | --- | --- |
| LeRobot SO-101 | [so101](so101/README.md) | `uv` + LeRobot 0.6.1 |
| ElRobot | [elrobot](elrobot/README.md) | NormaCore Station binary |

**SO-101 train → deploy:** [Train and deploy](so101/docs/train-and-deploy.md) · **ElRobot train → deploy:** [Train and deploy](elrobot/docs/train-and-deploy.md)

Status: **Draft v0.1**.

**Geometry + manipulation skeleton:** [Multi-Modal-Approach](Multi-Modal-Approach/README.md).
**ElRobot manual calibration profiles:** [Calibration and backups](elrobot/docs/calibration.md).
**ElRobot browser episode recorder:** [Recording UI](elrobot/docs/episode-recorder.md).

---

## SO-101 (LeRobot)

From `so101`. Full steps: [so101/README.md](so101/README.md).

```bash
cd so101
uv python install 3.12
uv sync
hf auth login
cp station.env.example station.env
cp ../.env.example ../.env
```

That install includes LeRobot **0.6.1** with robot CLIs (`core_scripts` + `feetech`) and policy extras **ACT** (core), **SmolVLA**, **Diffusion**, and **π0 / π0.5**.

| Step | Command (from `so101`) |
| --- | --- |
| Find USB ports | `bash scripts/list_ports.sh` |
| Calibrate | `bash scripts/calibrate.sh follower` then `leader` |
| Teleop | `bash scripts/teleoperate.sh` |
| Record 50 demos | `bash scripts/record.sh` |
| Preview in Rerun | `bash scripts/viz_dataset.sh` |
| Local ACT smoke (20 steps) | `bash training/scripts/train_local.sh act lerobot/pusht 20` |
| Cloud canary (500 steps) | `bash training/scripts/run_serverless.sh --debug act` |
| Cloud ACT | `bash training/scripts/run_serverless.sh act 30000` |
| Copy checkpoint from FS mount | `bash training/scripts/download_checkpoint.sh <job-name>` |
| Run policy | `bash scripts/rollout.sh <checkpoint>` |

Local robot work uses the `uv` venv in `so101` (MPS on Apple Silicon). Real SmolVLA training uses Docker / [Nebius AI Jobs](https://github.com/mnrozhkov/serverless-ai-cookbook/tree/clawbio/robotics), not this laptop.

Upstream: [LeRobot install](https://huggingface.co/docs/lerobot/installation) · [SO-101](https://huggingface.co/docs/lerobot/so101) · [real-robot IL](https://huggingface.co/docs/lerobot/il_robots)

## ElRobot (NormaCore Station)

From `elrobot`. Full steps: [elrobot/README.md](elrobot/README.md).

```bash
cd elrobot
bash scripts/install.sh
bash scripts/run.sh
```

`install.sh` downloads Station **v0.1.0-beta.10** and shallow-clones that tag into `norma-core/`. Serials and the task sentence live in `elrobot.env`. The SmolVLA trainer in the tag is 6-DoF and needs CUDA; ElRobot has 8 servos.

| Step | Command (from `elrobot`) |
| --- | --- |
| Install | `bash scripts/install.sh` |
| Config | `cp elrobot.env.example elrobot.env` |
| Start Station | `bash scripts/run.sh` |
| Calibrate follower | `bash scripts/calibrate.sh` |
| Teleop | `bash scripts/teleop.sh` |
| Export parquet | `bash scripts/export_dataset.sh <START> <END>` |

Upstream: [norma-core](https://github.com/norma-core/norma-core/tree/v0.1.0-beta.10)

---

## Document map

| Document | Covers |
| --- | --- |
| [SO-101 setup](so101/README.md) | Install, calibrate, collect, train, deploy |
| [ElRobot setup](elrobot/README.md) | Install, calibrate, collect, train, deploy |
| [SO-101 data collection](so101/docs/data-collection.md) | `lerobot-record`, H.264, LeRobot v3 |
| [ElRobot data collection](elrobot/docs/data-collection.md) | Station tags, parquet export |
| [SO-101 train and deploy](so101/docs/train-and-deploy.md) | Local smoke, Nebius GPU, rollout |
| [ElRobot train and deploy](elrobot/docs/train-and-deploy.md) | SmolVLA fine-tune, deploy, evaluation |

The two datasets are not interchangeable. A Station parquet will not open in `lerobot-train` without conversion. Norma's SmolVLA trainer reads Station parquet; SO-101 training reads LeRobot v3.

## Local data

Nothing under `data/` is committed.

| Stack | Where recordings land |
| --- | --- |
| ElRobot export | `data/elrobot-pizza.parquet` and `data/elrobot-pizza.mp4` (prefix `../data/elrobot-pizza` in `elrobot.env`) |
| SO-101 | `~/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}_<timestamp>/` |

---

## Quick facts

- **Format:** Teleoperate → Collect Data → Fine-Tune → Deploy → Compete
- **Three challenges:** Precision Topping (30 pts) · Recipe Assembly (40 pts) · Mystery Pizza (30 pts) — 100 pts total
- **Primary KPI:** ≥70% of participating teams successfully deploy one autonomous learned policy to a real robot
- **Platforms:** each team uses both the SO-101 kit (LeRobot) and the ElRobot kit (NormaCore Station)
