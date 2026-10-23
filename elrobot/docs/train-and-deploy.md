# ElRobot — Train, deploy, and evaluate

Fine-tune and run policies on the ElRobot follower using NormaCore Station parquet and the **`smolvla_py`** trainer shipped in norma-core **v0.1.0-beta.10**. This is **not** the LeRobot `lerobot-train` path — see [SO-101 train and deploy](../../so101/docs/train-and-deploy.md).

**Prerequisites:** Station running, buses calibrated, exported `.parquet` from [data collection](data-collection.md). Setup: [ElRobot README](../README.md).

Run commands from `elrobot` unless a step says `cd norma-core/...`.

## Contents

- [Fine-tune (SmolVLA)](#fine-tune)
- [Deploy on the arm](#deploy-on-arm)
- [Evaluation](#evaluation)
- [Limits and troubleshooting](#limits)

---

<a id="fine-tune"></a>

## 1. Fine-tune (SmolVLA)

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
