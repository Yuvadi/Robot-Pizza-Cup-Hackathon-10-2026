# Policy training

Fine-tune LeRobot policies with `lerobot-train` on the **pizza Hub dataset**. Local runs are **20-step smokes**. Full GPU training is a Nebius AI Job; checkpoints land on a **shared POSIX filesystem**.

**Arm rollout after training:** [Train and deploy](../docs/train-and-deploy.md)

Run the commands below from `so101`. Hugging Face, W&B, and Nebius settings live in the repo-root `.env` (`../.env` from `so101`; scripts load it automatically). The scripts load that file themselves.

Do these steps **in order**. Never run 30k steps on a Mac.

## Dataset

YAML configs default to this Hub repo (`DATASET_REPO_ID` in `.env`):

| Field | Value |
| --- | --- |
| Hub repo | `mnrozhkov/pizza_hack_v1_fake_ingredients` |
| Episodes | 70 |
| Frames | ~42k |
| Cameras | `observation.images.front`, `observation.images.top` |
| Task | Pick a pepperoni token and place it on the pizza. |

After you record your own demos, set `HF_USER`, `DATASET_NAME`, and `DATASET_REPO_ID` in the repo-root `.env` (not `station.env`).

## 0. `.env`

```bash
cd /path/to/pizza-robo-hack
cp .env.example .env
# HF_TOKEN, HF_USER, DATASET_NAME, DATASET_REPO_ID, DATASET_REVISION
# WANDB_API_KEY, WANDB_ENTITY (yours); WANDB_PROJECT=robotpizzacup
# FILESYSTEM_ID, SUBNET_ID, IMAGE, PARENT_ID
cd so101
set -a && source ../.env && set +a
```

`FILESYSTEM_ID` is required for any cloud submit (POSIX FS, not an S3 bucket). `PARENT_ID` and `IMAGE` in `.env.example` are organizer defaults — override for your own Nebius project.

Join W&B project `robotpizzacup` if you want the shared board. Metrics are scalars only (`wandb.disable_artifact=true`).

## 1. Build and push images

App image is `FROM $DEPS_IMAGE`. Hub must have `:deps` **before** `:latest` (or a local `:deps` must exist).

```bash
cd /path/to/pizza-robo-hack/so101
set -a && source ../.env && set +a

# Once (torch / lerobot layer — slow). Skip if :deps is already on Hub.
bash training/scripts/build_docker.sh --target deps --push

# YAML + train_entrypoint.sh (fast). Do this after training/policy or the entrypoint change.
bash training/scripts/build_docker.sh --push
```

Always `linux/amd64`. Confirm:

```bash
docker pull "$IMAGE"
docker inspect "$IMAGE" --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'
# ["/lerobot/train_entrypoint.sh"] ["--config_path=training/policy/act/config.yaml"]
```

| Tag | Contains | Rebuild when |
| --- | --- | --- |
| `:deps` | CUDA, uv, torch, lerobot | `pyproject.toml` / `uv.lock` / `Dockerfile.deps` |
| `:latest` | `:deps` + pizza policy YAML + entrypoint | YAML / entrypoint |

Participants should **pull** the published image unless they change dependencies.

## 2. Local smoke (pizza, 20 steps)

```bash
set -a && source ../.env && set +a
uv run lerobot-train --help
bash training/scripts/train_pizza.sh act 20
```

Same as:

```bash
uv run lerobot-train \
  --config_path=training/policy/act/config.yaml \
  --policy.device=mps \
  --steps=20 \
  --batch_size=2 \
  --policy.push_to_hub=false
```

`train_local.sh` / `train_pizza.sh` **refuse** more than 200 local steps. Expect MPS on Apple Silicon.

## 3. Docker CPU smoke (same amd64 image)

```bash
set -a && source ../.env && set +a
IMAGE=pizza-robo-hack:dev bash training/scripts/run_docker.sh --rebuild act 20
```

Uses `training/policy/act/config.yaml` (pizza dataset). `Using device: cpu` on macOS is expected. You want `lerobot-train`, not `python -m training.run`.

## 4. Cloud dry-run (no job)

Validates the Nebius spec. Creates nothing.

```bash
set -a && source ../.env && set +a
bash training/scripts/run_serverless.sh --dry-run --debug act
```

Fix CLI/validation errors here. Need `FILESYSTEM_ID` (and usually `SUBNET_ID`).

## 5. Cloud canary (`--debug`)

**500 steps, 1h timeout**, still mounts the filesystem — proves CUDA + Hub pizza dataset + volume.

```bash
bash training/scripts/run_serverless.sh --debug act
```

Save the printed **job name** and **id**.

```bash
JOB_ID=job-...
nebius ai job get --id "$JOB_ID" --format json
nebius ai job logs "$JOB_ID" --tail 200
```

Poll `job get` (do not `logs --follow`). Healthy: `Using device: cuda`, pizza dataset download, W&B URL, then steps.

| If | Do |
| --- | --- |
| `STARTING` > ~10 min | **Cancel** (still billing). Check public amd64 `IMAGE` |
| `multiple subnets` | set `SUBNET_ID` |
| Dataset 404 | `DATASET_REVISION` and `HF_TOKEN` in `.env` |

Do **not** start 30k until this job **completes** and files exist under `/mnt/checkpoints/$HF_USER/<job-name>/`.

## 6. Full pizza run

Same image, new unique output dir. Default: 1× L40S, 6h.

```bash
bash training/scripts/run_serverless.sh act 30000
# or
bash training/scripts/run_serverless.sh smolvla 20000
```

Optional Hub copy for the robot laptop (no FS mount on the Mac):

```bash
POLICY_PUSH_TO_HUB=true bash training/scripts/run_serverless.sh act 30000
# Hub repo defaults to $HF_USER/pizza-<policy> (override with POLICY_REPO_ID)
```

Checkpoint on the FS:

```text
/mnt/checkpoints/<HF_USER>/<job-name>/checkpoints/last/pretrained_model
```

If the FS is mounted locally:

```bash
export CHECKPOINT_FS_ROOT=/path/to/mount
export RUN_NAME=<job-name from step 6>
bash training/scripts/download_checkpoint.sh "$RUN_NAME"
uv run python -m training.eval lerobot-outputs/$RUN_NAME/checkpoints/last/pretrained_model
bash scripts/rollout.sh lerobot-outputs/$RUN_NAME/checkpoints/last/pretrained_model
```

Or `bash scripts/rollout.sh "$HF_USER/pizza-smolvla"` if you pushed (default Hub id is `$HF_USER/pizza-<policy>`).

---

## Scripts (`training/scripts/`)

| Script | Purpose |
| --- | --- |
| `train_pizza.sh` | Local pizza `lerobot-train` smoke (≤200 steps) |
| `train_local.sh` | Same, explicit policy/dataset/steps |
| `build_docker.sh` | Build/push `:latest` or `:deps` |
| `run_docker.sh` | CPU Docker smoke |
| `run_serverless.sh` | **Only** cloud entry: `nebius ai job create` |
| `download_checkpoint.sh` | Copy a run from a local FS mount |
| `train_entrypoint.sh` | Image entrypoint (fresh train or resume) |

`python -m training.run` is an organizer leftover. Do not use it.

YAML: `training/policy/{act,smolvla,diffusion}/config.yaml`. CLI flags override YAML.

Serverless AI is **one VM per job** (no multi-node). Default preset: `gpu-l40s-a` / `1gpu-24vcpu-96gb`.

## Resume from the latest checkpoint

Do **not** `nebius ai job restart` a from-scratch job: that spec still starts `lerobot-train` without `--resume`, and a new VM will either clobber the run or train from step 0.

**Supported:** a **new** job with the same `OUTPUT_DIR` (same `RUN_NAME`):

```bash
export RUN_NAME=pizza-<hf-user>-act-<dataset>-<timestamp>   # original job name
bash training/scripts/run_serverless.sh --resume act
RESUME_STEPS=40000 bash training/scripts/run_serverless.sh --resume act
```

The image entrypoint, if it finds `$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json`, runs:

```bash
lerobot-train \
  --config_path=.../train_config.json \
  --resume=true \
  --policy.device=cuda
```

Equivalent CLI:

```bash
lerobot-train \
  --config_path=/mnt/checkpoints/<user>/<run>/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## Preemptible + restart (usually skip this)

`--preemptible` is cheaper because the platform may **stop the VM at any time**. Pairing it with `--restart-policy on-failure` only helps if training **resumes from disk**.

| What goes wrong | Why |
| --- | --- |
| Restart trains from step 0 | Original argv had no `--resume`; empty or missing `train_config.json` |
| Lost 5–15+ minutes | Reclaim between `save_freq` dumps |
| Restart loop until `--timeout` | Bad config / OOM, still billing |
| Two teams share `output_dir` | Unique dirs are required; collisions corrupt checkpoints |

The entrypoint **does** resume when `checkpoints/last` exists, so a reclaim *after* the first save can continue. Until that first checkpoint, a preemptible kill is a full restart from zero.

For the hackathon use **on-demand** (no `--preemptible`). If you opt in:

```bash
bash training/scripts/run_serverless.sh --preemptible act 30000
```

you accept lost work since the last `save_freq` and a possible silent retrain if the FS write failed.

`nebius ai job restart` re-runs the **same spec**. Use `--resume` (new job, same `RUN_NAME` / `OUTPUT_DIR`) instead.

## Outputs

| Location | When |
| --- | --- |
| `outputs/train/...` | Local `lerobot-train` (smoke) |
| `lerobot-outputs/` | Docker smoke / FS download |
| `/mnt/checkpoints/$HF_USER/<job-name>/` | Cloud jobs |

## Policy profiles (pizza dataset)

| Profile | Policy | Batch | Steps | Where | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| Install check | ACT | 2 | 20 | Mac MPS / Docker CPU | local `outputs/` (throwaway) |
| Cloud canary | ACT | 8 | 500 | 1× L40S `--debug` | shared FS |
| First robot policy | ACT | 8 | 30k–80k | 1× L40S | FS (+ optional Hub) → rollout |
| Headline VLA | SmolVLA | 8 | 20k | 1× L40S | same; YAML maps `front`/`top` → `camera1`/`camera2` |
| Optional | Diffusion | 8 | 20k+ | 1× L40S | same |
