"""Predict from a LeRobot checkpoint without moving hardware unless --execute is set."""

import argparse
import asyncio
import io
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from PIL import Image

from safety import (
    MAX_DELTA_TICKS,
    Joint,
    ensure_fresh,
    goal_preview,
    goals,
    motor_ids,
    span,
    verify_station_config,
)


ELROBOT = Path(__file__).resolve().parents[1]
CAMERAS = ("observation.images.camera_0", "observation.images.camera_1")


def validate_policy(policy) -> None:
    config = policy.config
    if tuple(config.input_features["observation.state"].shape) != (8,):
        raise ValueError("Checkpoint must have 8 state dimensions.")
    if tuple(config.output_features["action"].shape) != (8,):
        raise ValueError("Checkpoint must have 8 action dimensions.")
    if tuple(config.image_features) != CAMERAS:
        raise ValueError(f"Checkpoint camera keys must be {CAMERAS} in this order.")
    for key in CAMERAS:
        if tuple(config.image_features[key].shape) != (3, 224, 224):
            raise ValueError(f"Checkpoint {key} must have shape (3, 224, 224).")


def predict(policy, batch: dict, strict: bool = True, reset: bool = True) -> list[float]:
    if reset:
        policy.reset()
    with torch.inference_mode():
        action = policy.select_action(batch)
    if tuple(action.shape) != (1, 8) or not torch.isfinite(action).all():
        raise ValueError(f"Expected one finite 8-joint action; got shape {tuple(action.shape)}.")
    values = action[0].cpu().tolist()
    if strict and any(value < 0 or value > 1 for value in values):
        raise ValueError(f"Predicted normalized goals outside [0, 1]: {values}; refusing to clip.")
    return values


async def warmup_policy(policy, batch: dict) -> None:
    worker = asyncio.create_task(asyncio.to_thread(predict, policy, batch, strict=False))
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Do not reset a policy while its inference thread is still using it.
        await worker
        raise
    finally:
        policy.reset()


def sample_indices(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("Dataset and sample count must be non-empty.")
    count = min(count, length)
    return sorted({round(i * (length - 1) / max(count - 1, 1)) for i in range(count)})


def offline(args, policy) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset, root=getattr(args, "dataset_root", None), video_backend="pyav")
    errors, out_of_range = [], 0
    for index in sample_indices(len(dataset), getattr(args, "samples", 1)):
        sample = dataset[index]
        if sample["task"] != args.task:
            raise ValueError(f"Task mismatch: dataset has {sample['task']!r}, supplied {args.task!r}.")
        batch = {
            key: sample[key].unsqueeze(0).to("cuda")
            for key in ("observation.state", *CAMERAS)
        }
        batch["task"] = [args.task]
        predicted = predict(policy, batch, strict=False)
        recorded = sample["action"].tolist()
        error = [abs(p - r) for p, r in zip(predicted, recorded, strict=True)]
        errors.append(error)
        bad = [i for i, value in enumerate(predicted) if value < 0 or value > 1]
        out_of_range += bool(bad)
        print(f"\nFrame {index}:")
        print("  predicted:", [round(v, 3) for v in predicted])
        print("  recorded: ", [round(v, 3) for v in recorded])
        print("  abs error:", [round(v, 3) for v in error])
        if bad:
            print(f"  OUT OF RANGE joints {bad}: live mode would REFUSE this prediction.")
    per_joint = np.mean(errors, axis=0)
    print("\nMean abs error per joint:", [round(float(v), 3) for v in per_joint])
    print(f"Overall mean abs error: {float(per_joint.mean()):.3f}")
    print(f"Frames with out-of-range predictions: {out_of_range}/{len(errors)}")
    print("OFFLINE ONLY: no Station connection and no robot commands.")


def decode_frame(frame, camera_order: list[int]) -> tuple[dict, list[Joint]]:
    raw_joints = frame.get_joints() or []
    images = frame.get_images() or []
    if len(raw_joints) != 8 or len(images) != 2:
        raise ValueError("Station must provide exactly 8 follower joints and 2 cameras.")
    joints = [
        Joint(j.get_position(), j.get_position_norm(), j.get_range_min(), j.get_range_max())
        for j in raw_joints
    ]
    for index, joint in enumerate(joints):
        try:
            span(joint)
        except ValueError as error:
            raise ValueError(f"Station joint slot {index} (zero-based): {error}") from error
    batch = {
        "observation.state": torch.tensor(
            [[j.normalized for j in joints]], dtype=torch.float32, device="cuda"
        )
    }
    for key, index in zip(CAMERAS, camera_order, strict=True):
        with Image.open(io.BytesIO(bytes(images[index].get_jpeg()))) as image:
            if image.size != (224, 224):
                raise ValueError(f"{key}: expected 224x224 training resolution, got {image.size}.")
            rgb = np.array(image.convert("RGB"), copy=True)
        batch[key] = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to("cuda").float() / 255
    return batch, joints


def check_frame_age(frame) -> None:
    samples = [*(frame.get_joints() or []), *(frame.get_images() or [])]
    ensure_fresh(
        frame.get_monotonic_stamp_ns(),
        [sample.get_monotonic_stamp_ns() for sample in samples],
        time.clock_gettime_ns(time.CLOCK_BOOTTIME),
        max_age=2.0,
    )


def rollout_goals(action, joints, max_delta, previous_targets, ids, phase):
    preview = goal_preview(action, joints)
    if any(abs(delta) > max_delta for _, delta in preview):
        print(f"ROLLOUT BLOCKED ({phase}): position gap exceeds {max_delta} ticks.", flush=True)
        print("motor present target gap previous_target target_change tracking_gap", flush=True)
        for index, (joint, (target, gap)) in enumerate(zip(joints, preview, strict=True)):
            if previous_targets is None:
                previous, change, tracking = "-", "-", "-"
            else:
                previous = previous_targets[index]
                previous_offset = (previous - joint.minimum) % 4096
                change = (target - joint.minimum) % 4096 - previous_offset
                tracking = previous_offset - (joint.position - joint.minimum) % 4096
            print(f"{ids[index]} {joint.position} {target} {gap:+d} {previous} {change} {tracking}",
                  flush=True)
    return goals(action, joints, max_delta)


async def rollout(args, policy, fetch, send) -> None:
    """Bounded rollout; callbacks supply telemetry and send validated targets."""
    if not args.execute or not math.isfinite(args.rollout_seconds) or args.rollout_seconds <= 0:
        raise ValueError("Rollout requires --execute and a positive finite duration.")
    max_invalid_retries = getattr(args, "max_invalid_retries", 0)
    if not 0 <= max_invalid_retries <= 3:
        raise ValueError("--max-invalid-retries must be in 0..3.")
    order = [int(value) for value in args.camera_order.split(",")]
    action_steps = policy.config.n_action_steps
    if not isinstance(action_steps, int) or action_steps < 1:
        raise ValueError("Checkpoint n_action_steps must be a positive integer.")
    policy.reset()
    deadline = time.monotonic() + args.rollout_seconds
    calibration = None
    last_stamp = None
    count = 0
    predictions = 0
    chunk_steps = 0
    consecutive_invalid = 0
    previous_targets = None
    ids = motor_ids(args.motor_ids)
    try:
        while time.monotonic() < deadline:
            tick = time.monotonic()
            frame = await fetch()
            stamp = frame.get_monotonic_stamp_ns()
            if last_stamp is not None and stamp <= last_stamp:
                raise ValueError("Rollout input did not advance; stopping new commands.")
            batch, joints = decode_frame(frame, order)
            ranges = [(j.minimum, j.maximum) for j in joints]
            if calibration is None:
                calibration = ranges
            if ranges != calibration:
                raise ValueError("Calibration changed during rollout.")
            batch["task"] = [args.task]
            chunk_start = chunk_steps % action_steps == 0
            inference_started = time.monotonic()
            action = predict(policy, batch, strict=not max_invalid_retries, reset=False)
            # TIP: Uncomment line below to clamp tiny float overshoots (e.g. 1.0005, -0.0008) at runtime:
            #action = [max(0.0, min(1.0, float(v))) for v in action]
            inference_seconds = time.monotonic() - inference_started
            predictions += 1
            chunk_steps += 1
            phase = (f"prediction={predictions}, chunk_start={chunk_start}, "
                     f"inference_s={inference_seconds:.3f}")
            print(f"Rollout inference complete ({phase}); checking observation freshness.", flush=True)
            check_frame_age(frame)
            if any(value < 0 or value > 1 for value in action):
                consecutive_invalid += 1
                print(f"ROLLOUT REJECTED ({phase}): normalized goals outside [0, 1]: "
                      f"{action}. No command sent; consecutive rejection "
                      f"{consecutive_invalid}, retry limit {max_invalid_retries}.", flush=True)
                policy.reset()
                chunk_steps = 0
                last_stamp = stamp
                if consecutive_invalid > max_invalid_retries:
                    raise ValueError("Out-of-range prediction retry limit exceeded; stopping rollout.")
                await asyncio.sleep(max(0.1, tick + 0.1 - time.monotonic()))
                continue
            rollout_goals(action, joints, args.max_delta_ticks, previous_targets, ids, phase + ", input")
            latest = await fetch()
            _, current = decode_frame(latest, order)
            if [(j.minimum, j.maximum) for j in current] != calibration:
                raise ValueError("Calibration changed before rollout send.")
            targets = rollout_goals(action, current, args.max_delta_ticks, previous_targets, ids,
                                    phase + ", refreshed")
            check_frame_age(latest)
            if time.monotonic() >= deadline:
                break
            await send(targets)
            previous_targets = targets
            count += 1
            consecutive_invalid = 0
            last_stamp = latest.get_monotonic_stamp_ns()
            print(f"Rollout command {count} enqueued; targets: {targets}", flush=True)
            await asyncio.sleep(max(0.1, tick + 0.1 - time.monotonic()))
    finally:
        policy.reset()
        print(f"Rollout ended after {count} command packs. No further targets will be sent.")
        print("Torque is unchanged; last sent goals may still be active. Use the physical stop if needed.")


async def live(args, policy) -> None:
    if args.server not in ("localhost", "127.0.0.1"):
        raise ValueError("Live bridge supports only same-host WSL Station (localhost), not remote clocks.")
    if not args.bus_serial or not args.motor_ids or args.camera_order is None:
        raise ValueError("Live mode requires --bus-serial, --motor-ids, and --camera-order.")
    ids = motor_ids(args.motor_ids)
    order = [int(value) for value in args.camera_order.split(",")]
    if sorted(order) != [0, 1]:
        raise ValueError("--camera-order must be 0,1 or 1,0 in training camera order.")
    config = yaml.safe_load(args.station_config.read_text())
    verify_station_config(config, args.bus_serial, args.queue)
    duration = getattr(args, "rollout_seconds", 0)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("--rollout-seconds must be finite and nonnegative.")
    max_invalid_retries = getattr(args, "max_invalid_retries", 0)
    if not 0 <= max_invalid_retries <= 3 or (max_invalid_retries and not duration):
        raise ValueError("--max-invalid-retries must be 0..3; nonzero requires a rollout.")
    if duration:
        if not args.execute:
            raise ValueError("Rollout requires --execute and a positive finite duration.")
        print(f"ROLLOUT: up to {duration:g} seconds. Stop mirroring, other policies, and recording.")
        print("Clear the workspace; keep the physical stop accessible. Ctrl+C stops NEW commands only.")
        print(f"Torque is not changed. At most 10 commands/second; checkpoint action horizon: "
              f"{policy.config.n_action_steps} actions before replanning.")
        print("Queued actions do not incorporate new observations until replanning; "
              "every send still requires fresh telemetry and movement checks.")
        if max_invalid_retries:
            print(f"Out-of-range actions: at most {max_invalid_retries} consecutive retries. "
                  "Rejected actions send nothing; previous goals remain active.")
        phrase = f"ROLLOUT {args.bus_serial} {duration:g}"
        if input(f"Type {phrase} to authorize: ").strip() != phrase:
            raise ValueError("Rollout not confirmed; no commands sent.")
    elif args.execute:
        print("ONE MOVEMENT ONLY. Stop teleoperation; verify follower bus, joint/camera order,")
        print("calibration, and a clear workspace. This script does not enable torque.")
        if input(f"Type MOVE {args.bus_serial} to permit one prediction/send: ").strip() != f"MOVE {args.bus_serial}":
            raise ValueError("Movement not confirmed; no commands sent.")

    norma = args.norma_core.resolve()
    if not (norma / "target/gen_python/protobuf/drivers/inferences/normvla.py").is_file():
        raise FileNotFoundError(f"Missing Station generated Python modules under {norma}.")
    sys.path[:0] = [str(norma), str(norma / "software/station/shared")]
    from station_py import new_station_client, send_commands
    from target.gen_python.protobuf.drivers.inferences import normvla
    from target.gen_python.protobuf.drivers.st3215 import st3215
    from target.gen_python.protobuf.station import commands, drivers

    client = await new_station_client(args.server, logging.getLogger("station"))

    async def fetch():
        reader = client.read_from_tail(args.queue, offset=b"\x00", limit=1, step=1, buf_size=1)
        try:
            entry = await asyncio.wait_for(reader.data.get(), timeout=5)
        except TimeoutError as error:
            raise TimeoutError(
                f"No Station frame received from {args.queue} within 5 seconds. "
                "No command sent for this observation. Check Station connection, "
                "fresh camera/joint telemetry, and inference skip messages."
            ) from error
        if entry is None:
            raise RuntimeError(f"Station did not return a frame: {reader.err}")
        frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
        check_frame_age(frame)
        return frame

    async def send(predicted):
        sync = st3215.ST3215SyncWriteCommand(
            address=0x2A,
            motors=[
                st3215.ST3215SyncWriteCommand_MotorWrite(
                    motor_id=mid, value=target.to_bytes(2, "little")
                )
                for mid, target in zip(ids, predicted, strict=True)
            ],
        )
        command = commands.DriverCommand(
            type=drivers.StationCommandType.STC_ST3215_COMMAND,
            body=st3215.Command(target_bus_serial=args.bus_serial, sync_write=sync).encode(),
        )
        await asyncio.wait_for(send_commands(client, [command]), timeout=1 if duration else 5)

    try:
        if duration:
            print("Warming up policy; no commands will be sent. Warm-up output is discarded.",
                  flush=True)
            frame = await fetch()
            batch, _ = decode_frame(frame, order)
            batch["task"] = [args.task]
            started = time.monotonic()
            await warmup_policy(policy, batch)
            print(f"Warm-up completed in {time.monotonic() - started:.3f}s; "
                  "fetching fresh observations for rollout.", flush=True)
            await rollout(args, policy, fetch, send)
            return
        frame = await fetch()
        batch, joints = decode_frame(frame, order)
        batch["task"] = [args.task]
        action = predict(policy, batch, strict=args.execute)
        if not args.execute and any(value < 0 or value > 1 for value in action):
            print("motor  predicted_normalized  range_check")
            for mid, value in zip(ids, action, strict=True):
                status = "OUT OF RANGE" if value < 0 or value > 1 else "within range"
                print(f"{mid:5d}  {value:20.6f}  {status}")
            print("BLOCKED for execution: prediction outside [0, 1]; no clipping applied.")
            print("DRY RUN: no commands sent. Invalid predictions are not converted to motor targets.")
            return
        if args.execute:
            check_frame_age(frame)
        else:
            latest = await fetch()
            _, current = decode_frame(latest, order)
            if [(j.minimum, j.maximum) for j in current] != [(j.minimum, j.maximum) for j in joints]:
                raise ValueError("Calibration changed during inference; no commands sent.")
            joints = current
            print("Prediction-only preview: movement bounds use refreshed telemetry.")
        preview = goal_preview(action, joints)
        print("motor  present  predicted  delta_ticks  limit_check")
        for mid, joint, (target, delta) in zip(ids, joints, preview, strict=True):
            status = "OVER LIMIT" if abs(delta) > args.max_delta_ticks else "within limit"
            print(f"{mid:5d}  {joint.position:7d}  {target:9d}  {delta:+11d}  {status}")
        if not args.execute:
            if any(abs(delta) > args.max_delta_ticks for _, delta in preview):
                print(f"BLOCKED for execution: movement exceeds {args.max_delta_ticks} ticks.")
            print("DRY RUN: no commands sent. A predicted action is not a safety certification.")
            return
        goals(action, joints, args.max_delta_ticks)
        latest = await fetch()
        _, current = decode_frame(latest, order)
        if [(j.minimum, j.maximum) for j in current] != [(j.minimum, j.maximum) for j in joints]:
            raise ValueError("Calibration changed during inference; no commands sent.")
        predicted = goals(action, current, args.max_delta_ticks)
        check_frame_age(latest)
        await send(predicted)
        print("One command pack enqueued. This is not confirmation of completed physical motion.")
    finally:
        client.close_connection()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Compare predictions with dataset frames; never connect to Station.")
    parser.add_argument("--samples", type=int, default=5, help="Offline: number of evenly spaced dataset frames to evaluate.")
    parser.add_argument("--dataset", default="PeteVLA/cheese2")
    parser.add_argument("--dataset-root", type=Path, help="Local converted dataset for offline prediction.")
    parser.add_argument("--task", default="spreading cheese on pizza")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--queue", default="inference/normvla")
    parser.add_argument("--norma-core", type=Path, default=ELROBOT / "norma-core")
    parser.add_argument("--station-config", type=Path, default=ELROBOT / "norma-core/bin/station.yaml")
    parser.add_argument("--bus-serial")
    parser.add_argument("--motor-ids", help="Eight ascending IDs matching training joint order.")
    parser.add_argument("--camera-order", help="Station image indices for training camera_0,camera_1, e.g. 0,1.")
    parser.add_argument(
        "--max-delta-ticks",
        type=int,
        default=100,
        help=f"Maximum allowed movement per command (1..{MAX_DELTA_TICKS}; default: 100).",
    )
    parser.add_argument("--execute", action="store_true", help="Explicitly permit ONE send after interactive confirmation.")
    parser.add_argument("--rollout-seconds", type=float, default=0,
                        help="With --execute: positive finite rollout duration in seconds; 0 means one send.")
    parser.add_argument(
        "--max-invalid-retries", type=int, default=0,
        help="Rollout only: retry out-of-range actions up to 3 consecutive times (default: 0, abort).",
    )
    args = parser.parse_args()
    if args.offline and args.execute:
        parser.error("--offline cannot be combined with --execute.")
    if (not math.isfinite(args.rollout_seconds) or args.rollout_seconds < 0
            or (args.rollout_seconds and (args.offline or not args.execute))):
        parser.error("--rollout-seconds must be finite and nonnegative; nonzero requires live --execute.")
    if not 0 <= args.max_invalid_retries <= 3 or (args.max_invalid_retries and not args.rollout_seconds):
        parser.error("--max-invalid-retries must be 0..3; nonzero requires --rollout-seconds.")
    if not 1 <= args.max_delta_ticks <= MAX_DELTA_TICKS:
        parser.error(
            f"--max-delta-ticks must be in 1..{MAX_DELTA_TICKS}; "
            "disabling the limit is not supported."
        )
    if not (args.checkpoint / "config.json").is_file():
        parser.error("--checkpoint must point to a trained checkpoint's pretrained_model directory.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required in this WSL environment.")
    logging.basicConfig(level=logging.WARNING)
    policy_type = json.loads((args.checkpoint / "config.json").read_text()).get("type")
    if policy_type == "act":
        from act_checkpoint import load_act_checkpoint

        policy = load_act_checkpoint(args.checkpoint).to("cuda").eval()
    elif policy_type == "smolvla":
        policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint)).to("cuda").eval()
    else:
        parser.error(f"Unsupported checkpoint policy type: {policy_type!r}")
    validate_policy(policy)
    if args.offline:
        offline(args, policy)
    else:
        asyncio.run(live(args, policy))


if __name__ == "__main__":
    main()
