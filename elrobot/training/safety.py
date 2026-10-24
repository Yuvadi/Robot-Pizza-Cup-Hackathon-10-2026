"""Pure validation for the Station follower bridge; no device writes."""

import math
from dataclasses import dataclass
from typing import Sequence


MAX_DELTA_TICKS = 1500


@dataclass(frozen=True)
class Joint:
    position: int
    normalized: float
    minimum: int
    maximum: int


def motor_ids(value: str) -> list[int]:
    ids = [int(part) for part in value.split(",")]
    if len(ids) != 8 or len(set(ids)) != 8 or any(mid < 1 or mid > 253 for mid in ids):
        raise ValueError("Provide exactly 8 distinct motor IDs in 1..253.")
    if ids != sorted(ids):
        raise ValueError("Station joint order is ascending motor ID; IDs must be sorted.")
    return ids


def span(joint: Joint) -> int:
    if not all(0 <= value < 4096 for value in (joint.position, joint.minimum, joint.maximum)):
        raise ValueError("Servo positions and calibration endpoints must be in 0..4095.")
    width = (joint.maximum - joint.minimum) % 4096
    if width == 0:
        raise ValueError("Missing/invalid calibration: minimum equals maximum.")
    if (joint.position - joint.minimum) % 4096 > width:
        offset = (joint.position - joint.minimum) % 4096
        outside = min(offset - width, 4096 - offset)
        raise ValueError(
            f"Present position is outside its calibrated arc: position={joint.position}, "
            f"minimum={joint.minimum}, maximum={joint.maximum}, "
            f"outside_by={outside} ticks. Refusing to clamp the observed position."
        )
    if not math.isfinite(joint.normalized) or not 0 <= joint.normalized <= 1:
        raise ValueError("Invalid normalized joint position.")
    expected = ((joint.position - joint.minimum) % 4096) / width
    if abs(expected - joint.normalized) > 0.01:
        raise ValueError("Normalized position disagrees with raw calibration.")
    return width


def goal_preview(actions: Sequence[float], joints: Sequence[Joint]) -> list[tuple[int, int]]:
    if len(actions) != 8 or len(joints) != 8:
        raise ValueError("Expected exactly 8 predicted actions and 8 observed joints.")
    result = []
    for index, (action, joint) in enumerate(zip(actions, joints, strict=True)):
        width = span(joint)
        if not math.isfinite(action) or not 0 <= action <= 1:
            raise ValueError(f"Joint {index}: action {action} is outside [0, 1]; refusing to clip.")
        offset = round(action * width)
        current_offset = (joint.position - joint.minimum) % 4096
        result.append(((joint.minimum + offset) % 4096, offset - current_offset))
    return result


def goals(actions: Sequence[float], joints: Sequence[Joint], max_delta: int) -> list[int]:
    if not 1 <= max_delta <= MAX_DELTA_TICKS:
        raise ValueError(f"The movement limit must be between 1 and {MAX_DELTA_TICKS} servo ticks.")
    preview = goal_preview(actions, joints)
    for index, (_, delta) in enumerate(preview):
        if abs(delta) > max_delta:
            raise ValueError(f"Joint {index}: requested movement exceeds {max_delta} ticks.")
    return [target for target, _ in preview]


def verify_station_config(config: dict, bus: str, queue: str) -> None:
    entries = [item for item in config.get("inference", []) if item.get("queue-id") == queue]
    if len(entries) != 1 or entries[0].get("format") != "normvla":
        raise ValueError("Station config must contain exactly one matching normvla inference queue.")
    if entries[0].get("st3215-bus") != bus:
        raise ValueError(
            "Pin the inference queue's st3215-bus to the follower serial and restart Station. "
            "'auto' cannot establish which bus produced the observations."
        )


def ensure_fresh(frame_ns: int, sample_ns: Sequence[int], now_ns: int, max_age: float) -> None:
    if not sample_ns or max_age <= 0:
        raise ValueError("Missing timestamps or invalid freshness limit.")
    for index, stamp in enumerate([frame_ns, *sample_ns]):
        age = (now_ns - stamp) / 1e9
        if stamp <= 0 or not 0 <= age <= max_age:
            label = "frame" if index == 0 else f"sensor {index - 1}"
            raise ValueError(
                f"Station {label} age is {age:.3f}s (allowed 0..{max_age:.3f}s). "
                "No commands sent. Check that the follower and both cameras are producing "
                "fresh inference frames; Station skips frames when all follower torque is disabled. "
                "Only same-host WSL Station is supported."
            )
