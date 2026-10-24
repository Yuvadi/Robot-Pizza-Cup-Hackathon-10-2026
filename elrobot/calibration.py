"""Guide manual Station calibration and save verified, serial-bound profiles.

This tool only reads Station. Calibration/reset/freeze actions are performed
explicitly in Station's UI; no torque or position commands are sent here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TORQUE = 0x28
STATUS = 0x41


def validate_bus(bus: Any, serial: str, ids: list[int], now_ns: int) -> list[dict[str, Any]]:
    info = bus.get_bus()
    if info is None or info.get_serial_number() != serial:
        raise ValueError(f"Expected bus {serial}.")
    motors = sorted(bus.get_motors() or [], key=lambda motor: motor.get_id())
    if [motor.get_id() for motor in motors] != ids:
        raise ValueError(f"{serial}: expected exactly motor IDs {ids}.")
    result = []
    for motor in motors:
        mid = motor.get_id()
        age = (now_ns - motor.get_monotonic_stamp_ns()) / 1e9
        if not 0 <= age <= 2:
            raise ValueError(f"{serial} motor {mid}: stale/future telemetry, age={age:.3f}s.")
        minimum, maximum = motor.get_range_min(), motor.get_range_max()
        if not motor.get_range_freezed():
            raise ValueError(f"{serial} motor {mid}: calibration is not frozen.")
        if not 0 <= minimum < 4096 or not 0 <= maximum < 4096 or minimum == maximum:
            raise ValueError(f"{serial} motor {mid}: invalid calibration endpoints.")
        state = bytes(motor.get_state())
        if len(state) <= STATUS:
            raise ValueError(f"{serial} motor {mid}: incomplete register telemetry.")
        if state[TORQUE] != 0:
            raise ValueError(f"{serial} motor {mid}: torque must be OFF for this manual workflow.")
        if state[STATUS] != 0:
            raise ValueError(f"{serial} motor {mid}: servo error status={state[STATUS]}.")
        result.append({
            "motor_id": mid,
            "range_min": minimum,
            "range_max": maximum,
            "range_frozen": True,
            "arc_ticks": (maximum - minimum) % 4096,
            "register_snapshot_hex": state.hex(),
        })
    return result


def write_new_json(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Link a complete temporary file into place without replacing an old profile.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(profile, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


async def read_profile(client: Any, st3215: Any, role: str, serial: str, ids: list[int]) -> dict[str, Any]:
    reader = client.read_from_tail("st3215/inference", offset=b"\x00", limit=1, step=1, buf_size=1)
    entry = await asyncio.wait_for(reader.data.get(), timeout=10)
    if entry is None:
        raise RuntimeError(f"Station returned no motor telemetry: {reader.err}")
    state = st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
    buses = [
        bus for bus in state.get_buses() or []
        if bus.get_bus() is not None and bus.get_bus().get_serial_number() == serial
    ]
    if len(buses) != 1:
        raise ValueError(f"Expected exactly one connected bus for {serial}; found {len(buses)}.")
    motors = validate_bus(buses[0], serial, ids, time.clock_gettime_ns(time.CLOCK_BOOTTIME))
    return {
        "schema_version": 1,
        "robot_type": "elrobot-st3215",
        "role": role,
        "bus_serial": serial,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "motor_order": ids,
        "motors": motors,
        "restore_supported": False,
        "note": "Verified frozen Station ranges and observed register bytes, NOT a complete EEPROM dump. "
                "Keep Station station_data and motor EEPROM. Do not replay freeze or register bytes to restore.",
    }


async def run(args: argparse.Namespace) -> None:
    if not sys.platform.startswith("linux") or args.server not in ("localhost", "127.0.0.1"):
        raise ValueError("Use this tool inside the same WSL/Linux instance as Station.")
    ids = [int(value) for value in args.motor_ids.split(",")]
    if len(ids) != 8 or ids != sorted(set(ids)) or any(not 1 <= mid <= 253 for mid in ids):
        raise ValueError("--motor-ids must contain eight distinct ascending IDs in 1..253.")
    if args.leader_serial == args.follower_serial:
        raise ValueError("Leader and follower must have different serials.")
    for serial in (args.leader_serial, args.follower_serial):
        if not serial or not serial.isascii() or not serial.isalnum():
            raise ValueError("Bus serials must be nonempty ASCII alphanumeric identifiers.")
    norma = args.norma_core.resolve()
    if not (norma / "target/gen_python/protobuf/drivers/st3215/st3215.py").is_file():
        raise FileNotFoundError(f"Station Python modules missing under {norma}.")
    sys.path[:0] = [str(norma), str(norma / "software/station/shared")]
    from station_py import new_station_client
    from target.gen_python.protobuf.drivers.st3215 import st3215

    session = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session.mkdir(parents=True, exist_ok=False)
    print(f"Profiles will be saved in {session}")
    print("Stop teleoperation and all policies. Support the arms before disabling torque.")
    print("This program sends NO hardware commands; do not force joints or drive them into stops.")
    roles = [
        (role, serial) for role, serial in
        (("leader", args.leader_serial), ("follower", args.follower_serial))
        if args.role in ("both", role)
    ]
    for role, serial in roles:
        if not args.snapshot_only:
            print(f"\n{role.upper()}: select bus {serial} at http://localhost:8889.")
            print("1. Confirm all eight motor IDs and disable torque with the arm supported.")
            print("2. Opening Station's calibration page resets THIS bus's calibration automatically.")
            print("   Back up existing frozen calibration BEFORE opening it, using --snapshot-only.")
            print("3. Gently move each joint through its intended safe range; never force mechanical stops.")
            print("4. Use Save/Done on Station's calibration page to freeze (not Web-controlled).")
            print("   Freeze writes calibration settings to motor EEPROM. Inspect Station for write errors.")
            print("5. Leave torque OFF and wait for fresh frozen ranges to appear.")
            answer = await asyncio.to_thread(input, f"Type SAVE {serial} when complete (anything else cancels): ")
            if answer.strip() != f"SAVE {serial}":
                raise ValueError(f"Cancelled; any previously saved profiles remain in {session}.")
        client = await new_station_client(args.server, logging.getLogger("calibration"))
        try:
            profile = await read_profile(client, st3215, role, serial, ids)
        finally:
            client.close_connection()
        path = session / f"{role}-{serial}.json"
        write_new_json(path, profile)
        print(f"Saved and verified {role}: {path}", flush=True)
    print(f"Saved {len(roles)} profile(s). These are not automatic restore files; see docs/calibration.md.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-serial", required=True)
    parser.add_argument("--follower-serial", required=True)
    parser.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--norma-core", type=Path, default=ROOT / "norma-core")
    parser.add_argument("--output", type=Path, default=ROOT / "calibrations")
    parser.add_argument("--snapshot-only", action="store_true", help="Save existing frozen calibration without UI guidance.")
    parser.add_argument("--role", choices=("both", "leader", "follower"), default="both",
                        help="Select one arm to retry saving without repeating the other.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
