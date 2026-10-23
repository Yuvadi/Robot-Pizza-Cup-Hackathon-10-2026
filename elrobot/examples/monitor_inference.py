#!/usr/bin/env python3
"""Print ST3215 inference frames from a running Station (localhost:8888).

Prerequisites:
  bash scripts/install.sh && bash scripts/run.sh    # in another terminal
  bash scripts/install.sh
  bash scripts/python_env.sh examples/monitor_inference.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from station_py import new_station_client
from target.gen_python.protobuf.drivers.st3215 import st3215

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    entries_queue: asyncio.Queue = asyncio.Queue()
    client = await new_station_client("localhost", logger)
    error_queue = client.follow("st3215/inference", entries_queue)
    logger.info("Following st3215/inference (Ctrl+C to stop)")

    while True:
        if not error_queue.empty():
            err = error_queue.get_nowait()
            raise RuntimeError(f"follow error: {err}")

        try:
            entry = await asyncio.wait_for(entries_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        if entry is None:
            logger.info("stream ended")
            return

        state = st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
        buses = state.get_buses() or []
        print(f"--- {time.strftime('%H:%M:%S')} buses={len(buses)} ---")
        for bus_state in buses:
            bus_info = bus_state.get_bus()
            if not bus_info:
                continue
            serial = bus_info.get_serial_number()
            motors = bus_state.get_motors() or []
            print(f"  bus {serial}: {len(motors)} motors")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
