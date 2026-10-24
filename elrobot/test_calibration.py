import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from calibration import read_profile, validate_bus, write_new_json


def make_bus():
    motors = []
    for mid in range(1, 9):
        state = bytearray(0x42)
        motors.append(SimpleNamespace(
            get_id=lambda mid=mid: mid,
            get_monotonic_stamp_ns=lambda: 9_900_000_000,
            get_range_min=lambda: 1000,
            get_range_max=lambda: 3000,
            get_range_freezed=lambda: True,
            get_state=lambda state=state: state,
        ))
    return SimpleNamespace(
        get_bus=lambda: SimpleNamespace(get_serial_number=lambda: "SERIAL"),
        get_motors=lambda: motors,
    )


class CalibrationTests(unittest.TestCase):
    def test_eight_serial_bound_frozen_motors(self):
        result = validate_bus(make_bus(), "SERIAL", list(range(1, 9)), 10_000_000_000)
        self.assertEqual([m["motor_id"] for m in result], list(range(1, 9)))
        self.assertTrue(all(m["arc_ticks"] == 2000 for m in result))

    def test_wrong_bus_missing_motor_stale_unfrozen_torque_and_errors_fail(self):
        changes = [
            lambda bus: setattr(bus, "get_bus", lambda: SimpleNamespace(get_serial_number=lambda: "OTHER")),
            lambda bus: bus.get_motors().pop(),
            lambda bus: setattr(bus.get_motors()[0], "get_monotonic_stamp_ns", lambda: 1),
            lambda bus: setattr(bus.get_motors()[0], "get_range_freezed", lambda: False),
            lambda bus: setattr(bus.get_motors()[0], "get_range_max", lambda: 1000),
            lambda bus: bus.get_motors()[0].get_state().__setitem__(0x28, 1),
            lambda bus: bus.get_motors()[0].get_state().__setitem__(0x41, 1),
            lambda bus: setattr(bus.get_motors()[0], "get_state", lambda: b""),
        ]
        for index, change in enumerate(changes):
            bus = make_bus()
            change(bus)
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_bus(bus, "SERIAL", list(range(1, 9)), 10_000_000_000)

    def test_profile_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            write_new_json(path, {"schema_version": 1})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"schema_version": 2})
            self.assertEqual(json.loads(path.read_text()), {"schema_version": 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])


class ProfileTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_serial_specific_profile_without_any_enqueue(self):
        channel = asyncio.Queue()
        channel.put_nowait(SimpleNamespace(Data=b"serialized-state"))
        client = Mock()
        client.read_from_tail.return_value = SimpleNamespace(data=channel, err=None)
        protocol = SimpleNamespace(
            InferenceStateReader=lambda data: SimpleNamespace(get_buses=lambda: [make_bus()])
        )
        with patch("calibration.time.clock_gettime_ns", return_value=10_000_000_000):
            profile = await read_profile(client, protocol, "leader", "SERIAL", list(range(1, 9)))
        self.assertEqual(profile["bus_serial"], "SERIAL")
        self.assertEqual(profile["role"], "leader")
        self.assertEqual(len(profile["motors"]), 8)
        self.assertFalse(profile["restore_supported"])
        client.enqueue.assert_not_called()

    async def test_missing_bus_fails_instead_of_saving_success_profile(self):
        channel = asyncio.Queue()
        channel.put_nowait(SimpleNamespace(Data=b"serialized-state"))
        client = Mock()
        client.read_from_tail.return_value = SimpleNamespace(data=channel, err=None)
        protocol = SimpleNamespace(
            InferenceStateReader=lambda data: SimpleNamespace(get_buses=lambda: [])
        )
        with self.assertRaisesRegex(ValueError, "found 0"):
            await read_profile(client, protocol, "follower", "SERIAL", list(range(1, 9)))
        client.enqueue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
