import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

import follow
from safety import Joint


class PredictionTests(unittest.TestCase):
    def test_freshness_uses_linux_station_boottime_clock(self):
        sensor = SimpleNamespace(get_monotonic_stamp_ns=lambda: 9_900_000_000)
        frame = SimpleNamespace(
            get_monotonic_stamp_ns=lambda: 9_900_000_000,
            get_joints=lambda: [sensor] * 8,
            get_images=lambda: [sensor] * 2,
        )
        with patch("follow.time.clock_gettime_ns", return_value=10_000_000_000) as clock:
            follow.check_frame_age(frame)
        clock.assert_called_once_with(time.CLOCK_BOOTTIME)

    def test_prediction_has_eight_values_and_resets_chunk_cache(self):
        policy = Mock()
        policy.select_action.return_value = torch.full((1, 8), 0.5)
        self.assertEqual(follow.predict(policy, {}), [0.5] * 8)
        policy.reset.assert_called_once()

    def test_prediction_rejects_invalid_output(self):
        for tensor in (torch.zeros(1, 6), torch.full((1, 8), float("nan")), torch.ones(1, 8) * 1.1):
            policy = Mock()
            policy.select_action.return_value = tensor
            with self.subTest(shape=tensor.shape), self.assertRaises(ValueError):
                follow.predict(policy, {})


    def test_offline_prediction_reports_out_of_range_without_raising(self):
        policy = Mock()
        policy.select_action.return_value = torch.full((1, 8), -0.05)
        self.assertEqual(len(follow.predict(policy, {}, strict=False)), 8)

    def test_non_strict_prediction_still_rejects_nan(self):
        policy = Mock()
        policy.select_action.return_value = torch.full((1, 8), float("nan"))
        with self.assertRaises(ValueError):
            follow.predict(policy, {}, strict=False)

    def test_sample_indices_are_evenly_spread(self):
        self.assertEqual(follow.sample_indices(2293, 5), [0, 573, 1146, 1719, 2292])
        self.assertEqual(follow.sample_indices(3, 10), [0, 1, 2])
        self.assertEqual(follow.sample_indices(10, 1), [0])


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = root / "station.yaml"
        self.config.write_text(
            "inference:\n- queue-id: inference/normvla\n  format: normvla\n  st3215-bus: FOLLOWER\n"
        )
        generated = root / "target/gen_python/protobuf/drivers/inferences/normvla.py"
        generated.parent.mkdir(parents=True)
        generated.touch()
        self.args = SimpleNamespace(
            server="localhost", bus_serial="FOLLOWER", motor_ids="1,2,3,4,5,6,7,8",
            camera_order="0,1", station_config=self.config, queue="inference/normvla",
            execute=False, norma_core=root, task="spreading cheese on pizza", max_delta_ticks=100,
        )
        self.sender = AsyncMock()
        self.client = Mock()

        def read(*args, **kwargs):
            data = asyncio.Queue()
            data.put_nowait(SimpleNamespace(Data=b"test-frame"))
            return SimpleNamespace(data=data, err=None)

        self.client.read_from_tail.side_effect = read
        stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        sensor = SimpleNamespace(get_monotonic_stamp_ns=lambda: stamp)
        self.frame = SimpleNamespace(
            get_monotonic_stamp_ns=lambda: stamp,
            get_joints=lambda: [sensor] * 8,
            get_images=lambda: [sensor] * 2,
        )
        self.joints = [Joint(2000, 0.5, 1000, 3000)] * 8
        modules = {
            "station_py": SimpleNamespace(
                new_station_client=AsyncMock(return_value=self.client), send_commands=self.sender,
            ),
            "target.gen_python.protobuf.drivers.inferences": SimpleNamespace(
                normvla=SimpleNamespace(FrameReader=lambda data: self.frame),
            ),
            "target.gen_python.protobuf.drivers.st3215": SimpleNamespace(
                st3215=SimpleNamespace(
                    ST3215SyncWriteCommand=lambda **kwargs: SimpleNamespace(**kwargs),
                    ST3215SyncWriteCommand_MotorWrite=lambda **kwargs: SimpleNamespace(**kwargs),
                    Command=lambda **kwargs: SimpleNamespace(encode=lambda: b"command"),
                ),
            ),
            "target.gen_python.protobuf.station": SimpleNamespace(
                commands=SimpleNamespace(DriverCommand=lambda **kwargs: SimpleNamespace(**kwargs)),
                drivers=SimpleNamespace(StationCommandType=SimpleNamespace(STC_ST3215_COMMAND=1)),
            ),
        }
        self.enterContext(patch.dict("sys.modules", modules))
        self.enterContext(patch.object(follow.sys, "path", follow.sys.path.copy()))
        self.decoder = self.enterContext(patch("follow.decode_frame", return_value=({}, self.joints)))
        self.prediction = self.enterContext(patch("follow.predict", return_value=[0.5] * 8))

    async def test_default_live_path_never_sends(self):
        await follow.live(self.args, Mock())
        self.sender.assert_not_awaited()
        self.client.close_connection.assert_called_once()

    async def test_out_of_range_dry_run_reports_without_converting_or_sending(self):
        self.prediction.return_value = [0.5, -0.0084, 0.5, 1.0075, 0.5, 0.5, 0.5, 0.5]
        with patch("builtins.print") as output, patch("follow.goal_preview") as preview:
            await follow.live(self.args, Mock())
        lines = [str(call.args[0]) for call in output.call_args_list]
        self.assertEqual(sum("OUT OF RANGE" in line for line in lines), 2)
        self.assertTrue(any("BLOCKED for execution" in line for line in lines))
        self.assertTrue(any("DRY RUN: no commands sent" in line for line in lines))
        self.assertFalse(self.prediction.call_args.kwargs["strict"])
        preview.assert_not_called()
        self.sender.assert_not_awaited()
        self.client.close_connection.assert_called_once()

    async def test_execution_keeps_strict_prediction_validation(self):
        self.args.execute = True
        self.prediction.side_effect = ValueError("Predicted normalized goals outside [0, 1]")
        with patch("builtins.input", return_value="MOVE FOLLOWER"), self.assertRaises(ValueError):
            await follow.live(self.args, Mock())
        self.assertTrue(self.prediction.call_args.kwargs["strict"])
        self.sender.assert_not_awaited()
        self.client.close_connection.assert_called_once()

    async def test_over_limit_dry_run_prints_all_motors_without_sending(self):
        self.prediction.return_value = [0.9] * 8
        with patch("builtins.print") as output:
            await follow.live(self.args, Mock())
        lines = [str(call.args[0]) for call in output.call_args_list]
        self.assertEqual(sum("OVER LIMIT" in line for line in lines), 8)
        self.assertTrue(any("BLOCKED for execution" in line for line in lines))
        self.sender.assert_not_awaited()

    async def test_slow_dry_run_refreshes_telemetry_without_sending(self):
        now = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        with patch("follow.time.clock_gettime_ns", side_effect=[now, now + 3_000_000_000]):
            self.frame.get_monotonic_stamp_ns = Mock(side_effect=[now, now + 3_000_000_000])
            sensor = SimpleNamespace(get_monotonic_stamp_ns=Mock(side_effect=[
                *([now] * 10), *([now + 3_000_000_000] * 10),
            ]))
            self.frame.get_joints = lambda: [sensor] * 8
            self.frame.get_images = lambda: [sensor] * 2
            await follow.live(self.args, Mock())
        self.assertEqual(self.client.read_from_tail.call_count, 2)
        self.sender.assert_not_awaited()

    async def test_slow_execution_still_rejects_old_prediction_input(self):
        self.args.execute = True
        now = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        with patch("builtins.input", return_value="MOVE FOLLOWER"), \
                patch("follow.time.clock_gettime_ns", side_effect=[now, now + 3_000_000_000]), \
                self.assertRaises(ValueError):
            await follow.live(self.args, Mock())
        self.sender.assert_not_awaited()

    async def test_execution_requires_confirmation(self):
        self.args.execute = True
        with patch("builtins.input", return_value="no"), self.assertRaises(ValueError):
            await follow.live(self.args, Mock())
        self.sender.assert_not_awaited()

    async def test_confirmed_execution_sends_exactly_one_pack(self):
        self.args.execute = True
        with patch("builtins.input", return_value="MOVE FOLLOWER"):
            await follow.live(self.args, Mock())
        self.sender.assert_awaited_once()
        self.assertEqual(len(self.sender.call_args.args[1]), 1)
        self.assertEqual(self.client.read_from_tail.call_count, 2)

    async def test_stale_telemetry_never_sends(self):
        self.frame.get_monotonic_stamp_ns = lambda: 1
        with self.assertRaises(ValueError):
            await follow.live(self.args, Mock())
        self.sender.assert_not_awaited()

    async def test_large_movement_never_sends(self):
        self.args.execute = True
        self.prediction.return_value = [0.9] * 8
        with patch("builtins.input", return_value="MOVE FOLLOWER"), self.assertRaises(ValueError):
            await follow.live(self.args, Mock())
        self.sender.assert_not_awaited()

    async def test_calibration_change_never_sends(self):
        self.args.execute = True
        other = [Joint(2000, 0.5, 900, 3100)] * 8
        self.decoder.side_effect = [({}, self.joints), ({}, other)]
        with patch("builtins.input", return_value="MOVE FOLLOWER"), self.assertRaises(ValueError):
            await follow.live(self.args, Mock())
        self.sender.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
