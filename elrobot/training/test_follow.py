import asyncio
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

import follow
from safety import Joint


class PredictionTests(unittest.TestCase):
    def test_rollout_cli_accepts_long_durations_without_loading_policy(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "config.json").touch()
            for duration in ("100", "150", "1000"):
                with self.subTest(duration=duration), \
                        patch("sys.argv", ["follow.py", "--checkpoint", root, "--execute",
                                           "--rollout-seconds", duration]), \
                        patch("follow.torch.cuda.is_available", return_value=False), \
                        self.assertRaisesRegex(RuntimeError, "CUDA is required"):
                    follow.main()

    def test_rollout_cli_rejects_invalid_durations(self):
        for duration in ("nan", "inf", "-inf", "-1"):
            with self.subTest(duration=duration), \
                    patch("sys.argv", ["follow.py", "--checkpoint", ".", "--execute",
                                       f"--rollout-seconds={duration}"]), \
                    patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                follow.main()
            self.assertEqual(error.exception.code, 2)

    def test_retry_cli_rejects_invalid_limits_and_nonrollout_modes(self):
        for options in (
            ["--max-invalid-retries", "-1"],
            ["--execute", "--rollout-seconds", "20", "--max-invalid-retries", "4"],
            ["--execute", "--max-invalid-retries", "1"],
            ["--offline", "--max-invalid-retries", "1"],
        ):
            with self.subTest(options=options), \
                    patch("sys.argv", ["follow.py", "--checkpoint", ".", *options]), \
                    patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                follow.main()
            self.assertEqual(error.exception.code, 2)

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

    def test_rollout_prediction_preserves_chunk_cache(self):
        policy = Mock()
        policy.select_action.return_value = torch.full((1, 8), 0.5)
        self.assertEqual(follow.predict(policy, {}, reset=False), [0.5] * 8)
        policy.reset.assert_not_called()

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

    def test_rollout_limit_diagnostics_separate_target_jump_from_tracking(self):
        joints = [Joint(2000, .5, 1000, 3000)] * 8
        with patch("builtins.print") as output, self.assertRaises(ValueError):
            follow.rollout_goals([.65] * 8, joints, 200, [2050] * 8,
                                 list(range(1, 9)), "chunk_start=True")
        lines = [str(call.args[0]) for call in output.call_args_list]
        self.assertIn("1 2000 2300 +300 2050 250 50", lines)
        self.assertTrue(any("chunk_start=True" in line for line in lines))


class WarmupTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_does_not_block_network_event_loop(self):
        policy = Mock()
        entered = threading.Event()
        release = threading.Event()

        def prediction(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("Test did not release inference worker.")

        with patch("follow.predict", side_effect=prediction):
            task = asyncio.create_task(follow.warmup_policy(policy, {}))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                await asyncio.sleep(.01)
                self.assertFalse(task.done())
                policy.reset.assert_not_called()
            finally:
                release.set()
                await task
        policy.reset.assert_called_once()

    async def test_cancellation_waits_before_resetting_policy(self):
        policy = Mock()
        entered = threading.Event()
        release = threading.Event()

        def prediction(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("Test did not release inference worker.")

        with patch("follow.predict", side_effect=prediction):
            task = asyncio.create_task(follow.warmup_policy(policy, {}))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                task.cancel()
                await asyncio.sleep(.01)
                policy.reset.assert_not_called()
                self.assertFalse(task.done())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        policy.reset.assert_called_once()


class RolloutTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.args = SimpleNamespace(execute=True, rollout_seconds=.25, camera_order="0,1",
                                    motor_ids="1,2,3,4,5,6,7,8",
                                    task="place cheese on pizza", max_delta_ticks=100)
        self.policy = Mock(config=SimpleNamespace(n_action_steps=50))
        self.policy.select_action.return_value = torch.full((1, 8), 0.5)
        self.send = AsyncMock()
        self.stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        self.joints = [Joint(2000, .5, 1000, 3000)] * 8
        self.decoder = self.enterContext(patch("follow.decode_frame", return_value=({}, self.joints)))
        self.age = self.enterContext(patch("follow.check_frame_age"))

        async def fetch():
            self.stamp += 100_000_000
            stamp = self.stamp
            return SimpleNamespace(get_monotonic_stamp_ns=lambda: stamp)

        self.fetch = AsyncMock(side_effect=fetch)

    async def test_duration_caps_sends_and_preserves_chunk_state(self):
        await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.assertGreaterEqual(self.send.await_count, 1)
        self.assertLessEqual(self.send.await_count, 3)
        self.assertEqual(self.policy.reset.call_count, 2)
        self.assertEqual(self.policy.config.n_action_steps, 50)
        self.assertEqual(self.policy.select_action.call_count, self.send.await_count)

    async def test_full_checkpoint_horizon_is_consumed_before_replanning(self):
        for horizon in (50, 100):
            with self.subTest(horizon=horizon):
                queue = deque()
                plans = []
                calls = 0

                def reset():
                    queue.clear()

                def select(batch):
                    nonlocal calls
                    if calls == horizon + 1:
                        raise asyncio.CancelledError()
                    if not queue:
                        plans.append(calls)
                        queue.extend(torch.full((1, 8), .5 + i / 10000)
                                     for i in range(self.policy.config.n_action_steps))
                    calls += 1
                    return queue.popleft()

                self.policy = Mock(config=SimpleNamespace(n_action_steps=horizon))
                self.policy.reset.side_effect = reset
                self.policy.select_action.side_effect = select
                self.send.reset_mock()
                self.args.rollout_seconds = 1000
                with patch("follow.asyncio.sleep", new_callable=AsyncMock), \
                        patch("builtins.print"), self.assertRaises(asyncio.CancelledError):
                    await follow.rollout(self.args, self.policy, self.fetch, self.send)
                self.assertEqual(plans, [0, horizon])
                self.assertEqual(self.send.await_count, horizon + 1)
                self.assertEqual(self.policy.reset.call_count, 2)
                self.assertEqual(self.policy.config.n_action_steps, horizon)
                self.assertNotEqual(self.send.await_args_list[0].args[0],
                                    self.send.await_args_list[horizon - 1].args[0])
                self.assertEqual(self.send.await_args_list[0].args[0],
                                 self.send.await_args_list[horizon].args[0])

    async def test_invalid_action_horizon_never_fetches_or_sends(self):
        for horizon in (0, -1, 1.5, None):
            self.policy.config.n_action_steps = horizon
            with self.subTest(horizon=horizon), \
                    self.assertRaisesRegex(ValueError, "positive integer"):
                await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.fetch.assert_not_awaited()
        self.send.assert_not_awaited()

    async def test_long_duration_preserves_deadline(self):
        for duration in (100, 150):
            with self.subTest(duration=duration):
                self.args.rollout_seconds = duration
                self.send.reset_mock()
                with patch("follow.time.monotonic",
                           side_effect=[0, 0, 0, 0, 0, duration]):
                    await follow.rollout(self.args, self.policy, self.fetch, self.send)
                self.send.assert_not_awaited()
                self.assertEqual(self.policy.config.n_action_steps, 50)

    async def test_invalid_duration_never_fetches_or_sends(self):
        for duration in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(duration=duration):
                self.args.rollout_seconds = duration
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    await follow.rollout(self.args, self.policy, self.fetch, self.send)
                self.fetch.assert_not_awaited()
                self.send.assert_not_awaited()

    async def test_invalid_prediction_never_sends(self):
        self.policy.select_action.return_value[0, 1] = -.01
        with self.assertRaises(ValueError):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.send.assert_not_awaited()
        self.assertEqual(self.policy.config.n_action_steps, 50)

    async def test_retry_discards_chunk_refetches_and_resets_consecutive_limit(self):
        self.args.max_invalid_retries = 1
        self.args.rollout_seconds = 1
        invalid = torch.full((1, 8), 0.5)
        invalid[0, 1] = -.01
        valid = torch.full((1, 8), 0.5)

        def first_valid():
            self.assertEqual(self.fetch.await_count, 2)
            self.assertEqual(self.policy.reset.call_count, 2)
            self.send.assert_not_awaited()
            return valid

        outputs = iter([invalid, first_valid, invalid, valid, asyncio.CancelledError()])

        def select(batch):
            result = next(outputs)
            if isinstance(result, BaseException):
                raise result
            return result() if callable(result) else result

        self.policy.select_action.side_effect = select
        with patch("follow.asyncio.sleep", new_callable=AsyncMock) as sleep, \
                patch("builtins.print") as output, self.assertRaises(asyncio.CancelledError):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.assertEqual(self.send.await_count, 2)
        self.assertEqual(self.policy.reset.call_count, 4)
        self.assertEqual(self.policy.config.n_action_steps, 50)
        self.assertTrue(all(call.args[0] >= .1 for call in sleep.await_args_list))
        lines = [str(call.args[0]) for call in output.call_args_list]
        self.assertTrue(any("prediction=3, chunk_start=False" in line for line in lines))
        self.assertTrue(any("prediction=4, chunk_start=True" in line for line in lines))

    async def test_retry_limit_aborts_without_sending(self):
        self.args.max_invalid_retries = 3
        self.args.rollout_seconds = 1
        self.policy.select_action.return_value = torch.full((1, 8), 1.01)
        with patch("follow.asyncio.sleep", new_callable=AsyncMock) as sleep, \
                self.assertRaisesRegex(ValueError, "retry limit exceeded"):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.assertEqual(self.policy.select_action.call_count, 4)
        self.assertEqual(self.fetch.await_count, 4)
        self.assertEqual(sleep.await_count, 3)
        self.send.assert_not_awaited()

    async def test_retry_rejects_nonadvancing_input(self):
        self.args.max_invalid_retries = 3
        self.policy.select_action.return_value = torch.full((1, 8), -.01)
        frame = SimpleNamespace(get_monotonic_stamp_ns=lambda: self.stamp)
        self.fetch.side_effect = None
        self.fetch.return_value = frame
        with patch("follow.asyncio.sleep", new_callable=AsyncMock), \
                self.assertRaisesRegex(ValueError, "did not advance"):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.assertEqual(self.policy.select_action.call_count, 1)
        self.send.assert_not_awaited()

    async def test_retry_does_not_bypass_other_guards(self):
        self.args.max_invalid_retries = 3
        for tensor, message in (
            (torch.full((1, 8), float("nan")), "finite"),
            (torch.full((1, 8), float("inf")), "finite"),
            (torch.zeros(1, 6), "finite"),
            (torch.full((1, 8), .9), "requested movement"),
        ):
            with self.subTest(message=message):
                self.policy.select_action.return_value = tensor
                with patch("follow.asyncio.sleep", new_callable=AsyncMock) as sleep, \
                        self.assertRaisesRegex(ValueError, message):
                    await follow.rollout(self.args, self.policy, self.fetch, self.send)
                sleep.assert_not_awaited()
                self.send.assert_not_awaited()

    async def test_retry_does_not_bypass_staleness_or_deadline(self):
        self.args.max_invalid_retries = 3
        self.policy.select_action.return_value = torch.full((1, 8), -.01)
        self.age.side_effect = ValueError("stale")
        with self.assertRaisesRegex(ValueError, "stale"):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.age.side_effect = None
        with patch("follow.time.monotonic", side_effect=[0, 0, 0, 0, 0, 1, 1]), \
                patch("follow.asyncio.sleep", new_callable=AsyncMock):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.send.assert_not_awaited()

    async def test_large_delta_never_sends(self):
        self.policy.select_action.return_value = torch.full((1, 8), .9)
        with self.assertRaises(ValueError):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.send.assert_not_awaited()

    async def test_stale_prediction_never_sends(self):
        self.age.side_effect = ValueError("stale")
        with patch("builtins.print") as output, self.assertRaises(ValueError):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        lines = [str(call.args[0]) for call in output.call_args_list]
        self.assertTrue(any("inference_s=" in line for line in lines))
        self.send.assert_not_awaited()

    async def test_calibration_change_never_sends(self):
        self.decoder.side_effect = [({}, self.joints), ({}, [Joint(2000, .5, 900, 3100)] * 8)]
        with self.assertRaisesRegex(ValueError, "Calibration"):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.send.assert_not_awaited()

    async def test_expired_deadline_after_inference_never_sends(self):
        with patch("follow.time.monotonic", side_effect=[0, 0, 0, 0, 0, 1]):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.send.assert_not_awaited()

    async def test_duplicate_frame_aborts_after_one_send(self):
        frame = SimpleNamespace(get_monotonic_stamp_ns=lambda: self.stamp)
        self.fetch.side_effect = None
        self.fetch.return_value = frame
        with self.assertRaisesRegex(ValueError, "did not advance"):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.send.assert_awaited_once()

    async def test_cancellation_stops_without_extra_send(self):
        self.send.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await follow.rollout(self.args, self.policy, self.fetch, self.send)
        self.assertEqual(self.send.await_count, 1)
        self.assertEqual(self.policy.reset.call_count, 2)


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

    async def test_rollout_confirmation_is_distinct_and_can_cancel(self):
        self.args.execute = True
        self.args.rollout_seconds = 20
        with patch("builtins.input", return_value="MOVE FOLLOWER"), self.assertRaisesRegex(ValueError, "not confirmed"):
            await follow.live(self.args, Mock())
        self.sender.assert_not_awaited()

    async def test_rollout_warmup_discards_output_and_cache_before_refetch(self):
        self.args.execute = True
        self.args.rollout_seconds = 150
        policy = Mock()

        def warmup(*args, **kwargs):
            self.sender.assert_not_awaited()
            return [1.1] * 8

        self.prediction.side_effect = warmup

        async def run(args, warmed_policy, fetch, send):
            self.assertIs(warmed_policy, policy)
            policy.reset.assert_called_once()
            self.assertEqual(self.client.read_from_tail.call_count, 1)
            await fetch()
            self.assertEqual(self.client.read_from_tail.call_count, 2)
            self.sender.assert_not_awaited()

        with patch("builtins.input", return_value="ROLLOUT FOLLOWER 150"), \
                patch("follow.rollout", side_effect=run) as rollout:
            await follow.live(self.args, policy)
        self.assertFalse(self.prediction.call_args.kwargs["strict"])
        rollout.assert_awaited_once()
        self.sender.assert_not_awaited()
        self.client.close_connection.assert_called_once()

    async def test_failed_warmup_resets_cache_and_never_starts_rollout(self):
        self.args.execute = True
        self.args.rollout_seconds = 20
        policy = Mock()
        self.prediction.side_effect = ValueError("Invalid warm-up prediction")
        with patch("builtins.input", return_value="ROLLOUT FOLLOWER 20"), \
                patch("follow.rollout", new_callable=AsyncMock) as rollout, \
                self.assertRaisesRegex(ValueError, "Invalid warm-up prediction"):
            await follow.live(self.args, policy)
        policy.reset.assert_called_once()
        rollout.assert_not_awaited()
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
