import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

import kitchen
from safety import Joint


def joints_at(offset):
    return [Joint(1000 + offset, offset / 2000, 1000, 3000)] * 8


class RecipeTests(unittest.TestCase):
    def test_cheese_steps_come_before_salami(self):
        steps = kitchen.recipe({"q2": ["salami", "cheese"], "q4": ["cheese"], "q1": ["salami"]},
                               "place {topping} in quadrant {quadrant}")
        self.assertEqual([s.task for s in steps], [
            "place cheese in quadrant 2", "place cheese in quadrant 4",
            "place salami in quadrant 1", "place salami in quadrant 2"])

    def test_counts_mystery_slot_and_piece_order(self):
        steps = kitchen.recipe({"q1": {"cheese": 2, "salami": 1}, "q3": {"mystery": 1, "salami": 0}},
                               "pick-n-put-{topping} put-in-q{quadrant}", mystery_slot="salami")
        self.assertEqual([(s.topping, s.source, s.quadrant, s.piece, s.task) for s in steps], [
            ("cheese", "cheese", 1, 1, "pick-n-put-cheese put-in-q1"),
            ("cheese", "cheese", 1, 2, "pick-n-put-cheese put-in-q1"),
            ("mystery", "salami", 3, 1, "pick-n-put-salami put-in-q3"),
            ("salami", "salami", 1, 1, "pick-n-put-salami put-in-q1")])
        self.assertEqual(len(kitchen.recipe({"q2": {"salami": 20}}, "{topping}{quadrant}")), 20)

    def test_rejects_invalid_orders(self):
        for order in ({}, {"q1": []}, {"q5": ["cheese"]}, {"q1": ["ham"]}, {"q1": "cheese"}, [],
                      {"q1": {"cheese": 21}}, {"q1": {"cheese": -1}}, {"q1": {"cheese": True}},
                      {"q1": {"cheese": "2"}}, {"q1": {"ham": 1}}, {"q1": {"cheese": 0}}):
            with self.subTest(order=order), self.assertRaises(ValueError):
                kitchen.recipe(order, "{topping} {quadrant}")
        with self.assertRaises(ValueError):
            kitchen.recipe({"q1": ["mystery"]}, "{topping} {quadrant}", mystery_slot="ham")


class SettleTests(unittest.TestCase):
    def run_poses(self, poses, dt=.1):
        now = [0.0]
        detector = kitchen.SettleDetector(200, 25, 2.0, clock=lambda: now[0])
        results = []
        for pose in poses:
            results.append(detector.update(joints_at(pose)))
            now[0] += dt
        return results

    def test_still_without_moving_never_settles(self):
        self.assertFalse(any(self.run_poses([0] * 60)))

    def test_settles_after_moving_and_holding_still(self):
        results = self.run_poses([0, 300, 600, 400, 100] + [50] * 25)
        self.assertFalse(any(results[:24]))
        self.assertTrue(results[-1])

    def test_wobble_prevents_settling(self):
        self.assertFalse(any(self.run_poses([0, 400] + [50, 100] * 20)))


class VerifierHelperTests(unittest.TestCase):
    def test_crop_uses_quadrant_mapping(self):
        image = Image.new("RGB", (200, 100))
        layout = {"box": [0, 0, 1, 1], "quadrants": {"1": "tr", "2": "tl", "3": "bl", "4": "br"}}
        self.assertEqual(kitchen.crop_quadrant(image, layout, 1).getbbox() or (0, 0, 100, 50), (0, 0, 100, 50))
        self.assertEqual(kitchen.crop_quadrant(image, layout, 1).size, (100, 50))
        self.assertEqual(kitchen.crop_quadrant(image, layout, 3).size, (100, 50))
        with self.assertRaises(ValueError):
            kitchen.crop_quadrant(image, {"box": None}, 1)

    def test_change_score(self):
        black, white = Image.new("RGB", (20, 20)), Image.new("RGB", (20, 20), "white")
        self.assertEqual(kitchen.change_score(black, black), 0)
        self.assertAlmostEqual(kitchen.change_score(black, white), 1, places=3)

    def test_verifier_change_gates_and_vlm_only_in_strict_mode(self):
        layout = {"box": [0, 0, 1, 1], "quadrants": {"1": "tl", "2": "tr", "3": "br", "4": "bl"}}
        before = Image.new("RGB", (40, 40))
        after = before.copy()
        after.paste((200, 30, 30), (0, 0, 20, 20))
        for strict, yes, expected_q1 in ((False, [.3, .3], True), (True, [.3, .3], False), (True, [.3, .5], True)):
            with self.subTest(strict=strict, yes=yes):
                v = kitchen.Verifier(min_gain=.05, min_change=.03, strict=strict)
                v.yes_probability = Mock(side_effect=yes * 2)
                self.assertEqual(v.verify("salami", 1, before, after, layout)["passed"], expected_q1)
                self.assertFalse(v.verify("salami", 3, before, after, layout)["passed"])


class KitchenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.args = SimpleNamespace(
            camera_order="1,0", motor_ids="1,2,3,4,5,6,7,8", bus_serial="BUS", work=Path(self.tmp.name),
            task_template="place {topping} in quadrant {quadrant}", segment_seconds=5, move_ticks=200,
            still_ticks=25, settle_seconds=2, max_delta_ticks=1500, control_fps=10,
            verify_pick=0.02, empty_match=0.03)
        self.policy = Mock(config=SimpleNamespace(image_features={}))
        self.client = Mock()
        self.opened = AsyncMock(return_value=(self.client, AsyncMock(), AsyncMock()))
        self.sleep = AsyncMock()
        self.verifier = Mock()
        self.k = kitchen.Kitchen(self.args, self.policy, self.verifier, open_station=self.opened, sleep=self.sleep)
        self.k.layout = {"camera": 0, "box": [0, 0, 1, 1], "quadrants": {"1": "tl", "2": "tr", "3": "br", "4": "bl"},
                         "bins": {}}
        self.k.snapshot = AsyncMock(return_value=[Image.new("RGB", (8, 8))] * 2)
        self.k.warmup = AsyncMock()
        self.enterContext(patch("builtins.print"))

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_order_requires_typed_authorization(self):
        with self.assertRaisesRegex(ValueError, "COOK BUS"):
            self.k.start({"q1": ["cheese"]}, "yes")
        self.assertIsNone(self.k.task)

    async def test_recovers_from_errors_until_verified(self):
        rollout = AsyncMock(side_effect=[ValueError("Rollout input did not advance"), TimeoutError("no frame"),
                                         "settled", "settled"])
        self.verifier.verify.side_effect = [{"passed": False}, {"passed": True}]
        with patch("follow.rollout", rollout):
            self.k.start({"q3": ["cheese"]}, "COOK BUS")
            await self.k.task
        step = self.k.steps[0]
        self.assertEqual((self.k.status, step.status, step.recoveries, step.attempts), ("finished", "done", 2, 4))
        self.assertEqual(rollout.await_args.args[0].task, "place cheese in quadrant 3")
        self.assertEqual(rollout.await_args.args[0].max_invalid_retries, 20)
        self.assertIsNotNone(rollout.await_args.kwargs["monitor"])
        self.client.close_connection.assert_not_called()  # connection was never opened by the mocked rollout
        self.assertEqual([c.args[0] for c in self.sleep.await_args_list], [2, 4])

    async def test_non_safety_errors_reconnect(self):
        self.k.station = (self.client, AsyncMock(), AsyncMock())
        rollout = AsyncMock(side_effect=[RuntimeError("Station did not return a frame"), "settled"])
        self.verifier.verify.return_value = {"passed": True}
        with patch("follow.rollout", rollout):
            self.k.start({"q1": ["salami"]}, "COOK BUS")
            await self.k.task
        self.client.close_connection.assert_called_once()
        self.assertEqual(self.k.status, "finished")

    async def test_stop_cancels_and_never_restarts(self):
        started = asyncio.Event()

        async def forever(*args, **kwargs):
            started.set()
            await asyncio.sleep(3600)

        with patch("follow.rollout", forever):
            self.k.start({"q1": ["cheese"], "q2": ["cheese"]}, "COOK BUS")
            await started.wait()
            await self.k.stop()
        self.assertEqual(self.k.status, "stopped")
        self.assertEqual([s.status for s in self.k.steps], ["stopped", "pending"])
        with self.assertRaisesRegex(ValueError, "COOK"):
            self.k.start({"q1": ["cheese"]}, "")

    async def test_without_calibration_settle_finishes_unverified(self):
        self.k.layout["box"] = None
        rollout = AsyncMock(side_effect=[None, "settled"])
        with patch("follow.rollout", rollout):
            self.k.start({"q1": ["cheese"]}, "COOK BUS")
            await self.k.task
        self.assertEqual(self.k.steps[0].verification, {"passed": True, "unverified": True})
        self.assertEqual(self.k.steps[0].attempts, 2)
        self.verifier.verify.assert_not_called()

    async def test_layout_validation_and_persistence(self):
        for bad in ({"camera": 2, "box": [0, 0, 1, 1], "quadrants": self.k.layout["quadrants"]},
                    {"camera": 0, "box": [.5, 0, .4, 1], "quadrants": self.k.layout["quadrants"]},
                    {"camera": 0, "box": [0, 0, 1, 1], "quadrants": {"1": "tl", "2": "tl", "3": "br", "4": "bl"}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.k.save_layout(bad)
        self.k.save_layout({"camera": 1, "box": [.1, .2, .8, .9], "quadrants": self.k.layout["quadrants"]})
        reloaded = kitchen.Kitchen(self.args, self.policy, self.verifier, open_station=self.opened)
        self.assertEqual(reloaded.layout["camera"], 1)


    @staticmethod
    def frame(bin_color):
        image = Image.new("RGB", (20, 20), "gray")
        image.paste(bin_color, (10, 0, 20, 10))
        return [Image.new("RGB", (20, 20)), image]

    async def test_bin_pick_check_gates_verification(self):
        self.k.layout["box"] = None
        self.k.save_bin({"slot": "salami", "camera": 1, "box": [.5, 0, 1, .5]})
        full, picked = self.frame((180, 30, 30)), self.frame((60, 60, 60))
        self.k.snapshot = AsyncMock(side_effect=[full, full, picked])
        rollout = AsyncMock(return_value="settled")
        with patch("follow.rollout", rollout):
            self.k.start({"q2": {"mystery": 1}}, "COOK BUS", mystery_slot="salami")
            await self.k.task
        step = self.k.steps[0]
        self.assertEqual(step.attempts, 2)
        self.assertTrue(step.verification["picked"])
        self.assertGreater(step.verification["bin_change"], 0.02)
        self.assertEqual(rollout.await_args.args[0].task, "place salami in quadrant 2")

    async def test_empty_bin_waits_without_commands_until_refilled(self):
        self.k.layout["box"] = None
        self.k.args.verify_pick = 0
        self.k.save_bin({"slot": "cheese", "camera": 1, "box": [.5, 0, 1, .5]})
        empty, full = self.frame((60, 60, 60)), self.frame((250, 230, 90))
        self.k.snapshot = AsyncMock(return_value=empty)
        await self.k.capture_empty("cheese")
        self.assertTrue(self.k.layout["bins"]["cheese"]["empty_ref"])
        self.k.snapshot = AsyncMock(side_effect=[empty, empty, full])
        rollout = AsyncMock(return_value="settled")

        async def no_motion_while_waiting(delay):
            rollout.assert_not_awaited()

        self.sleep.side_effect = no_motion_while_waiting
        with patch("follow.rollout", rollout):
            self.k.start({"q1": ["cheese"]}, "COOK BUS")
            await self.k.task
        self.assertEqual(self.sleep.await_count, 2)
        rollout.assert_awaited_once()
        self.assertTrue(any("EMPTY" in line for line in self.k.log))

    async def test_bin_validation_and_removal(self):
        for bad in ({"slot": "ham", "camera": 0, "box": [0, 0, 1, 1]}, {"slot": "cheese", "camera": 3, "box": [0, 0, 1, 1]},
                    {"slot": "cheese", "camera": 0, "box": [.5, 0, .2, 1]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.k.save_bin(bad)
        self.k.save_bin({"slot": "cheese", "camera": 0, "box": [0, 0, .5, .5]})
        self.assertIn("cheese", kitchen.Kitchen(self.args, self.policy, self.verifier).layout["bins"])
        self.k.save_bin({"slot": "cheese", "box": None})
        self.assertEqual(self.k.layout["bins"], {})
        with self.assertRaises(ValueError):
            await self.k.capture_empty("cheese")


class ColourInspectTests(unittest.IsolatedAsyncioTestCase):
    SAUCE, DOUGH = (200, 30, 30), (230, 200, 150)
    SALAMI, CHEESE = (130, 25, 40), (250, 240, 200)

    def bare(self):
        image = Image.new("RGB", (64, 64), self.DOUGH)
        image.paste(self.SAUCE, (0, 0, 64, 32))
        return image

    def topped(self):
        image = self.bare()
        image.paste(self.SALAMI, (4, 4, 28, 28))     # q1 (tl), on sauce
        image.paste(self.CHEESE, (36, 36, 60, 60))   # q3 (br), on dough
        return image

    def bins(self):
        image = Image.new("RGB", (64, 64), self.CHEESE)
        image.paste(self.SALAMI, (32, 0, 64, 64))
        return image

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        args = SimpleNamespace(camera_order="1,0", motor_ids="1,2,3,4,5,6,7,8", bus_serial="BUS",
                               work=Path(self.tmp.name), task_template="t {topping} {quadrant}",
                               verify_pick=0, empty_match=0.03)
        self.k = kitchen.Kitchen(args, Mock(), None, open_station=AsyncMock(), sleep=AsyncMock())
        self.k.layout = {"camera": 0, "box": [0, 0, 1, 1], "quadrants": {"1": "tl", "2": "tr", "3": "br", "4": "bl"},
                         "bins": {"cheese": {"camera": 1, "box": [0, 0, 0.5, 1]},
                                  "salami": {"camera": 1, "box": [0.5, 0, 1, 1]}}}

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def test_sauce_and_dough_alone_are_not_toppings(self):
        colors = {"cheese": list(self.CHEESE), "salami": list(self.SALAMI)}
        cover = kitchen.topping_coverage(self.bare(), self.bare(), colors)
        self.assertEqual(cover, {"cheese": 0.0, "salami": 0.0})

    def test_sample_color_reads_bin(self):
        self.assertEqual(kitchen.sample_color(Image.new("RGB", (20, 20), self.SALAMI)), list(self.SALAMI))

    async def test_inspect_requires_bare_pizza_reference(self):
        self.k.snapshot = AsyncMock(return_value=[self.topped(), self.bins()])
        with self.assertRaisesRegex(ValueError, "Capture bare pizza"):
            await self.k.inspect()

    async def test_inspect_finds_toppings_not_sauce(self):
        self.k.snapshot = AsyncMock(return_value=[self.bare(), self.bins()])
        await self.k.capture_pizza_ref()
        self.k.snapshot = AsyncMock(return_value=[self.topped(), self.bins()])
        result = (await self.k.inspect())["quadrants"]
        present = {q: [t for t, v in tops.items() if v["present"]] for q, tops in result.items()}
        self.assertEqual(present, {"q1": ["salami"], "q2": [], "q3": ["cheese"], "q4": []})

    async def test_moving_pizza_box_drops_reference(self):
        self.k.snapshot = AsyncMock(return_value=[self.bare(), self.bins()])
        await self.k.capture_pizza_ref()
        self.k.save_layout({"camera": 0, "box": [0.1, 0, 1, 1], "quadrants": self.k.layout["quadrants"]})
        self.assertFalse(self.k.layout["pizza_ref"])
        self.assertFalse(self.k.pizza_ref_path().exists())


if __name__ == "__main__":
    unittest.main()
