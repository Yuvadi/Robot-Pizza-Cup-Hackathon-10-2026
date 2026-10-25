import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

import kitchen_free


def pizza(*quadrants):
    """Pizza view with a topping-coloured square in each given quadrant (tl/tr/br/bl layout)."""
    image = Image.new("RGB", (40, 40), (220, 170, 90))
    spots = {1: (2, 2), 2: (22, 2), 3: (22, 22), 4: (2, 22)}
    for q in quadrants:
        x, y = spots[q]
        image.paste((150, 20, 20), (x, y, x + 15, y + 15))
    return [image, Image.new("RGB", (40, 40))]


class FreeKitchenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.args = SimpleNamespace(
            camera_order="1,0", motor_ids="1,2,3,4,5,6,7,8", bus_serial="BUS", work=Path(self.tmp.name),
            task_template="pick-n-put-{topping}", segment_seconds=5, move_ticks=200, still_ticks=25,
            settle_seconds=2, max_delta_ticks=1500, control_fps=10, verify_pick=0.02, empty_match=0.03,
            verify_change=0.03)
        self.sleep = AsyncMock()
        self.k = kitchen_free.FreeKitchen(self.args, Mock(config=SimpleNamespace(image_features={})), None,
                                          open_station=AsyncMock(), sleep=self.sleep)
        self.k.layout.update({"camera": 0, "box": [0, 0, 1, 1],
                              "quadrants": {"1": "tl", "2": "tr", "3": "br", "4": "bl"}})
        self.k.warmup = AsyncMock()
        self.enterContext(patch("builtins.print"))

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def cook(self, order, frames, rollouts):
        self.k.snapshot = AsyncMock(side_effect=frames)
        rollout = AsyncMock(side_effect=rollouts)
        with patch("follow.rollout", rollout):
            self.k.start(order, "COOK BUS")
            await self.k.task
        return rollout

    async def test_tasks_have_no_quadrant_and_landing_is_measured(self):
        rollout = await self.cook({"q1": {"cheese": 1}, "q3": {"salami": 1}},
                                  [pizza(), pizza(1), pizza(1), pizza(1, 2)], ["settled", "settled"])
        self.assertEqual([c.args[0].task for c in rollout.await_args_list], ["pick-n-put-cheese", "pick-n-put-salami"])
        first, second = (s.verification for s in self.k.steps)
        self.assertEqual((first["landed"], first["on_target"]), (1, True))
        self.assertEqual((second["landed"], second["on_target"]), (2, False))
        self.assertEqual(self.k.status, "finished")
        tally = self.k.state()["tally"]
        self.assertEqual((tally["on_target"], tally["off_target"]), (1, 1))
        self.assertEqual(tally["landed"]["q2"], {"salami": 1})

    async def test_retries_only_when_nothing_landed(self):
        await self.cook({"q4": {"cheese": 1}}, [pizza(), pizza(), pizza(4)], ["settled", "settled"])
        step = self.k.steps[0]
        self.assertEqual((step.attempts, step.verification["landed"]), (2, 4))
        self.assertEqual(self.k.tally["on_target"], 1)

    def test_uncalibrated_pizza_is_not_verifiable(self):
        self.k.layout["box"] = None
        self.assertFalse(self.k.verifiable())



if __name__ == "__main__":
    unittest.main()
