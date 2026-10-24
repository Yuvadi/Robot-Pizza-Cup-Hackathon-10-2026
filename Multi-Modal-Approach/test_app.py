import json
import unittest
from types import SimpleNamespace

from app import build_plan
from models import CameraFrame, Ingredient, LayoutConstraints, MotionRequest, PixelPoint, Pizza, Placement, RobotPoint
from recipe_parser import parse_recipe
from robot_executor import RobotExecutor


class SkeletonTests(unittest.TestCase):
    def test_recipe_is_extensible_without_hardcoded_toppings(self):
        result = parse_recipe('{"items":[{"ingredient":"mushroom","count":3}]}')
        self.assertEqual((result[0].ingredient, result[0].count), ("mushroom", 3))

    def test_bad_recipes_rejected(self):
        for value in ({}, {"items": []}, {"items": [{"ingredient": "olive", "count": True}]},
                      {"items": [{"ingredient": "olive", "count": 0}]},
                      {"items": [{"ingredient": "olive", "count": 1}, {"ingredient": "Olive", "count": 2}]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_recipe(json.dumps(value))

    def test_injected_plan_is_dry_run_and_checks_camera_counts_and_inventory(self):
        frame = CameraFrame(b"test", "workspace", 123, 224, 224)
        point = PixelPoint(100, 100)
        pizza = SimpleNamespace(detect=lambda frame: Pizza(point, (point,), 0.99))
        ingredients = SimpleNamespace(detect=lambda frame: (Ingredient("olive", point, 0.99),))
        planner = SimpleNamespace(plan=lambda *args: (Placement("olive", point),))
        calibration = SimpleNamespace(camera_id="workspace")
        constraints = LayoutConstraints(10, 20)
        recipe = '{"items":[{"ingredient":"olive","count":1}]}'
        plan = build_plan(frame, recipe, pizza, ingredients, planner, calibration, constraints)
        self.assertFalse(plan.execution_enabled)
        calibration.camera_id = "wrist"
        with self.assertRaises(ValueError):
            build_plan(frame, recipe, pizza, ingredients, planner, calibration, constraints)
        calibration.camera_id = "workspace"
        planner.plan = lambda *args: ()
        with self.assertRaises(ValueError):
            build_plan(frame, recipe, pizza, ingredients, planner, calibration, constraints)
        ingredients.detect = lambda frame: ()
        with self.assertRaises(ValueError):
            build_plan(frame, recipe, pizza, ingredients, planner, calibration, constraints)

    def test_executor_never_moves_hardware(self):
        point = RobotPoint(0, 0, 0)
        with self.assertRaises(NotImplementedError):
            RobotExecutor().execute(MotionRequest("olive", point, point))


if __name__ == "__main__":
    unittest.main()
