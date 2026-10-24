"""Skeleton composition root. Detection and planning adapters are not bundled."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from coordinate_calibration import CoordinateCalibration
from ingredient_detection import IngredientDetector
from layout_planner import LayoutPlanner
from models import CameraFrame, LayoutConstraints, Plan
from pizza_detection import PizzaDetector
from recipe_parser import parse_recipe


def build_plan(
    frame: CameraFrame,
    recipe_json: str,
    pizza_detector: PizzaDetector,
    ingredient_detector: IngredientDetector,
    planner: LayoutPlanner,
    calibration: CoordinateCalibration,
    constraints: LayoutConstraints,
) -> Plan:
    """Dependency-injected dry-run wiring, never connected to RobotExecutor."""
    recipe = parse_recipe(recipe_json)
    if frame.camera_id != calibration.camera_id:
        raise ValueError("Camera frame does not match coordinate calibration.")
    pizza = pizza_detector.detect(frame)
    detections = ingredient_detector.detect(frame)
    for item in recipe:
        if sum(d.label == item.ingredient for d in detections) < item.count:
            raise ValueError(f"Insufficient detected instances of {item.ingredient}.")
    placements = planner.plan(pizza, recipe, calibration, constraints)
    if len(placements) != sum(item.count for item in recipe):
        raise ValueError("Planner returned the wrong number of placements.")
    for item in recipe:
        if sum(p.ingredient == item.ingredient for p in placements) != item.count:
            raise ValueError(f"Planner count mismatch for {item.ingredient}.")
    return Plan(recipe, placements, frame.camera_id, frame.captured_at_ns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True, help="Validate a structured recipe JSON file.")
    args = parser.parse_args()
    recipe = parse_recipe(args.recipe.read_text(encoding="utf-8"))
    print(json.dumps({
        "approach": "Multi-Modal-Approach",
        "recipe": [asdict(item) for item in recipe],
        "status": "recipe_validated_only",
        "missing_adapters": ["pizza_detector", "ingredient_detector", "layout_planner", "coordinate_calibration"],
        "execution_enabled": False,
    }, indent=2))


if __name__ == "__main__":
    main()
