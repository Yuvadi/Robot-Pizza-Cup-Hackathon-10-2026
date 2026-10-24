"""Geometry-only planning contract, independent of the manipulation policy."""

from typing import Protocol

from coordinate_calibration import CoordinateCalibration
from models import LayoutConstraints, Pizza, Placement, RecipeItem


class LayoutPlanner(Protocol):
    def plan(
        self,
        pizza: Pizza,
        recipe: tuple[RecipeItem, ...],
        calibration: CoordinateCalibration,
        constraints: LayoutConstraints,
    ) -> tuple[Placement, ...]:
        """Enforce counts, topping footprints, boundary margins and spacing, or fail as infeasible."""
        ...
