"""Contracts for separately trained detection and post-placement verification."""

from typing import Protocol

from models import CameraFrame, Ingredient, Placement


class IngredientDetector(Protocol):
    def detect(self, frame: CameraFrame) -> tuple[Ingredient, ...]:
        """Return labeled detections with confidence; never invent missing ingredients."""
        ...

    def verify_placement(self, before: CameraFrame, after: CameraFrame, expected: Placement) -> bool:
        """Check placement against a new observation; errors must not be treated as success."""
        ...
