"""Plug in the separately trained pizza detector here."""

from typing import Protocol

from models import CameraFrame, Pizza


class PizzaDetector(Protocol):
    def detect(self, frame: CameraFrame) -> Pizza:
        """Return center, boundary, and confidence; raise if no valid pizza is found."""
        ...
