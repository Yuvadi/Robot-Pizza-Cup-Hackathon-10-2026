"""Shared contracts. Coordinates are explicit; no implicit camera/robot mapping."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PixelPoint:
    x: float
    y: float


@dataclass(frozen=True)
class RobotPoint:
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class CameraFrame:
    jpeg: bytes
    camera_id: str
    captured_at_ns: int
    width: int
    height: int


@dataclass(frozen=True)
class Pizza:
    center: PixelPoint
    boundary: tuple[PixelPoint, ...]
    confidence: float


@dataclass(frozen=True)
class Ingredient:
    label: str
    center: PixelPoint
    confidence: float


@dataclass(frozen=True)
class RecipeItem:
    ingredient: str
    count: int


@dataclass(frozen=True)
class Placement:
    ingredient: str
    target: PixelPoint


@dataclass(frozen=True)
class LayoutConstraints:
    boundary_margin_mm: float
    minimum_spacing_mm: float


@dataclass(frozen=True)
class MotionRequest:
    ingredient: str
    source: RobotPoint
    target: RobotPoint


@dataclass(frozen=True)
class Plan:
    recipe: tuple[RecipeItem, ...]
    placements: tuple[Placement, ...]
    camera_id: str
    captured_at_ns: int
    execution_enabled: Literal[False] = False
