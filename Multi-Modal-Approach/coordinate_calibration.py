"""Camera/workplane registration is distinct from servo joint calibration."""

from typing import Protocol

from models import PixelPoint, RobotPoint


class CoordinateCalibration(Protocol):
    camera_id: str
    robot_serial: str
    calibration_id: str

    def pixel_to_robot(self, point: PixelPoint, *, surface_z_mm: float) -> RobotPoint:
        """Map only inside the validated workplane and camera pose; reject extrapolation."""
        ...

    def distance_mm(self, first: PixelPoint, second: PixelPoint) -> float:
        """Measure workplane distance for metric spacing checks."""
        ...
