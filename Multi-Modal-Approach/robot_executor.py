"""Execution boundary: this skeleton deliberately contains no hardware writes."""

from typing import Literal, Protocol

from models import MotionRequest


class ManipulationBackend(Protocol):
    strategy: Literal["deterministic", "hybrid-smolvla"]

    def preview(self, request: MotionRequest) -> dict[str, object]:
        """Validate robot mapping, calibration, reachability and collision constraints before motion."""
        ...


class RobotExecutor:
    def execute(self, request: MotionRequest) -> None:
        raise NotImplementedError(
            "Hardware execution is disabled in Multi-Modal-Approach. "
            "Station joint writes are available, but a validated Cartesian/IK or "
            "goal-conditioned manipulation adapter is not implemented."
        )
