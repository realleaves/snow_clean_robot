"""Shared exception hierarchy used by the V1.0 virtual acceptance suite.

All recoverable failures raised by modules must derive from
:class:`SnowCleanError` so the system layer can map them onto the
``ERROR`` state and on the ``MANUAL_CHECK`` / ``FAILED`` task statuses.
"""


class SnowCleanError(Exception):
    """Base class for every domain error raised by this project."""


class ConfigError(SnowCleanError):
    """Invalid or missing configuration value (``InvalidConfig``)."""


class CameraError(SnowCleanError):
    """Camera cannot be started, read or released."""


class InvalidFrameError(CameraError):
    """Frame payload is empty, oversized, or RGB/Depth disagree."""


class CalibrationError(SnowCleanError):
    """Clean reference cannot be built from the supplied frames."""


class PerceptionError(SnowCleanError):
    """Ground ROI, preprocessing or candidate extraction failed."""


class HumidityError(SnowCleanError):
    """Humidity sensor transport or calibration failure."""


class DepthError(SnowCleanError):
    """Depth value or depth plane is unusable."""


class TaskError(SnowCleanError):
    """Task bookkeeping failure (unknown ID, illegal status change)."""


class PlanningError(SnowCleanError):
    """Planner failure, including ``NoPath``."""


class NoPathError(PlanningError):
    """No collision-free path exists between two cells."""


class RobotError(SnowCleanError):
    """Mock/real robot refused an action or reported a fault."""
