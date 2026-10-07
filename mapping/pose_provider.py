"""Robot pose sources.

The mock provider is deterministic and scriptable; the abstract interface is
shared with the (not yet available) real localization adapter, so the rest of
the system never depends on mock-only behaviour.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from utils.errors import ConfigError


def normalize_yaw(yaw: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
        raise ConfigError('yaw must be a number')
    if not math.isfinite(float(yaw)):
        raise ConfigError('yaw must be finite')
    return (float(yaw) + math.pi) % (2 * math.pi) - math.pi


@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float

    def copy(self) -> 'RobotPose':
        return RobotPose(self.x, self.y, self.yaw)


class PoseProvider(ABC):
    @abstractmethod
    def get_pose(self) -> RobotPose: ...

    def is_valid(self) -> bool:
        """A real provider returns False when localization is lost."""
        return True

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        raise NotImplementedError('this pose provider is read-only')


class MockPoseProvider(PoseProvider):
    """Scriptable pose used by every mock scenario and test."""

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self._pose = RobotPose(float(x), float(y), normalize_yaw(yaw))
        self.updates = 0

    def get_pose(self) -> RobotPose:
        return self._pose.copy()

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        if not all(math.isfinite(float(v)) for v in (x, y, yaw)):
            raise ConfigError('pose values must be finite')
        self._pose = RobotPose(float(x), float(y), normalize_yaw(yaw))
        self.updates += 1

    def move_by(self, forward: float, left: float = 0.0) -> RobotPose:
        """Translate the pose along its own heading (mock kinematics)."""
        pose = self._pose
        self.set_pose(pose.x + math.cos(pose.yaw) * forward - math.sin(pose.yaw) * left,
                      pose.y + math.sin(pose.yaw) * forward + math.cos(pose.yaw) * left,
                      pose.yaw)
        return self._pose

    def reset(self) -> None:
        self.set_pose(0.0, 0.0, 0.0)
