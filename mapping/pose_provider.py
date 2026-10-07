from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float


class PoseProvider(ABC):
    @abstractmethod
    def get_pose(self) -> RobotPose: ...


class MockPoseProvider(PoseProvider):
    def __init__(self, x: float = 0, y: float = 0, yaw: float = 0):
        self.pose = RobotPose(x, y, yaw)

    def get_pose(self) -> RobotPose:
        return self.pose

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        self.pose = RobotPose(x, y, yaw)
