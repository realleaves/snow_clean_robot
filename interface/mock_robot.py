import logging
from interface.robot_interface import RobotInterface
from mapping.pose_provider import MockPoseProvider


class MockRobot(RobotInterface):
    def __init__(self, pose_provider: MockPoseProvider):
        self.pose_provider = pose_provider
        self.cleaning = False
        self.cleaner_lowered = False
        self.cleaning_passes = 0
        self.log = logging.getLogger('mock_robot')

    def _act(self, name: str) -> None:
        self.log.info('[MOCK ROBOT] %s', name)

    def move_forward(self, distance: float | None = None) -> None: self._act(f'move_forward {distance}m')
    def move_backward(self, distance: float | None = None) -> None: self._act(f'move_backward {distance}m')
    def turn_left(self, angle: float | None = None) -> None: self._act(f'turn_left {angle}rad')
    def turn_right(self, angle: float | None = None) -> None: self._act(f'turn_right {angle}rad')
    def stop(self) -> None: self._act('stop')
    def cleaner_down(self) -> None: self.cleaner_lowered = True; self._act('cleaner_down')
    def cleaner_up(self) -> None: self.cleaner_lowered = False; self._act('cleaner_up')

    def start_cleaning(self) -> None:
        if not self.cleaner_lowered:
            raise RuntimeError('cleaner must be lowered first')
        self.cleaning = True
        self._act('start_cleaning')

    def stop_cleaning(self) -> None:
        self.cleaning = False
        self._act('stop_cleaning')

    def execute_pass(self) -> None:
        if not self.cleaning:
            raise RuntimeError('cleaning motor is off')
        self.cleaning_passes += 1
        self._act(f'cleaning_pass {self.cleaning_passes}')

    def get_status(self) -> dict:
        return dict(cleaning=self.cleaning, cleaner_lowered=self.cleaner_lowered,
                    cleaning_passes=self.cleaning_passes)
