"""Mock robot: records every action and enforces the cleaning interlock."""
import logging

from interface.robot_interface import RobotInterface
from mapping.pose_provider import MockPoseProvider
from utils.errors import RobotError


class MockRobot(RobotInterface):
    """Deterministic actuator stub used by the whole virtual acceptance suite.

    Guarantees that V1.0 relies on:

    * every requested action is appended to :meth:`action_log`;
    * ``start_cleaning`` refuses to run while the cleaner is raised;
    * ``execute_pass`` refuses to run while the cleaning motor is off;
    * ``stop`` and :meth:`emergency_stop` always succeed, even after a fault.
    """

    def __init__(self, pose_provider: MockPoseProvider | None = None):
        self.pose_provider = pose_provider
        self.cleaning = False
        self.cleaner_lowered = False
        self.cleaning_passes = 0
        self.faulted = False
        self._actions: list[str] = []
        self._fault_message: str | None = None
        self.log = logging.getLogger('mock_robot')

    # ------------------------------------------------------------------ logging
    def _act(self, name: str) -> None:
        self._actions.append(name)
        self.log.info('[MOCK ROBOT] %s', name)

    def action_log(self) -> list[str]:
        return list(self._actions)

    def clear_action_log(self) -> None:
        self._actions.clear()

    def last_action(self) -> str | None:
        return self._actions[-1] if self._actions else None

    def actions_since(self, index: int) -> list[str]:
        return list(self._actions[index:])

    # ------------------------------------------------------------------ motion
    def move_forward(self, distance: float | None = None) -> None:
        self._require_operational()
        self._act(f'move_forward {distance}m')

    def move_backward(self, distance: float | None = None) -> None:
        self._require_operational()
        self._act(f'move_backward {distance}m')

    def turn_left(self, angle: float | None = None) -> None:
        self._require_operational()
        self._act(f'turn_left {angle}rad')

    def turn_right(self, angle: float | None = None) -> None:
        self._require_operational()
        self._act(f'turn_right {angle}rad')

    def stop(self) -> None:
        # An emergency stop must always be accepted, even while faulted.
        self._act('stop')

    def emergency_stop(self, reason: str = 'unspecified') -> None:
        self.cleaning = False
        self.cleaner_lowered = False
        self.faulted = True
        self._fault_message = reason
        self._actions.append(f'emergency_stop {reason}')
        self.log.error('[MOCK ROBOT] emergency_stop %s', reason)

    def clear_fault(self) -> None:
        self.faulted = False
        self._fault_message = None

    def _require_operational(self) -> None:
        if self.faulted:
            raise RobotError(f'robot faulted: {self._fault_message}')

    # ------------------------------------------------------------------ cleaning
    def cleaner_down(self) -> None:
        self._require_operational()
        self.cleaner_lowered = True
        self._act('cleaner_down')

    def cleaner_up(self) -> None:
        self.cleaner_lowered = False
        self._act('cleaner_up')

    def start_cleaning(self) -> None:
        self._require_operational()
        if not self.cleaner_lowered:
            raise RobotError('cleaner must be lowered before cleaning')
        if self.cleaning:
            raise RobotError('cleaning is already running')
        self.cleaning = True
        self._act('start_cleaning')

    def stop_cleaning(self) -> None:
        if self.cleaning:
            self._act('stop_cleaning')
        self.cleaning = False

    def execute_pass(self) -> None:
        self._require_operational()
        if not self.cleaning:
            raise RobotError('cannot clean while the cleaning motor is off')
        self.cleaning_passes += 1
        self._act(f'cleaning_pass {self.cleaning_passes}')

    # ------------------------------------------------------------------ status
    def get_status(self) -> dict:
        return dict(cleaning=self.cleaning, cleaner_lowered=self.cleaner_lowered,
                    cleaning_passes=self.cleaning_passes, faulted=self.faulted,
                    last_action=self.last_action())

    def reset(self) -> None:
        self.cleaning = False
        self.cleaner_lowered = False
        self.cleaning_passes = 0
        self.faulted = False
        self._fault_message = None
        self._actions.clear()
