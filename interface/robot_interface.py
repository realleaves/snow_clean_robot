"""Hardware abstraction shared by the mock and the future real robot adapter."""
from abc import ABC, abstractmethod


class RobotInterface(ABC):
    """Every motion and cleaning primitive the decision layer may request.

    Business code must never touch GPIO/PWM directly; it only calls this
    interface. Real adapters must confirm action completion before returning.
    """

    @abstractmethod
    def move_forward(self, distance: float | None = None) -> None: ...

    @abstractmethod
    def move_backward(self, distance: float | None = None) -> None: ...

    @abstractmethod
    def turn_left(self, angle: float | None = None) -> None: ...

    @abstractmethod
    def turn_right(self, angle: float | None = None) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def cleaner_down(self) -> None: ...

    @abstractmethod
    def cleaner_up(self) -> None: ...

    @abstractmethod
    def start_cleaning(self) -> None: ...

    @abstractmethod
    def stop_cleaning(self) -> None: ...

    @abstractmethod
    def execute_pass(self) -> None:
        """Complete one cleaning traverse and report success or raise on failure."""
        ...

    @abstractmethod
    def get_status(self) -> dict: ...

    @abstractmethod
    def action_log(self) -> list[str]:
        """Ordered record of the actions requested since the last reset."""
        ...

    @abstractmethod
    def clear_action_log(self) -> None: ...

    def set_speed(self, speed: float) -> None:
        raise NotImplementedError('speed control is not available in V1.0')
