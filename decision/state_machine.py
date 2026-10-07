"""System-level state machine.

Normal path::

    INIT -> CALIBRATION -> PATROL -> DETECT -> APPROACH -> HUMIDITY_CONFIRM
         -> CLASSIFY -> CREATE_TASK -> PLANNING -> NAVIGATING -> CLEANING
         -> RECHECK -> PATROL            (RECHECK success == task PASS)

Compensation path::

    RECHECK -> COMPENSATE -> PLANNING -> NAVIGATING -> CLEANING -> RECHECK

Failure path::

    RECHECK -> COMPENSATE -> ... -> MANUAL_CHECK -> PATROL

Any serious error may move the machine to ``ERROR`` from any state except
``SHUTDOWN``; ``ERROR`` can only be left for ``SHUTDOWN``.
"""
from enum import Enum


class State(str, Enum):
    INIT = 'INIT'
    CALIBRATION = 'CALIBRATION'
    PATROL = 'PATROL'
    DETECT = 'DETECT'
    APPROACH = 'APPROACH'
    HUMIDITY_CONFIRM = 'HUMIDITY_CONFIRM'
    CLASSIFY = 'CLASSIFY'
    CREATE_TASK = 'CREATE_TASK'
    PLANNING = 'PLANNING'
    NAVIGATING = 'NAVIGATING'
    CLEANING = 'CLEANING'
    RECHECK = 'RECHECK'
    COMPENSATE = 'COMPENSATE'
    MANUAL_CHECK = 'MANUAL_CHECK'
    ERROR = 'ERROR'
    SHUTDOWN = 'SHUTDOWN'


ALLOWED = {
    State.INIT: {State.CALIBRATION, State.ERROR},
    State.CALIBRATION: {State.PATROL, State.ERROR},
    State.PATROL: {State.DETECT, State.SHUTDOWN, State.ERROR},
    State.DETECT: {State.PATROL, State.APPROACH, State.ERROR, State.SHUTDOWN},
    State.APPROACH: {State.HUMIDITY_CONFIRM, State.PATROL, State.ERROR},
    State.HUMIDITY_CONFIRM: {State.CLASSIFY, State.PATROL, State.ERROR},
    State.CLASSIFY: {State.CREATE_TASK, State.PATROL, State.ERROR},
    State.CREATE_TASK: {State.PLANNING, State.PATROL, State.ERROR},
    State.PLANNING: {State.NAVIGATING, State.PATROL, State.ERROR},
    State.NAVIGATING: {State.CLEANING, State.PATROL, State.ERROR},
    State.CLEANING: {State.RECHECK, State.PATROL, State.ERROR},
    State.RECHECK: {State.PATROL, State.COMPENSATE, State.MANUAL_CHECK, State.ERROR},
    State.COMPENSATE: {State.PLANNING, State.PATROL, State.MANUAL_CHECK, State.ERROR},
    State.MANUAL_CHECK: {State.PATROL, State.SHUTDOWN, State.ERROR},
    State.ERROR: {State.SHUTDOWN},
    State.SHUTDOWN: set(),
}

#: States that represent a normal end of the task cycle.
CYCLE_END_STATES = (State.PATROL,)


class StateMachine:
    def __init__(self):
        self.state = State.INIT
        self.history: list[State] = [State.INIT]
        self.error_reason: str | None = None

    def can_transition(self, target: State) -> bool:
        return State(target) in ALLOWED[self.state]

    def transition(self, target: State) -> State:
        target = State(target)
        if target == State.ERROR and self.state != State.SHUTDOWN:
            self.state = target
        elif target in ALLOWED[self.state]:
            self.state = target
        else:
            raise ValueError(f'invalid transition {self.state.value} -> {target.value}')
        self.history.append(self.state)
        return self.state

    def fail(self, reason: str) -> State:
        """Move to ``ERROR`` recording *reason*."""
        self.error_reason = reason
        if self.state != State.SHUTDOWN:
            self.transition(State.ERROR)
        return self.state

    def reset(self) -> None:
        self.state = State.INIT
        self.history = [State.INIT]
        self.error_reason = None

    def visited(self, state: State) -> bool:
        return State(state) in self.history
