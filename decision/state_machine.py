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
    ERROR = 'ERROR'
    SHUTDOWN = 'SHUTDOWN'


ALLOWED = {
    State.INIT: {State.CALIBRATION},
    State.CALIBRATION: {State.PATROL},
    State.PATROL: {State.DETECT, State.SHUTDOWN},
    State.DETECT: {State.PATROL, State.APPROACH},
    State.APPROACH: {State.HUMIDITY_CONFIRM},
    State.HUMIDITY_CONFIRM: {State.CLASSIFY},
    State.CLASSIFY: {State.CREATE_TASK},
    State.CREATE_TASK: {State.PLANNING},
    State.PLANNING: {State.NAVIGATING, State.PATROL},
    State.NAVIGATING: {State.CLEANING},
    State.CLEANING: {State.RECHECK},
    State.RECHECK: {State.PATROL, State.COMPENSATE},
    State.COMPENSATE: {State.CLEANING, State.PATROL},
    State.ERROR: {State.SHUTDOWN},
    State.SHUTDOWN: set(),
}


class StateMachine:
    def __init__(self):
        self.state = State.INIT

    def transition(self, target: State) -> None:
        if target == State.ERROR and self.state != State.SHUTDOWN:
            self.state = target
        elif target in ALLOWED[self.state]:
            self.state = target
        else:
            raise ValueError(f'invalid transition {self.state.value} -> {target.value}')
