"""V1.0 cleaning policy: a fixed number of passes per pollution level.

The policy deliberately contains **no** pressure, motor-speed or PWM control:
V1.0 only decides how many cleaning traverses are attempted.
"""
from utils.errors import ConfigError, TaskError

LEVELS = ('LIGHT', 'MEDIUM', 'HEAVY')


class CleaningPolicy:
    def __init__(self, cfg: dict):
        missing = [f'{level.lower()}_passes' for level in LEVELS
                   if f'{level.lower()}_passes' not in cfg]
        if missing:
            raise ConfigError(f'cleaning config is missing keys: {", ".join(missing)}')
        self.cfg = dict(cfg)
        for level in LEVELS:
            value = cfg[f'{level.lower()}_passes']
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigError(f'{level.lower()}_passes must be an integer >= 1')

    def passes(self, level: str, compensation: bool = False) -> int:
        """Number of traverses for *level*; compensation always runs a single pass."""
        if compensation:
            return 1
        key = f'{str(level).upper()}_passes'.lower()
        if key not in self.cfg:
            raise TaskError(f'unknown pollution level {level!r}')
        return int(self.cfg[key])

    def base_passes(self, level: str) -> int:
        return self.passes(level, compensation=False)

    def compensation_passes(self) -> int:
        return self.passes('LIGHT', compensation=True)

    @property
    def allows_compensation(self) -> bool:
        return True

    def describe(self) -> dict[str, int]:
        return {level: self.passes(level) for level in LEVELS}
