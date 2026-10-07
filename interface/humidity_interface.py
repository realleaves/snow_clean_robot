"""Humidity sensor abstraction and deterministic mock implementation."""
from abc import ABC, abstractmethod
import random

from fusion.humidity_processor import normalize_humidity
from utils.errors import ConfigError, HumidityError

ADC_MIN, ADC_MAX = 0, 4095


class HumidityInterface(ABC):
    @abstractmethod
    def read_raw(self) -> int: ...

    def read_normalized(self, dry_raw: int, wet_raw: int) -> float:
        return normalize_humidity(self.read_raw(), dry_raw, wet_raw)

    def is_available(self) -> bool:
        return True


class MockHumiditySensor(HumidityInterface):
    """Fixed-reading mock; ``mode`` selects a calibrated ADC level.

    Modes: ``dry`` (700), ``medium`` (2100), ``wet`` (3500), ``random`` inside the
    calibration window, or any explicit integer passed to :meth:`set_raw`.
    """

    VALUES = {'dry': 700, 'medium': 2100, 'wet': 3500}

    def __init__(self, mode: str = 'wet', seed: int | None = None):
        if mode not in (*self.VALUES, 'random'):
            raise ConfigError(f'unknown mock humidity mode {mode!r}')
        self.mode = mode
        self.fixed_raw: int | None = None
        self.fail_next = False
        self._rng = random.Random(seed)
        self.read_count = 0

    def set_mode(self, mode: str) -> None:
        if mode not in (*self.VALUES, 'random'):
            raise ConfigError(f'unknown mock humidity mode {mode!r}')
        self.mode, self.fixed_raw = mode, None

    def set_raw(self, raw: int) -> None:
        if not ADC_MIN <= raw <= ADC_MAX:
            raise ConfigError(f'raw value {raw} outside ADC range')
        self.fixed_raw = int(raw)

    def read_raw(self) -> int:
        if self.fail_next:
            self.fail_next = False
            raise HumidityError('mock humidity sensor read failure injected')
        self.read_count += 1
        if self.fixed_raw is not None:
            return self.fixed_raw
        if self.mode == 'random':
            return self._rng.randint(ADC_MIN, ADC_MAX)
        return self.VALUES[self.mode]
