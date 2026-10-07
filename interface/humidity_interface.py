from abc import ABC, abstractmethod
import random


class HumidityInterface(ABC):
    @abstractmethod
    def read_raw(self) -> int: ...


class MockHumiditySensor(HumidityInterface):
    VALUES = {'dry': 700, 'medium': 2100, 'wet': 3500}

    def __init__(self, mode: str = 'wet'):
        if mode not in (*self.VALUES, 'random'):
            raise ValueError("unknown mock humidity mode")
        self.mode = mode

    def read_raw(self) -> int:
        return random.randint(700, 3500) if self.mode == 'random' else self.VALUES[self.mode]
