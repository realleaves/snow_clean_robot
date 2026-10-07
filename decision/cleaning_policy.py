class CleaningPolicy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        for key in ('light_passes', 'medium_passes', 'heavy_passes'):
            if cfg[key] < 1:
                raise ValueError(f'{key} must be at least one')

    def passes(self, level: str, compensation: bool = False) -> int:
        if compensation:
            return 1
        return self.cfg[f'{level.lower()}_passes']
