"""Humidity ADC normalization for both sensor polarities."""
from utils.errors import HumidityError

ADC_MIN, ADC_MAX = 0, 4095


def normalize_humidity(raw: int, dry_raw: int, wet_raw: int) -> float:
    """Normalize either ADC polarity using experimentally calibrated endpoints.

    ``H = (raw - dry_raw) / (wet_raw - dry_raw)`` clipped to ``[0, 1]``. A sensor
    whose ADC value *falls* when wet is supported by swapping the endpoints
    (``dry_raw > wet_raw``), and both polarities must produce the same ``[0, 1]``
    contract.
    """
    for name, value in (('raw', raw), ('dry_raw', dry_raw), ('wet_raw', wet_raw)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HumidityError(f'{name} must be a number')
        if not ADC_MIN <= value <= ADC_MAX:
            raise HumidityError(f'{name}={value} is outside the ADC range {ADC_MIN}..{ADC_MAX}')
    if dry_raw == wet_raw:
        raise HumidityError('dry_raw and wet_raw must differ')
    return max(0.0, min(1.0, (raw - dry_raw) / (wet_raw - dry_raw)))


class HumidityCalibration:
    """Holds a validated dry/wet endpoint pair for repeated readings."""

    def __init__(self, dry_raw: int, wet_raw: int):
        for name, value in (('dry_raw', dry_raw), ('wet_raw', wet_raw)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HumidityError(f'{name} must be a number')
        if dry_raw == wet_raw:
            raise HumidityError('dry_raw and wet_raw must differ')
        for name, value in (('dry_raw', dry_raw), ('wet_raw', wet_raw)):
            if not ADC_MIN <= value <= ADC_MAX:
                raise HumidityError(f'{name}={value} is outside the ADC range 0..4095')
        self.dry_raw, self.wet_raw = int(dry_raw), int(wet_raw)

    @property
    def reversed_polarity(self) -> bool:
        return self.wet_raw < self.dry_raw

    def normalize(self, raw: int) -> float:
        return normalize_humidity(raw, self.dry_raw, self.wet_raw)
