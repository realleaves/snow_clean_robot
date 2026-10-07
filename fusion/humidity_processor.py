def normalize_humidity(raw: int, dry_raw: int, wet_raw: int) -> float:
    """Normalize either ADC polarity using experimentally calibrated endpoints."""
    if not 0 <= raw <= 4095 or not 0 <= dry_raw <= 4095 or not 0 <= wet_raw <= 4095:
        raise ValueError("humidity ADC values must be in 0..4095")
    if dry_raw == wet_raw:
        raise ValueError("dry_raw and wet_raw must differ")
    return max(0.0, min(1.0, (raw - dry_raw) / (wet_raw - dry_raw)))
