"""tests/test_humidity.py -- humidity normalization for both sensor polarities."""
import pytest

from fusion.humidity_processor import HumidityCalibration, normalize_humidity
from interface.humidity_interface import MockHumiditySensor
from tests.helpers import HUMIDITY_CFG
from utils.errors import ConfigError, HumidityError

DRY, WET = HUMIDITY_CFG['dry_raw'], HUMIDITY_CFG['wet_raw']

#: spec section 12 -- raw -> normalized with the configured endpoints.
FORWARD_TABLE = [
    (700, 0.0),
    (2100, 0.5),
    (3500, 1.0),
    (600, 0.0),      # below dry -> clipped
    (4095, 1.0),     # above wet -> clipped
]


@pytest.mark.parametrize('raw,expected', FORWARD_TABLE)
def test_forward_calibration(raw, expected):
    assert normalize_humidity(raw, DRY, WET) == pytest.approx(expected)


@pytest.mark.parametrize('raw,expected', [
    (3500, 0.0),
    (2100, 0.5),
    (700, 1.0),
    (3600, 0.0),     # beyond the dry end -> clipped
    (600, 1.0),      # beyond the wet end -> clipped
])
def test_reversed_calibration(raw, expected):
    assert normalize_humidity(raw, WET, DRY) == pytest.approx(expected)


@pytest.mark.parametrize('raw', range(0, 4096, 97))
def test_output_always_within_unit_interval(raw):
    for dry, wet in ((DRY, WET), (WET, DRY)):
        value = normalize_humidity(raw, dry, wet)
        assert 0.0 <= value <= 1.0


def test_dry_equal_wet_is_rejected():
    with pytest.raises(HumidityError):
        normalize_humidity(1000, 700, 700)
    with pytest.raises(HumidityError):
        HumidityCalibration(1500, 1500)


@pytest.mark.parametrize('raw,dry,wet', [
    (-1, 700, 3500), (4096, 700, 3500), (1000, -5, 3500), (1000, 700, 99999),
])
def test_out_of_adc_range_is_rejected(raw, dry, wet):
    with pytest.raises(HumidityError):
        normalize_humidity(raw, dry, wet)


def test_non_numeric_input_is_rejected():
    with pytest.raises(HumidityError):
        normalize_humidity('wet', 700, 3500)
    with pytest.raises(HumidityError):
        HumidityCalibration(None, 3500)


def test_calibration_object_helpers():
    forward = HumidityCalibration(DRY, WET)
    assert forward.reversed_polarity is False
    assert forward.normalize(2100) == pytest.approx(0.5)
    assert HumidityCalibration(WET, DRY).reversed_polarity is True


def test_mock_sensor_modes_and_injection():
    with pytest.raises(ConfigError):
        MockHumiditySensor('humid')
    sensor = MockHumiditySensor('dry')
    assert sensor.read_raw() == 700
    assert sensor.read_normalized(DRY, WET) == pytest.approx(0.0)
    sensor.set_mode('wet')
    assert sensor.read_normalized(DRY, WET) == pytest.approx(1.0)
    sensor.set_raw(2100)
    assert sensor.read_normalized(DRY, WET) == pytest.approx(0.5)
    with pytest.raises(ConfigError):
        sensor.set_raw(9999)
    sensor.fail_next = True
    with pytest.raises(HumidityError):
        sensor.read_raw()
    assert sensor.is_available() is True


def test_random_mode_stays_inside_adc_range():
    sensor = MockHumiditySensor('random', seed=7)
    values = [sensor.read_raw() for _ in range(50)]
    assert all(0 <= value <= 4095 for value in values)

    def retry_read():
        return MockHumiditySensor('random', seed=3)


def test_random_mode_can_be_replayed_with_seed():
    first = [MockHumiditySensor('random', seed=11).read_raw() for _ in range(5)]
    second = [MockHumiditySensor('random', seed=11).read_raw() for _ in range(5)]
    assert first == second
