"""tests/test_ground_roi.py -- fixed ground ROI extraction and depth validity gating.

Covers the V1.0 contract of :mod:`perception.ground_roi`: crop geometry, ROI
bounds rejection (never an ``IndexError``/numpy crash), config validation, the
depth-validity table and the RGB/depth alignment requirement.
"""
import unittest

import numpy as np
import pytest

from camera.frame_packet import FramePacket
from camera.virtual_scene import DEFAULT_INTRINSICS
from perception.ground_roi import GroundROI, depth_status, validate_roi
from tests.helpers import DEPTH_CFG, ROI_CFG
from utils.errors import ConfigError, PerceptionError, SnowCleanError

#: ``(depth_m, expected_valid)`` for the configured 0.15 m .. 2.0 m gate.
VALID_DEPTH_TABLE = (
    (0.05, False),
    (0.15, True),   # inclusive lower bound
    (0.80, True),
    (2.00, True),   # inclusive upper bound
    (3.00, False),
    (float('nan'), False),
    (0.0, False),
    (-0.5, False),
)

#: ``(depth_m, expected_status)`` for the same readings plus a missing depth.
DEPTH_STATUS_TABLE = (
    (0.05, 'INVALID'),
    (0.15, 'VALID'),
    (0.80, 'VALID'),
    (2.00, 'VALID'),
    (3.00, 'INVALID'),
    (float('nan'), 'INVALID'),
    (float('inf'), 'INVALID'),
    (0.0, 'INVALID'),
    (-1.0, 'INVALID'),
    (None, 'MISSING'),
)


def make_frame(width: int = 640, height: int = 480, depth_m: float = 0.8,
               aligned_shape: tuple[int, int] | None = None) -> FramePacket:
    """Build a deterministic 640x480-style packet without touching a camera."""
    color = np.full((height, width, 3), 128, np.uint8)
    depth = np.full((height, width), depth_m, np.float32)
    aligned = depth if aligned_shape is None else np.full(aligned_shape, depth_m, np.float32)
    return FramePacket(0.0, color, depth, aligned, DEFAULT_INTRINSICS)


def depth_vector(values) -> np.ndarray:
    """A column vector so one ``valid_depth`` call checks a whole table."""
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


# --------------------------------------------------------------------- normal
def test_standard_frame_roi_crop(camera):
    """The configured ROI crops to 300x560x3 at offset (40, 180)."""
    roi = GroundROI(ROI_CFG, DEPTH_CFG)
    frame = camera.get_frame()
    color, depth, offset = roi.extract(frame)
    assert roi.bounds(frame) == (40, 180, 600, 480)
    assert color.shape == (300, 560, 3)
    assert color.dtype == np.uint8
    assert offset == (40, 180)
    # RGB ROI and depth ROI describe exactly the same pixel grid ...
    assert depth.shape == color.shape[:2]
    # ... and the depth ROI is the aligned-depth sub-array, bit for bit.
    assert np.array_equal(depth, frame.aligned_depth[180:480, 40:600])


def test_bounds_and_crop_are_consistent_for_a_full_frame_roi():
    roi = GroundROI(dict(x_start=0, x_end=640, y_start=0, y_end=480), DEPTH_CFG)
    color, depth, offset = roi.extract(make_frame())
    assert (color.shape, depth.shape, offset) == ((480, 640, 3), (480, 640), (0, 0))


# -------------------------------------------------------------------- boundary
def test_roi_bounds_are_inclusive_of_the_image_size():
    roi = GroundROI(dict(x_start=0, x_end=640, y_start=0, y_end=480), DEPTH_CFG)
    assert validate_roi(roi.roi, 640, 480) == (0, 0, 640, 480)


@pytest.mark.parametrize('roi_cfg', [
    dict(x_start=40, x_end=641, y_start=180, y_end=480),    # x_end past the width
    dict(x_start=40, x_end=600, y_start=180, y_end=481),    # y_end past the height
    dict(x_start=-10, x_end=600, y_start=180, y_end=480),   # negative x start
    dict(x_start=40, x_end=600, y_start=-1, y_end=480),     # negative y start
    dict(x_start=40, x_end=40, y_start=180, y_end=480),     # empty: x_start == x_end
    dict(x_start=40, x_end=600, y_start=300, y_end=300),    # empty: y_start == y_end
    dict(x_start=120, x_end=40, y_start=180, y_end=480),    # inverted x range
    dict(x_start=40, x_end=600, y_start=400, y_end=200),    # inverted y range
])
def test_out_of_range_roi_is_rejected_without_a_numpy_crash(roi_cfg):
    """Every malformed range raises a domain error, never IndexError."""
    roi = GroundROI(roi_cfg, DEPTH_CFG)
    frame = make_frame()
    # ``SnowCleanError`` is not an IndexError, so this also proves no numpy crash.
    with pytest.raises(SnowCleanError):
        roi.extract(frame)


def test_empty_roi_raises_domain_error():
    """x_start == x_end is an empty ROI, rejected before any slicing."""
    roi = GroundROI(dict(x_start=40, x_end=40, y_start=180, y_end=480), DEPTH_CFG)
    with pytest.raises(PerceptionError):
        roi.extract(make_frame())


# ------------------------------------------------------------------- exception
def test_missing_roi_keys_raise_config_error():
    for missing in ('x_start', 'x_end', 'y_start', 'y_end'):
        cfg = dict(ROI_CFG)
        del cfg[missing]
        with pytest.raises(ConfigError):
            GroundROI(cfg, DEPTH_CFG).extract(make_frame())


def test_non_integer_roi_values_raise_config_error():
    with pytest.raises(ConfigError):
        GroundROI(dict(ROI_CFG, x_start='left'), DEPTH_CFG).extract(make_frame())
    with pytest.raises(ConfigError):
        GroundROI(dict(ROI_CFG, y_end=None), DEPTH_CFG).extract(make_frame())


def test_invalid_depth_config_raises_config_error():
    with pytest.raises(ConfigError):   # min <= 0
        GroundROI(ROI_CFG, dict(min_distance_m=0.0, max_distance_m=2.0))
    with pytest.raises(ConfigError):   # min > max
        GroundROI(ROI_CFG, dict(min_distance_m=3.0, max_distance_m=2.0))
    with pytest.raises(ConfigError):   # negative min
        GroundROI(ROI_CFG, dict(min_distance_m=-0.15, max_distance_m=2.0))


def test_missing_depth_keys_raise_config_error():
    with pytest.raises(ConfigError):
        GroundROI(ROI_CFG, dict(max_distance_m=2.0))
    with pytest.raises(ConfigError):
        GroundROI(ROI_CFG, dict(min_distance_m=0.15))
    with pytest.raises(ConfigError):
        GroundROI(ROI_CFG, {})


def test_rgb_depth_dimension_mismatch_is_rejected():
    """Uses ``camera.frame_align.require_aligned``: mismatched shapes -> ValueError."""
    roi = GroundROI(ROI_CFG, DEPTH_CFG)
    frame = make_frame(aligned_shape=(240, 320))
    with pytest.raises(ValueError):
        roi.extract(frame)


# ------------------------------------------------------------- depth validity
def test_valid_depth_table(roi):
    """The depth gate is a small, explicit table (inclusive on both ends)."""
    values = depth_vector([depth for depth, _ in VALID_DEPTH_TABLE])
    mask = roi.valid_depth(values)
    assert mask.shape == values.shape
    assert [bool(v) for v in mask[:, 0]] == [expected for _, expected in VALID_DEPTH_TABLE]


def test_valid_depth_ratio_all_valid_and_all_invalid(roi):
    assert roi.valid_depth_ratio(np.full((20, 20), 0.8, np.float32)) == 1.0
    assert roi.valid_depth_ratio(np.full((20, 20), 5.0, np.float32)) == 0.0
    assert roi.valid_depth_ratio(np.full((20, 20), np.nan, np.float32)) == 0.0
    # Half-and-half is the boundary in between.
    half = np.concatenate([np.full(10, 0.8, np.float32), np.full(10, 0.0, np.float32)])
    assert roi.valid_depth_ratio(half) == pytest.approx(0.5)


def test_depth_status_table(roi):
    for value, expected in DEPTH_STATUS_TABLE:
        assert depth_status(value, roi.depth['min_distance_m'],
                            roi.depth['max_distance_m']) == expected


def test_depth_status_uses_the_configured_range():
    assert depth_status(0.10, 0.15, 2.0) == 'INVALID'
    assert depth_status(0.15, 0.15, 2.0) == 'VALID'
    assert depth_status(2.0, 0.15, 2.0) == 'VALID'
    assert depth_status(2.01, 0.15, 2.0) == 'INVALID'


if __name__ == '__main__':
    unittest.main()
