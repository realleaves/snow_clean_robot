"""tests/test_depth.py -- metric depth acceptance across ROI, world and detector.

The V1.0 depth contract is deliberately narrow: a depth is usable only when it is
finite and inside the configured metric range. Nothing downstream may crash on a
missing, zero, NaN or infinite reading, and no NaN may reach a region's
``depth_m`` (it is ``None`` instead).
"""
import math

import cv2
import numpy as np
import pytest

from camera.mock_camera import MockCamera
from perception.ground_roi import GroundROI, depth_status
from tests.helpers import DEPTH_CFG, ROI_CFG
from utils.errors import SnowCleanError

#: ``(depth_m, expected status)`` for depth: {min 0.15, max 2.0}.
ACCEPTANCE_TABLE = (
    (0.05, 'INVALID'),   # too close
    (0.15, 'VALID'),     # inclusive lower bound
    (0.80, 'VALID'),
    (2.00, 'VALID'),     # inclusive upper bound
    (3.00, 'INVALID'),   # too far
)

#: Readings that are never usable.
UNUSABLE_TABLE = (
    (0.0, 'INVALID'),
    (float('nan'), 'INVALID'),
    (float('inf'), 'INVALID'),
    (float('-inf'), 'INVALID'),
    (-0.8, 'INVALID'),
)


# ------------------------------------------------------- configured V1.0 table
def test_acceptance_depth_table():
    for value, expected in ACCEPTANCE_TABLE:
        assert depth_status(value, DEPTH_CFG['min_distance_m'],
                            DEPTH_CFG['max_distance_m']) == expected, value


def test_unusable_readings_and_missing_depth():
    for value, expected in UNUSABLE_TABLE:
        assert depth_status(value, DEPTH_CFG['min_distance_m'],
                            DEPTH_CFG['max_distance_m']) == expected, value
    assert depth_status(None, DEPTH_CFG['min_distance_m'],
                        DEPTH_CFG['max_distance_m']) == 'MISSING'


def test_acceptance_table_through_the_roi_validity_gate(roi):
    values = np.array([value for value, _ in ACCEPTANCE_TABLE], np.float64).reshape(-1, 1)
    expected = np.array([status == 'VALID' for _, status in ACCEPTANCE_TABLE]).reshape(-1, 1)
    assert np.array_equal(roi.valid_depth(values), expected)
    assert roi.valid_depth(np.array([[0.0], [float('nan')], [float('inf')]])).sum() == 0


# ------------------------------------------------------------- virtual world
def test_virtual_world_depth_plane_defaults_to_0_8(world):
    depth = world.render_depth()
    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32
    assert np.allclose(depth, 0.8)


def test_virtual_world_invalid_mask_marks_masked_pixels_zero(world, roi):
    mask = np.zeros((480, 640), bool)
    mask[100:200, 200:400] = True
    depth = world.render_depth(invalid_mask=mask)
    assert np.all(depth[mask] == 0.0)
    assert np.allclose(depth[~mask], 0.8)
    assert int(roi.valid_depth(depth).sum()) == int((~mask).sum())


def test_virtual_world_whole_frame_invalid_plane(world, roi):
    depth = world.render_depth(invalid_mask=np.ones((480, 640), bool))
    assert np.all(depth == 0.0)
    assert roi.valid_depth_ratio(depth) == 0.0


# ------------------------------------------------- uint16 mm sidecar depths
def test_uint16_millimetre_sidecar_is_converted_to_metres(tmp_path):
    image = np.full((48, 64, 3), 90, np.uint8)
    image_path = tmp_path / 'frame.png'
    assert cv2.imwrite(str(image_path), image)
    np.save(tmp_path / 'frame.npy', np.full((48, 64), 800, np.uint16))

    camera = MockCamera(64, 48, 30, source=image_path)
    camera.start()
    try:
        packet = camera.get_frame()
        assert packet.aligned_depth.dtype == np.float32
        assert packet.aligned_depth[0, 0] == pytest.approx(0.8, abs=1e-3)
        assert float(packet.aligned_depth.min()) == pytest.approx(0.8, abs=1e-3)
        assert float(packet.aligned_depth.max()) == pytest.approx(0.8, abs=1e-3)
    finally:
        camera.stop()


# --------------------------------------------- whole-frame invalid depth plane
def test_all_invalid_depth_plane_keeps_the_region_without_depth(calibrated, camera, roi):
    """An all-False valid mask still yields the region, but with depth_m None."""
    camera.clean_reference_mode = False
    color, depth, offset = roi.extract(camera.get_frame())
    regions = calibrated.detect(color, depth, np.zeros(depth.shape, bool), offset)
    assert len(regions) == 1
    assert regions[0].depth_m is None
    assert regions[0].valid_depth_ratio == 0.0
    regions[0].validate()


def test_nan_depth_plane_never_leaks_nan_into_region_depth(calibrated, camera, roi):
    camera.clean_reference_mode = False
    color, depth, offset = roi.extract(camera.get_frame())
    nan_depth = np.full(depth.shape, np.nan, np.float32)
    regions = calibrated.detect(color, nan_depth, np.zeros(nan_depth.shape, bool), offset)
    assert len(regions) == 1
    assert regions[0].depth_m is None            # not NaN, not a crash
    assert not any(r.depth_m is not None and not math.isfinite(r.depth_m) for r in regions)
    regions[0].validate()


def test_valid_depth_plane_yields_a_finite_metric_depth(calibrated, camera, roi):
    camera.clean_reference_mode = False
    color, depth, offset = roi.extract(camera.get_frame())
    valid = roi.valid_depth(depth)
    assert valid.all()
    regions = calibrated.detect(color, depth, valid, offset)
    assert len(regions) == 1
    assert regions[0].depth_m == pytest.approx(0.8, abs=1e-3)
    assert regions[0].valid_depth_ratio == 1.0


# -------------------------------------------------------------- exception path
def test_depth_plane_shape_mismatch_raises(calibrated, camera, roi):
    camera.clean_reference_mode = False
    color, depth, offset = roi.extract(camera.get_frame())
    with pytest.raises(SnowCleanError):
        calibrated.detect(color, depth[:-1, :-1], np.zeros(depth.shape, bool), offset)


def test_depth_gate_rejects_out_of_range_configuration():
    with pytest.raises(SnowCleanError):
        GroundROI(ROI_CFG, dict(min_distance_m=0.0, max_distance_m=2.0))
    with pytest.raises(SnowCleanError):
        GroundROI(ROI_CFG, dict(min_distance_m=2.0, max_distance_m=0.5))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
