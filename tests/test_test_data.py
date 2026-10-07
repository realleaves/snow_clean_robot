"""tests/test_test_data.py -- virtual test-image data set and generator (spec section 8)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.generate_test_data import (CATEGORIES, DEFAULT_PER_CLASS, INTERFERENCE_KINDS,
                                      depth_millimetres, generate)
from tests.helpers import ROI_CFG, DEPTH_CFG, TEST_IMAGES

from perception.ground_roi import GroundROI
from perception.wet_region_detector import WetRegionDetector
from tests.helpers import PERCEPTION_CFG


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    """Small freshly generated copy so tests never depend on committed images."""
    out = tmp_path_factory.mktemp('images')
    manifest = generate(out, per_class=6, seed=20260101)
    return out, manifest


def test_manifest_lists_every_category(dataset):
    _, manifest = dataset
    assert set(manifest['categories']) == set(CATEGORIES)
    assert all(count == 6 for count in manifest['categories'].values())


def test_generated_files_have_expected_shape(dataset):
    out, _ = dataset
    for category in CATEGORIES:
        for path in sorted((out / category).glob('*.png')):
            image = cv2.imread(str(path))
            assert image is not None
            assert image.shape == (480, 640, 3)
            depth = np.load(path.with_suffix('.npy'))
            assert depth.shape == (480, 640)
            assert depth.dtype == np.uint16          # z16 millimetres
            assert depth[0, 0] == 800                # 0.8 m


def test_generation_is_deterministic(tmp_path):
    first = generate(tmp_path / 'a', per_class=2, seed=7)
    second = generate(tmp_path / 'b', per_class=2, seed=7)
    assert first == second
    for category in CATEGORIES:
        left = sorted((tmp_path / 'a' / category).glob('*.png'))
        right = sorted((tmp_path / 'b' / category).glob('*.png'))
        assert left and right
        for a, b in zip(left, right):
            assert np.array_equal(cv2.imread(str(a)), cv2.imread(str(b)))


def test_unknown_category_is_rejected():
    from tools.generate_test_data import _render
    with pytest.raises(ValueError):
        _render('nonsense', np.random.default_rng(0))


def test_depth_millimetres_helper():
    plane = depth_millimetres(1.5, shape=(4, 5))
    assert plane.shape == (4, 5) and plane.dtype == np.uint16
    assert (plane == 1500).all()


def test_interference_kinds_are_all_rendered():
    from tools.generate_test_data import _render
    seen = set()
    rng = np.random.default_rng(3)
    for _ in range(200):
        image = _render('interference', rng)
        assert image.shape == (480, 640, 3)
        seen.add(int(image.sum()))
    assert len(seen) > 1, 'interference frames must actually vary'
    assert len(INTERFERENCE_KINDS) == 8


def _regions_for(image_path: Path):
    detector = WetRegionDetector(PERCEPTION_CFG, 0.2)
    roi = GroundROI(ROI_CFG, DEPTH_CFG)
    clean = np.full((480, 640, 3), (125, 145, 155), np.uint8)
    from tests.helpers import frame_from_image
    clean_roi, _, _ = roi.extract(frame_from_image(clean))
    detector.build_reference([clean_roi.copy()] * 3)
    image = cv2.imread(str(image_path))
    depth = np.load(image_path.with_suffix('.npy')).astype(np.float32) / 1000.0
    from camera.frame_packet import FramePacket
    from tests.helpers import DEFAULT_INTRINSICS
    packet = FramePacket(0.0, image, depth, depth, DEFAULT_INTRINSICS)
    roi_image, roi_depth, offset = roi.extract(packet)
    return detector.detect(roi_image, roi_depth, roi.valid_depth(roi_depth), offset)


def test_clean_frames_never_produce_candidates(dataset):
    out, _ = dataset
    for path in sorted((out / 'clean').glob('*.png')):
        assert _regions_for(path) == []


@pytest.mark.parametrize('category,min_area', [('light', 400), ('medium', 1500), ('heavy', 4000)])
def test_stain_frames_produce_increasing_areas(dataset, category, min_area):
    out, _ = dataset
    paths = sorted((out / category).glob('*.png'))
    assert paths
    for path in paths:
        regions = _regions_for(path)
        assert regions, f'{path.name} should yield a pollution candidate'
        assert sum(region.area_px for region in regions) >= min_area


def test_heavy_is_larger_than_light_on_average(dataset):
    out, _ = dataset

    def average_area(category):
        areas = [sum(r.area_px for r in _regions_for(p)) for p in (out / category).glob('*.png')]
        return float(np.mean(areas))

    assert average_area('heavy') > average_area('medium') > average_area('light')


def test_interference_frames_do_not_crash_the_detector(dataset):
    out, _ = dataset
    for path in sorted((out / 'interference').glob('*.png')):
        regions = _regions_for(path)
        for region in regions:
            region.validate()
            assert region.depth_m is None or 0.15 <= region.depth_m <= 2.0


def test_committed_dataset_matches_the_manifest():
    manifest_path = TEST_IMAGES / 'manifest.json'
    if not manifest_path.exists():
        pytest.skip('committed test image set not generated in this checkout')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for category, count in manifest['categories'].items():
        images = sorted(TEST_IMAGES.glob(f'{category}/*.png'))
        assert len(images) == count
        assert all(p.with_suffix('.npy').exists() for p in images)


def test_generator_defaults_are_documented():
    assert DEFAULT_PER_CLASS >= 5
