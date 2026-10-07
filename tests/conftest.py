"""Shared pytest fixtures for the V1.0 virtual acceptance suite.

Everything here is deterministic and hardware-free: no D435i, no MCU, no
localization, no humidity probe. Tests import helpers from :mod:`tests.helpers`
so the suite can be read like the acceptance plan.
"""
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.helpers import (  # noqa: E402  (path setup must run first)
    DEPTH_CFG, PERCEPTION_CFG, ROI_CFG, TEST_IMAGES, contaminated_system,
    make_camera, make_system, make_world, reference_for,
)


def _reset_logging() -> None:
    """Flush and detach every handler so per-test log files stay isolated."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            root.removeHandler(handler)
    logging.disable(logging.NOTSET)


@pytest.fixture(autouse=True)
def quiet_logging():
    """Keep module logs from drowning the test report, and reset logging state."""
    _reset_logging()
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)
    _reset_logging()


@pytest.fixture
def world():
    return make_world()


@pytest.fixture
def camera(world):
    cam = make_camera(world)
    yield cam
    cam.stop()


@pytest.fixture
def roi():
    from perception.ground_roi import GroundROI
    return GroundROI(ROI_CFG, DEPTH_CFG)


@pytest.fixture
def detector():
    from perception.wet_region_detector import WetRegionDetector
    return WetRegionDetector(PERCEPTION_CFG, 0.2)


@pytest.fixture
def calibrated(detector):
    """Detector pre-loaded with a clean reference taken at the fixed viewpoint."""
    reference_for(detector)
    return detector


@pytest.fixture
def system():
    """Clean mock system (no pollution) with the closed loop initialized."""
    instance = make_system()
    try:
        instance.initialize()
        yield instance
    finally:
        instance.shutdown()


@pytest.fixture
def dirty_system():
    """System whose mock world contains one MEDIUM stain."""
    instance = contaminated_system()
    try:
        instance.initialize()
        yield instance
    finally:
        instance.shutdown()


def pytest_report_header(config):
    return (f'snow_clean_robot V1.0 virtual acceptance suite | '
            f'test images: {TEST_IMAGES} (present: {TEST_IMAGES.exists()})')
