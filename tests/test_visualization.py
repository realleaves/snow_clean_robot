"""tests/test_visualization.py -- lightweight OpenCV monitor, headless and CLI runs."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from camera.mock_camera import MockCamera
from decision.cleaning_task import CleaningTask
from perception.ground_roi import GroundROI
from perception.wet_region_detector import WetRegionDetector
from tests.helpers import DEPTH_CFG, PERCEPTION_CFG, ROI_CFG, ROOT, make_camera, make_world
from visualization.camera_view import annotate
from visualization.map_view import render_map
from visualization.system_monitor import SystemMonitor
from mapping.grid_map import GridMap


@pytest.fixture
def detection():
    """A dirty frame plus its detector mask and regions."""
    camera = make_camera(make_world())
    roi = GroundROI(ROI_CFG, DEPTH_CFG)
    detector = WetRegionDetector(PERCEPTION_CFG, 0.2)
    detector.build_reference([roi.extract(camera.get_frame())[0].copy()] * 3)
    camera.clean_reference_mode = False
    frame = camera.get_frame()
    image, depth, offset = roi.extract(frame)
    regions = detector.detect(image, depth, roi.valid_depth(depth), offset)
    yield frame, detector.last_mask, regions
    camera.stop()


def test_annotate_draws_bounding_boxes(detection):
    frame, _, regions = detection
    annotated = annotate(frame, regions)
    assert annotated.shape == frame.color_image.shape
    assert not np.array_equal(annotated, frame.color_image)
    x, y, w, h = regions[0].bbox
    assert annotated[y, x].tolist() != frame.color_image[y, x].tolist()


def test_annotate_without_regions_returns_a_copy(detection):
    frame, _, _ = detection
    annotated = annotate(frame, [])
    assert np.array_equal(annotated, frame.color_image)
    annotated[0, 0] = 0
    assert not np.array_equal(annotated, frame.color_image)


def test_monitor_renders_the_documented_panel(detection):
    frame, mask, regions = detection
    monitor = SystemMonitor(headless=True)
    task = CleaningTask(1, 0.0, 0.8, 0.62, 'MEDIUM')
    panel = monitor.render(frame, mask, regions, 'CLEANING', task, fps=10.0,
                           region=regions[0], path_length=1.25)
    assert panel.ndim == 3 and panel.shape[0] == frame.color_image.shape[0]
    assert panel.shape[1] == 2 * frame.color_image.shape[1]
    assert panel.dtype == np.uint8
    assert monitor.frames_rendered == 1
    assert panel.any()


def test_monitor_handles_missing_mask_and_task(detection):
    frame, _, regions = detection
    monitor = SystemMonitor(headless=True)
    panel = monitor.render(frame, None, regions, 'PATROL')
    assert panel.shape[1] == 2 * frame.color_image.shape[1]


def test_monitor_handles_empty_region_list(detection):
    frame, mask, _ = detection
    panel = SystemMonitor(headless=True).render(frame, mask, [], 'PATROL')
    assert panel.any()


def test_headless_monitor_never_touches_the_window_api(detection):
    frame, mask, regions = detection
    monitor = SystemMonitor(headless=True)
    monitor.show(frame, mask, regions, 'PATROL', None, 30.0)
    assert monitor.frames_rendered == 0      # show() is a no-op when headless
    monitor.close()
    assert monitor.headless is True


def test_non_headless_monitor_degrades_without_a_display(detection):
    """Without an X display cv2.imshow raises; the monitor must swallow it."""
    frame, mask, regions = detection
    monitor = SystemMonitor(headless=False)
    monitor.show(frame, mask, regions, 'PATROL', None, 30.0)
    if monitor.display_error is not None:
        assert monitor.headless is True      # degraded to headless, no crash
    monitor.close()


def test_render_map_shows_obstacles():
    grid = GridMap(np.zeros((4, 4), np.uint8), resolution_m=0.1)
    grid.set_obstacle((1, 1))
    image = render_map(grid, scale=10)
    assert image.shape == (40, 40, 3)
    assert image[15, 15].tolist() == [0, 0, 0]         # cell (1,1) -> black
    assert image[5, 5].tolist() == [255, 255, 255]     # cell (0,0) -> white


# ------------------------------------------------------------------- CLI / headless
def run_cli(args, extra_env=None):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    env['MPLBACKEND'] = 'Agg'
    env.pop('DISPLAY', None)          # guarantee a display-less environment
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, str(ROOT / 'main.py'), *args],
                          cwd=str(ROOT), env=env, capture_output=True, text=True,
                          timeout=180)


def test_cli_headless_run_completes():
    result = run_cli(['--mode', 'mock', '--headless', '--steps', '16', '--json'])
    assert result.returncode == 0, result.stderr
    assert '"mode": "mock"' in result.stdout
    assert 'Traceback' not in result.stderr


def test_cli_headless_run_creates_a_log(tmp_path):
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    for path in (ROOT / 'config').glob('*.yaml'):
        (config_dir / path.name).write_text(path.read_text(encoding='utf-8'),
                                            encoding='utf-8')
    result = run_cli(['--mode', 'mock', '--headless', '--steps', '4'])
    assert result.returncode == 0, result.stderr
    assert 'state=' in result.stdout


def test_cli_all_scenarios_report_passes():
    result = run_cli(['--all-scenarios'])
    assert result.returncode == 0, result.stderr
    assert '8/8' in result.stdout
    assert '[FAIL]' not in result.stdout


def test_cli_rejects_unknown_scenario():
    result = run_cli(['--scenario', 'Z'])
    assert result.returncode != 0
