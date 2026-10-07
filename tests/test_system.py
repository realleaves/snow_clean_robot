"""tests/test_system.py -- configuration, composition root, error handling and teardown."""
import json
import logging
from pathlib import Path

import pytest

from camera.mock_camera import MockCamera
from camera.virtual_scene import VirtualWorld
from decision.state_machine import State
from system import DETECTION_STATION, SnowCleanSystem
from tests.helpers import contaminated_system, make_system, sandbox_root
from utils.config_loader import load_config, require_keys
from utils.errors import (CameraError, ConfigError, InvalidFrameError, NoPathError,
                          RobotError, SnowCleanError)


def test_config_files_load_and_contain_expected_sections():
    config = load_config(Path(__file__).resolve().parent.parent)
    assert set(config) == {'system', 'camera', 'perception', 'fusion', 'planner', 'cleaning'}
    require_keys(config['camera'], ('camera', 'ground_roi', 'depth', 'calibration'), 'camera')
    require_keys(config['fusion'], ('humidity', 'fusion', 'pollution_level'), 'fusion')


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_require_keys_lists_every_missing_key():
    with pytest.raises(ConfigError) as ctx:
        require_keys({'a': 1}, ('a', 'b', 'c'), 'demo')
    assert 'b' in str(ctx.value) and 'c' in str(ctx.value)


def test_real_mode_is_explicitly_refused():
    system = SnowCleanSystem(mode='real', headless=True)
    with pytest.raises(ConfigError):
        system.initialize()


def test_initialize_sets_up_every_module(dirty_system):
    system = dirty_system
    assert system.initialized is True and system.running is True
    assert system.machine.state is State.PATROL
    assert system.reference_path.exists()
    assert system.roi is not None and system.detector.reference is not None
    assert system.scorer is not None and system.classifier is not None
    assert system.grid.width == 40 and system.grid.height == 40
    assert system.tasks is not None and system.pollution_map is not None
    assert system.policy.describe() == {'LIGHT': 1, 'MEDIUM': 1, 'HEAVY': 2}
    assert system.evaluator.success_threshold == pytest.approx(0.80)


def test_double_initialize_is_rejected():
    system = make_system()
    try:
        system.initialize()
        with pytest.raises(RuntimeError):
            system.initialize()
    finally:
        system.shutdown()


def test_update_before_initialize_is_a_noop():
    system = SnowCleanSystem(headless=True)
    system.update()
    assert system.step_count == 0
    assert system.initialized is False


def test_shutdown_is_idempotent_and_releases_hardware():
    system = contaminated_system()
    system.initialize()
    system.shutdown()
    assert system.shutdown_complete is True
    assert system.running is False
    assert system.camera.released is True
    assert system.machine.state is State.SHUTDOWN
    system.shutdown()  # second call must not raise


def test_summary_contains_the_documented_fields(dirty_system):
    summary = dirty_system.summary()
    for key in ('mode', 'state', 'steps', 'running', 'errors', 'tasks', 'completed',
                'manual_check', 'failed', 'cleaning_passes', 'task_status'):
        assert key in summary
    assert json.dumps(summary, ensure_ascii=False)  # JSON serialisable for reports


def test_detection_station_is_the_fixed_viewpoint(dirty_system):
    assert dirty_system.pose.get_pose().x == 0.0
    assert dirty_system.pose.get_pose().y == 0.0
    assert DETECTION_STATION == (0.0, 0.0, 0.0)


def test_robot_error_is_contained_and_reported():
    system = contaminated_system()
    try:
        system.initialize()
        system.robot.emergency_stop('injected fault')
        for _ in range(8):
            system.update()
        assert system.machine.state is State.ERROR
        assert 'injected fault' in (system.last_error or '')
        assert system.error_count >= 1
        assert system.robot.cleaning is False
    finally:
        system.shutdown()


def test_no_path_handling_marks_error_and_stops():
    world = VirtualWorld(640, 480)
    world.add_target(0.0, 0.0, 0.05, depth_m=0.8)

    def block(system):
        system.grid.cells[:, :] = 1

    system = make_system(world, setup_hook=block)
    system.initialize()
    try:
        for _ in range(8):
            system.update()
        assert system.machine.state is State.ERROR
        assert isinstance(system.last_error, str) and system.last_error
    finally:
        system.shutdown()


def test_invalid_frame_is_contained():
    world = VirtualWorld(640, 480)
    world.add_target(0.0, 0.0, 0.05, depth_m=0.8)
    system = make_system(world)
    try:
        system.initialize()
        system.camera.failure_hook = lambda index: (
            (_ for _ in ()).throw(InvalidFrameError('empty frame')) if index >= 30 else None)
        for _ in range(4):
            system.update()
        assert system.machine.state is State.ERROR
        assert 'empty frame' in (system.last_error or '')
    finally:
        system.shutdown()


def test_camera_released_after_error_path():
    world = VirtualWorld(640, 480)
    world.add_target(0.0, 0.0, 0.05, depth_m=0.8)
    system = make_system(world)
    system.initialize()
    system.camera.failure_hook = lambda index: (
        (_ for _ in ()).throw(InvalidFrameError('empty frame')) if index >= 30 else None)
    for _ in range(4):
        system.update()
    system.shutdown()
    assert system.camera.released is True


def test_system_uses_injected_camera_and_world():
    world = VirtualWorld(640, 480)
    camera = MockCamera(640, 480, 30, world=world)
    system = SnowCleanSystem(camera=camera, world=world, headless=True)
    try:
        system.initialize()
        assert system.camera is camera
        assert system.world is world
    finally:
        system.shutdown()


def test_setup_hook_runs_after_calibration():
    seen = {}
    system = make_system(setup_hook=lambda s: seen.update(state=s.machine.state.value))
    try:
        system.initialize()
        assert seen['state'] == State.PATROL.value
    finally:
        system.shutdown()


def test_logs_go_to_the_configured_directory(tmp_path):
    root = sandbox_root(tmp_path)
    system = make_system(root=root)
    try:
        system.initialize()
        system.update()
    finally:
        system.shutdown()
    assert list((root / 'logs').glob('*.log'))


def test_reference_image_is_written_to_the_configured_path(tmp_path):
    root = sandbox_root(tmp_path)
    system = make_system(root=root)
    try:
        system.initialize()
        reference = root / system.config['camera']['calibration']['reference_path']
        assert reference.exists()
        assert system.reference_path == reference
    finally:
        system.shutdown()


def test_blacklist_prevents_duplicate_tasks_for_a_dead_area():
    world = VirtualWorld(640, 480)
    world.add_target(0.0, 0.0, 0.05, depth_m=0.8)
    for target in world.targets:
        target.freeze_after_pass = True
    world.detection_hint = (0.60, 0.10)
    system = make_system(world, humidity_mode='medium')
    system.initialize()
    try:
        for _ in range(30):
            system.update()
            if system.sleeping:
                break
        assert len(system.tasks.tasks) == 1
        assert system.blacklisted_targets
        assert system.skipped_detections >= 1
    finally:
        system.shutdown()
