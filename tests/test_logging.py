"""tests/test_logging.py -- log file creation, format, levels and non-overwrite."""
import logging
import re
from pathlib import Path

import pytest

from interface.mock_robot import MockRobot
from mapping.pose_provider import MockPoseProvider
from tests.helpers import contaminated_system, make_system, sandbox_root
from utils.logger import configure_logger


def flush_logging():
    """Flush file handlers so a file can be read inside the same test."""
    for handler in logging.getLogger().handlers:
        handler.flush()

TIMESTAMP = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \w+ [\w.]+ ')


def read_log(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_configure_logger_creates_file_with_expected_format(tmp_path):
    path = configure_logger(tmp_path, 'INFO')
    logging.getLogger('system').info('hello 日志')
    flush_logging()
    assert path.exists() and path.parent.name == 'logs'
    lines = read_log(path)
    assert any('hello 日志' in line for line in lines)
    assert TIMESTAMP.match(lines[-1]), lines[-1]
    assert ' INFO system ' in lines[-1]


def test_log_file_is_not_overwritten_between_runs(tmp_path):
    first = configure_logger(tmp_path, 'INFO')
    logging.getLogger('system').info('first run')
    flush_logging()
    second = configure_logger(tmp_path, 'INFO')
    logging.getLogger('system').info('second run')
    flush_logging()
    assert first != second
    assert 'first run' in first.read_text(encoding='utf-8')
    assert 'first run' not in second.read_text(encoding='utf-8')


@pytest.mark.parametrize('level,marker', [('INFO', 'info marker'),
                                          ('WARNING', 'warning marker'),
                                          ('ERROR', 'error marker')])
def test_levels_are_recorded(tmp_path, level, marker):
    path = configure_logger(tmp_path, 'INFO')
    logger = logging.getLogger('system')
    logger.info('info marker')
    logger.warning('warning marker')
    logger.error('error marker')
    flush_logging()
    text = path.read_text(encoding='utf-8')
    assert marker in text
    assert ' INFO ' in text and ' WARNING ' in text and ' ERROR ' in text


def test_exception_stack_is_recorded(tmp_path):
    path = configure_logger(tmp_path, 'INFO')
    try:
        raise RuntimeError('boom-stack')
    except RuntimeError:
        logging.getLogger('system').exception('caught')
    flush_logging()
    text = path.read_text(encoding='utf-8')
    assert 'RuntimeError: boom-stack' in text
    assert 'Traceback' in text


def test_debug_is_filtered_at_info_level(tmp_path):
    path = configure_logger(tmp_path, 'INFO')
    logging.getLogger('system').debug('should-not-appear')
    flush_logging()
    assert 'should-not-appear' not in path.read_text(encoding='utf-8')


def test_system_run_writes_state_task_and_score_lines(tmp_path):
    root = sandbox_root(tmp_path)
    system = contaminated_system(root=root)
    try:
        system.initialize()
        for _ in range(20):
            system.update()
            if system.task and system.task.status == 'COMPLETED':
                break
    finally:
        system.shutdown()
        flush_logging()
    logs = sorted((root / 'logs').glob('*.log'))
    assert logs, 'a log file must be created for the run'
    text = logs[-1].read_text(encoding='utf-8')
    assert 'state=' in text and 'task_id=' in text
    assert 'S=' in text and 'level=' in text
    assert 'cleaning_passes' in text
    assert '[MOCK ROBOT]' in text


def test_mock_robot_logs_every_action():
    robot = MockRobot(MockPoseProvider())
    robot.cleaner_down()
    robot.start_cleaning()
    robot.execute_pass()
    robot.stop_cleaning()
    robot.cleaner_up()
    assert robot.action_log() == ['cleaner_down', 'start_cleaning', 'cleaning_pass 1',
                                 'stop_cleaning', 'cleaner_up']


def test_headless_run_logs_normally(tmp_path):
    root = sandbox_root(tmp_path)
    system = make_system(root=root)
    try:
        system.initialize()
        for _ in range(4):
            system.update()
    finally:
        system.shutdown()
        flush_logging()
    logs = sorted((root / 'logs').glob('*.log'))
    assert logs and 'mode=mock' in logs[-1].read_text(encoding='utf-8')


def test_configured_log_level_is_respected(tmp_path):
    path = configure_logger(tmp_path, 'WARNING')
    logging.getLogger('system').info('below-threshold')
    logging.getLogger('system').warning('at-threshold')
    flush_logging()
    text = path.read_text(encoding='utf-8')
    assert 'below-threshold' not in text
    assert 'at-threshold' in text
