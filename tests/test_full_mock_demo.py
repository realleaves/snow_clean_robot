"""tests/test_full_mock_demo.py -- complete virtual mock closed loop (spec section 30)."""
import pytest

from scenario_runner import SCENARIOS, run_all, run_scenario
from tests.helpers import contaminated_system, make_system

EXPECTED_SCENARIOS = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')


def test_all_demo_scenarios_are_registered():
    assert tuple(sorted(SCENARIOS)) == EXPECTED_SCENARIOS
    for key, scenario in SCENARIOS.items():
        assert scenario.title and scenario.description


@pytest.mark.parametrize('key', EXPECTED_SCENARIOS)
def test_scenario_passes(key):
    result = run_scenario(key)
    failed = {name: ok for name, ok in result['checks'].items() if not ok}
    assert result['passed'], f'scenario {key} failed: {failed}'
    assert result['summary']['steps'] > 0


def test_run_all_reports_every_scenario():
    report = run_all()
    assert len(report) == len(EXPECTED_SCENARIOS)
    assert all(entry['passed'] for entry in report)
    for entry in report:
        assert entry['summary']['errors'] == 0 or entry['key'] in ('G', 'H')


def test_scenario_a_creates_no_task():
    summary = run_scenario('A')['summary']
    assert summary['tasks'] == 0
    assert summary['state'] == 'PATROL'


def test_scenario_b_is_a_single_pass_success():
    summary = run_scenario('B')['summary']
    assert summary['task_level'] == 'LIGHT'
    assert summary['task_retry'] == 0
    assert summary['cleaning_passes'] == 1


def test_scenario_c_uses_one_compensation():
    summary = run_scenario('C')['summary']
    assert summary['task_level'] == 'MEDIUM'
    assert summary['task_retry'] == 1
    assert summary['cleaning_passes'] == 2


def test_scenario_d_is_heavy_and_multi_pass():
    summary = run_scenario('D')['summary']
    assert summary['task_level'] == 'HEAVY'
    assert summary['cleaning_passes'] >= 2


def test_scenario_e_ends_in_manual_check_without_looping():
    summary = run_scenario('E')['summary']
    assert summary['manual_check'] == 1
    assert summary['task_retry'] == 2
    assert summary['steps'] < 40


def test_scenario_f_schedules_multiple_tasks():
    summary = run_scenario('F')['summary']
    assert summary['tasks'] >= 3
    assert summary['errors'] == 0


def test_scenario_g_no_path_stops_safely():
    summary = run_scenario('G')['summary']
    assert summary['state'] == 'ERROR'
    assert summary['last_error']
    assert summary['running'] is False


def test_scenario_h_invalid_frame_is_contained():
    summary = run_scenario('H')['summary']
    assert summary['state'] == 'ERROR'
    assert summary['last_error']
    assert summary['running'] is False


def test_default_loop_completes_one_task():
    system = contaminated_system()
    try:
        summary = system.run(max_steps=40)
    finally:
        system.shutdown()
    assert summary['completed'] == 1
    assert summary['errors'] == 0
    assert summary['state'] in ('PATROL', 'SHUTDOWN')


def test_clean_default_loop_never_cleans():
    system = make_system()
    try:
        summary = system.run(max_steps=8)
    finally:
        system.shutdown()
    assert summary['tasks'] == 0
    assert summary['cleaning_passes'] == 0
    assert summary['state'] == 'PATROL'
