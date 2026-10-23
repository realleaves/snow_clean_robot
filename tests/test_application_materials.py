"""Regression checks for machine-readable demos and evidence exports."""
import hashlib
import json
import subprocess
import sys

import cv2
import pytest

import main
from scenario_runner import run_scenario
from tests.helpers import ROOT, contaminated_system
from tools.export_application_materials import collect_sources, export_materials


def run_cli(*arguments):
    return subprocess.run([sys.executable, str(ROOT / 'main.py'), *arguments],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)


def test_scenario_with_stop_callback_is_json_serializable():
    result = run_scenario('C')
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded['passed']
    assert 'stop_when' not in decoded['context']
    assert 'COMPENSATE' in decoded['state_history']
    assert decoded['trace'][-1] == decoded['summary']
    assert [action for action in decoded['robot_actions'] if action.startswith('cleaning_pass')] == [
        'cleaning_pass 1', 'cleaning_pass 2']


def test_cli_all_scenarios_json_is_a_complete_json_document():
    result = run_cli('--all-scenarios', '--json')
    assert result.returncode == 0, result.stderr
    reports = json.loads(result.stdout)
    assert [report['key'] for report in reports] == list('ABCDEFGH')
    assert all(report['passed'] for report in reports)
    assert '8/8' in result.stderr


@pytest.mark.parametrize('arguments', [('--steps', '0'), ('--steps', '-1'),
                                     ('--scenario', 'A', '--all-scenarios'),
                                     ('--mode', 'real', '--scenario', 'A')])
def test_invalid_cli_arguments_are_rejected(arguments):
    result = run_cli(*arguments)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr


def test_real_mode_refusal_is_readable_and_returns_failure():
    result = run_cli('--mode', 'real', '--headless', '--json')
    assert result.returncode == 1
    assert 'Traceback' not in result.stderr
    summary = json.loads(result.stdout)
    assert summary['state'] == 'ERROR'
    assert summary['errors'] == 1
    assert 'requires' in summary['last_error']


def test_runtime_error_does_not_return_success(monkeypatch, capsys):
    monkeypatch.setattr('system.configure_logger', lambda *arguments: None)
    system = contaminated_system()
    original_initialize = system.initialize

    def initialize_with_fault():
        original_initialize()
        system.robot.emergency_stop('test runtime failure')

    monkeypatch.setattr(system, 'initialize', initialize_with_fault)
    monkeypatch.setattr(main, 'SnowCleanSystem', lambda **kwargs: system)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--headless', '--json'])
    assert main.main() == 1
    assert json.loads(capsys.readouterr().out)['state'] == 'ERROR'
    assert system.shutdown_complete


def test_source_collection_excludes_dependencies_tests_and_empty_files():
    paths = collect_sources()
    assert paths[:3] == [ROOT / 'main.py', ROOT / 'system.py', ROOT / 'scenario_runner.py']
    assert len(paths) == len(set(paths))
    assert all('tests' not in path.relative_to(ROOT).parts for path in paths)
    assert all(path.read_text(encoding='utf-8').strip() for path in paths)


def test_exported_materials_are_reproducible_and_linked(tmp_path):
    destination = tmp_path / 'materials'
    report = export_materials(destination)
    assert report['passed']
    assert json.loads((destination / 'report.json').read_text()) == report
    manifest = json.loads((destination / 'manifest.json').read_text())
    for relative, digest in manifest['artifacts'].items():
        assert hashlib.sha256((destination / relative).read_bytes()).hexdigest() == digest
    sources = manifest['sources']
    full_source = ''.join(path.read_text(encoding='utf-8') for path in collect_sources())
    assert (destination / 'source_full.txt').read_text(encoding='utf-8') == full_source
    assert sources['selected_pages'] == min(sources['full_pages'], 60)
    printed = (destination / 'source_review.html').read_text(encoding='utf-8')
    assert printed.count('<section>') == sources['selected_pages']
    gallery = (destination / 'index.html').read_text(encoding='utf-8')
    assert '不是真机照片' in gallery
    for scenario in report['scenarios']:
        for filename in scenario['screenshots']:
            panel = cv2.imread(str(destination / 'screenshots' / filename))
            assert panel is not None and panel.shape == (480, 1280, 3)
            assert filename in gallery
    assert report['scenarios'][2]['screenshots']
    assert (destination / 'USER_MANUAL.md').exists()
    with pytest.raises(FileExistsError):
        export_materials(destination)
