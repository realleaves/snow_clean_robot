"""Command line entry point for the V1.0 mock closed loop."""
import argparse
import json
import sys

from scenario_runner import SCENARIOS, run_scenario
from system import SnowCleanSystem
from utils.errors import SnowCleanError


def main() -> int:
    parser = argparse.ArgumentParser(description='室内雪迹清除机器人 V1.0 上位机')
    parser.add_argument('--mode', choices=['mock', 'real'], default='mock')
    parser.add_argument('--headless', action='store_true', default=None,
                        help='disable the OpenCV window (default from config/system.yaml)')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--scenario', choices=sorted(SCENARIOS),
                        help='run one prepared virtual demo scenario instead of the default loop')
    parser.add_argument('--all-scenarios', action='store_true',
                        help='run every demo scenario A-H and print a JSON report')
    parser.add_argument('--json', action='store_true', help='print the run summary as JSON')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    if args.all_scenarios and args.scenario:
        parser.error('--scenario and --all-scenarios cannot be combined')
    if args.mode == 'real' and (args.all_scenarios or args.scenario):
        parser.error('prepared scenarios are mock-only; do not combine with --mode real')

    if args.all_scenarios or args.scenario:
        names = sorted(SCENARIOS) if args.all_scenarios else [args.scenario]
        reports = []
        failed = 0
        for name in names:
            result = run_scenario(name)
            reports.append(result)
            failed += 0 if result['passed'] else 1
            stream = sys.stderr if args.json else sys.stdout
            print(f"[{'PASS' if result['passed'] else 'FAIL'}] 场景 {name}: {result['title']}", file=stream)
            for check, ok in result['checks'].items():
                print(f"    {'✓' if ok else '✗'} {check}", file=stream)
        if args.json:
            print(json.dumps(reports, ensure_ascii=False, indent=2))
        print(f'\n{len(reports) - failed}/{len(reports)} 场景通过', file=stream)
        return 1 if failed else 0

    system = None
    exit_code = 0
    try:
        system = SnowCleanSystem(mode=args.mode, headless=args.headless)
        system.initialize()
        for _ in range(args.steps):
            if not system.running:
                break
            system.update()
            if system.task is not None and system.task.status in ('COMPLETED', 'MANUAL_CHECK'):
                break
    except KeyboardInterrupt:
        print('\ninterrupted', file=sys.stderr)
        exit_code = 130
    except (SnowCleanError, OSError, ValueError) as exc:
        print(f'启动或运行失败: {exc}', file=sys.stderr)
        exit_code = 1
        if system is not None:
            system.last_error = str(exc)
            system.error_count += 1
            system.machine.fail(str(exc))
    finally:
        if system is not None:
            summary = system.summary()
            system.shutdown()
    if system is None:
        return exit_code
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else
          f"state={summary['state']} tasks={summary['tasks']} completed={summary['completed']} "
          f"manual_check={summary['manual_check']} failed={summary['failed']} "
          f"passes={summary['cleaning_passes']} errors={summary['errors']}")
    return exit_code or (1 if summary['errors'] or summary['failed'] else 0)


if __name__ == '__main__':
    raise SystemExit(main())
