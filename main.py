"""Command line entry point for the V1.0 mock closed loop."""
import argparse
import json
import sys
from pathlib import Path

from scenario_runner import SCENARIOS, run_scenario
from system import SnowCleanSystem


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

    if args.all_scenarios or args.scenario:
        names = sorted(SCENARIOS) if args.all_scenarios else [args.scenario]
        reports = []
        failed = 0
        for name in names:
            result = run_scenario(name)
            reports.append(result)
            failed += 0 if result['passed'] else 1
            print(f"[{'PASS' if result['passed'] else 'FAIL'}] 场景 {name}: {result['title']}")
            for check, ok in result['checks'].items():
                print(f"    {'✓' if ok else '✗'} {check}")
        if args.json:
            print(json.dumps(reports, ensure_ascii=False, indent=2))
        print(f'\n{len(reports) - failed}/{len(reports)} 场景通过')
        return 1 if failed else 0

    system = SnowCleanSystem(mode=args.mode, headless=args.headless)
    try:
        system.initialize()
        for _ in range(args.steps):
            if not system.running:
                break
            system.update()
            if system.task is not None and system.task.status in ('COMPLETED', 'MANUAL_CHECK'):
                break
    except KeyboardInterrupt:
        print('\ninterrupted', file=sys.stderr)
    finally:
        summary = system.summary()
        system.shutdown()
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else
          f"state={summary['state']} tasks={summary['tasks']} completed={summary['completed']} "
          f"manual_check={summary['manual_check']} failed={summary['failed']} "
          f"passes={summary['cleaning_passes']} errors={summary['errors']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
