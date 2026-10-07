import argparse
from system import SnowCleanSystem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mock', 'real'], default='mock')
    parser.add_argument('--headless', action='store_true', default=None)
    parser.add_argument('--steps', type=int, default=60)
    args = parser.parse_args()
    system = SnowCleanSystem(mode=args.mode, headless=args.headless)
    try:
        system.initialize()
        for _ in range(args.steps):
            if not system.running:
                break
            system.update()
            if system.task and system.task.status in ('COMPLETED', 'MANUAL_CHECK'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        system.shutdown()


if __name__ == '__main__':
    main()
