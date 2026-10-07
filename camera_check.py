"""Read-only D435i capture diagnostic; does not initialize robot control.

Requires real hardware and ``pyrealsense2``. Without them it exits with a clear
message instead of a traceback, so it is safe to run as a smoke check.
"""
import argparse
import sys
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from camera.realsense_camera import RealSenseCamera
from utils.errors import CameraError


def main() -> int:
    parser = argparse.ArgumentParser(description='D435i 只读采集自检（不启动机器人）')
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--save-dir', type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load((Path(__file__).parent / 'config/camera.yaml').read_text())['camera']
    camera = RealSenseCamera(**cfg)
    count = 0
    start = time.monotonic()
    try:
        camera.start()
        while time.monotonic() - start < args.seconds:
            frame = camera.get_frame()
            count += 1
            if args.save_dir and count == 1:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save_dir / 'color.png'), frame.color_image)
                np.save(args.save_dir / 'aligned_depth_m.npy', frame.aligned_depth)
            if not args.headless:
                depth = np.clip(frame.aligned_depth / 2.0 * 255, 0, 255).astype(np.uint8)
                cv2.imshow('D435i RGB', frame.color_image)
                cv2.imshow('D435i aligned depth', depth)
                if cv2.waitKey(1) == 27:
                    break
    except CameraError as exc:
        print(f'camera check unavailable: {exc}', file=sys.stderr)
        return 1
    finally:
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()
    elapsed = max(time.monotonic() - start, 1e-9)
    print(f'frames={count} elapsed_s={elapsed:.1f} average_fps={count / elapsed:.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
