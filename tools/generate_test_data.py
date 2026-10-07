"""Deterministic generator for the virtual test-image data set.

Creates ``data/test_images/<category>/`` with synthetic 640x480 RGB frames plus a
matching depth ``.npy`` plane, so the whole perception test set is reproducible
from source control without shipping binary images::

    python -m tools.generate_test_data --per-class 30

Categories (V1.0 acceptance plan section 8):

* ``clean``         clean floor only
* ``light``         faint small stain
* ``medium``        mid-size stain
* ``heavy``         large dark stain
* ``interference``  specular highlight, shadow, light change, table leg, shoe,
                    floor texture, bright spot, colour shift

Depth planes are stored as compressed ``uint16`` millimetres (RealSense ``z16``
convention) and converted to metres by :class:`camera.mock_camera.MockCamera`.

The generator is intentionally rule-based (no deep learning) and every random
choice comes from a seeded RNG, so regenerating produces byte-identical frames.
"""
import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / 'data' / 'test_images'
DEFAULT_PER_CLASS = 8

BASE_COLOR = (125, 145, 155)
GROUND_DEPTH_M = 0.8
CATEGORIES = ('clean', 'light', 'medium', 'heavy', 'interference')
INTERFERENCE_KINDS = ('reflection', 'shadow', 'light_change', 'table_leg', 'shoe',
                      'texture', 'bright_spot', 'color_shift')


def _base_frame(rng: np.random.Generator) -> np.ndarray:
    return np.full((480, 640, 3), BASE_COLOR, np.uint8)


def _add_stain(image: np.ndarray, rng: np.random.Generator, radius: int,
               color: tuple[int, int, int], center=(320, 300)) -> None:
    jitter = rng.integers(-30, 31, 2)
    cv2.circle(image, (int(center[0] + jitter[0]), int(center[1] + jitter[1])),
               int(radius), color, -1)


def _render(kind: str, rng: np.random.Generator) -> np.ndarray:
    image = _base_frame(rng)
    if kind == 'clean':
        pass
    elif kind == 'light':
        _add_stain(image, rng, int(rng.integers(14, 20)), (95, 110, 120))
    elif kind == 'medium':
        _add_stain(image, rng, int(rng.integers(24, 32)), (60, 78, 88))
    elif kind == 'heavy':
        _add_stain(image, rng, int(rng.integers(42, 58)), (30, 45, 55))
    elif kind == 'interference':
        subtype = INTERFERENCE_KINDS[int(rng.integers(0, len(INTERFERENCE_KINDS)))]
        if subtype == 'reflection':
            cv2.circle(image, (int(rng.integers(200, 440)), int(rng.integers(240, 360))),
                       int(rng.integers(30, 70)), (250, 250, 250), -1)
        elif subtype == 'shadow':
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 260), (640, 480), (70, 82, 90), -1)
            image = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
        elif subtype == 'light_change':
            image = np.clip(image.astype(np.int16) + int(rng.integers(-40, 41)), 0, 255)
            image = image.astype(np.uint8)
        elif subtype == 'table_leg':
            x = int(rng.integers(120, 520))
            cv2.rectangle(image, (x, 180), (x + int(rng.integers(10, 26)), 480), (60, 60, 62), -1)
        elif subtype == 'shoe':
            _add_stain(image, rng, int(rng.integers(26, 40)), (40, 40, 45))
            _add_stain(image, rng, int(rng.integers(26, 40)), (35, 45, 120),
                       center=(int(rng.integers(240, 400)), int(rng.integers(300, 380))))
        elif subtype == 'texture':
            noise = rng.normal(0, 8, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif subtype == 'bright_spot':
            cv2.circle(image, (int(rng.integers(240, 400)), int(rng.integers(240, 340))),
                       int(rng.integers(18, 34)), (255, 255, 255), -1)
        else:  # color_shift
            image = np.clip(image.astype(np.int16) + np.array([0, 12, -12]), 0, 255).astype(np.uint8)
        return image
    else:  # pragma: no cover - guarded by CATEGORIES
        raise ValueError(f'unknown category {kind}')
    return image


def depth_millimetres(depth_m: float = GROUND_DEPTH_M, shape=(480, 640)) -> np.ndarray:
    """Depth plane in compressed z16 millimetres (what the generator stores)."""
    return np.full(shape, int(round(depth_m * 1000)), np.uint16)


def generate(out_dir: Path = DEFAULT_OUTPUT, per_class: int = DEFAULT_PER_CLASS,
             seed: int = 20260924) -> dict:
    out_dir = Path(out_dir)
    manifest = {'seed': seed, 'per_class': per_class, 'categories': {}}
    for kind in CATEGORIES:
        directory = out_dir / kind
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed + zlib.crc32(kind.encode()) % 10_000)
        for index in range(per_class):
            image = _render(kind, rng)
            cv2.imwrite(str(directory / f'{kind}_{index:03d}.png'), image)
            np.save(directory / f'{kind}_{index:03d}.npy',
                    depth_millimetres(), allow_pickle=False)
        manifest['categories'][kind] = per_class
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description='生成虚拟测试图像数据集')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--per-class', type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument('--seed', type=int, default=20260924)
    args = parser.parse_args()
    manifest = generate(args.output, args.per_class, args.seed)
    total = sum(manifest['categories'].values())
    print(f'generated {total} frames in {args.output} (per class: {args.per_class})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
