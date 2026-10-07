"""Clean-reference calibration for the fixed-viewpoint ground ROI."""
from pathlib import Path

import cv2
import numpy as np

from utils.errors import CalibrationError

MIN_FRAMES = 3


def build_reference(frames: list[np.ndarray], frame_count: int | None = None) -> np.ndarray:
    """Build the clean ground reference as the per-pixel median of *frames*.

    The median suppresses occasional outliers (a stray bright pixel, one noisy
    frame) far better than a mean. Every frame must be a 3-channel image with an
    identical shape.
    """
    if frames is None or len(frames) == 0:
        raise CalibrationError('no calibration frames supplied')
    if frame_count is not None and len(frames) < frame_count:
        raise CalibrationError(
            f'calibration needs {frame_count} frames, received {len(frames)}')
    if len(frames) < MIN_FRAMES:
        raise CalibrationError(f'calibration needs at least {MIN_FRAMES} frames')
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise CalibrationError(f'calibration frames must be BGR images, got shape {first.shape}')
    for index, frame in enumerate(frames):
        if np.asarray(frame).shape != first.shape:
            raise CalibrationError(
                f'calibration frame {index} has shape {np.asarray(frame).shape}, expected {first.shape}')
    stack = np.stack([np.asarray(frame) for frame in frames]).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def save_reference(reference: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    if reference is None or np.asarray(reference).size == 0:
        raise CalibrationError('refusing to save an empty reference image')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(reference)):
        raise CalibrationError(f'failed to write reference image: {path}')
    return path


def load_reference(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise CalibrationError(f'reference image does not exist: {path}')
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise CalibrationError(f'cannot decode reference image: {path}')
    return image


class Calibrator:
    """Collects ROI frames and produces the clean reference image."""

    def __init__(self, frame_count: int = 30):
        if frame_count < MIN_FRAMES:
            raise CalibrationError(f'frame_count must be at least {MIN_FRAMES}')
        self.frame_count = int(frame_count)
        self.frames: list[np.ndarray] = []
        self.reference: np.ndarray | None = None

    def add_frame(self, roi_image: np.ndarray) -> None:
        self.frames.append(np.asarray(roi_image).copy())

    def build(self) -> np.ndarray:
        self.reference = build_reference(self.frames, self.frame_count)
        return self.reference

    def reset(self) -> None:
        self.frames.clear()
        self.reference = None
