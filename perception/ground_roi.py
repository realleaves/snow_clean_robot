"""Fixed ground ROI extraction with explicit depth validity gating."""
import numpy as np

from camera.frame_align import require_aligned
from utils.errors import ConfigError, PerceptionError

ROI_KEYS = ('x_start', 'x_end', 'y_start', 'y_end')


def validate_roi(roi: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` after checking bounds, or raise."""
    missing = [key for key in ROI_KEYS if key not in roi]
    if missing:
        raise ConfigError(f'ground ROI is missing keys: {", ".join(missing)}')
    try:
        x0, x1 = int(roi['x_start']), int(roi['x_end'])
        y0, y1 = int(roi['y_start']), int(roi['y_end'])
    except (TypeError, ValueError) as exc:
        raise ConfigError('ground ROI values must be integers') from exc
    if not (0 <= x0 < x1 <= width):
        raise PerceptionError(
            f'ground ROI x range {x0}..{x1} is outside image width {width}')
    if not (0 <= y0 < y1 <= height):
        raise PerceptionError(
            f'ground ROI y range {y0}..{y1} is outside image height {height}')
    return x0, y0, x1, y1


class GroundROI:
    def __init__(self, roi: dict, depth: dict):
        self.roi, self.depth = dict(roi), dict(depth)
        for key in ('min_distance_m', 'max_distance_m'):
            if key not in self.depth:
                raise ConfigError(f'depth config is missing {key}')
        if not 0 < self.depth['min_distance_m'] <= self.depth['max_distance_m']:
            raise ConfigError('depth range must satisfy 0 < min_distance_m <= max_distance_m')

    def bounds(self, frame) -> tuple[int, int, int, int]:
        height, width = frame.color_image.shape[:2]
        return validate_roi(self.roi, width, height)

    def extract(self, frame) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        require_aligned(frame)
        x0, y0, x1, y1 = self.bounds(frame)
        color = frame.color_image[y0:y1, x0:x1]
        depth = frame.aligned_depth[y0:y1, x0:x1]
        if color.size == 0 or depth.size == 0:
            raise PerceptionError('ground ROI is empty')
        return color, depth, (x0, y0)

    def valid_depth(self, depth: np.ndarray) -> np.ndarray:
        """Mask of pixels with a finite depth inside the configured metric range."""
        values = np.asarray(depth)
        return (np.isfinite(values) & (values >= self.depth['min_distance_m'])
                & (values <= self.depth['max_distance_m']))

    def valid_depth_ratio(self, depth: np.ndarray) -> float:
        values = np.asarray(depth)
        if values.size == 0:
            return 0.0
        return float(np.count_nonzero(self.valid_depth(values)) / values.size)


def depth_status(depth_m: float | None, min_distance_m: float, max_distance_m: float) -> str:
    """Classify a single depth reading as ``VALID``/``INVALID``/``MISSING``."""
    if depth_m is None:
        return 'MISSING'
    value = float(depth_m)
    if not np.isfinite(value) or value <= 0:
        return 'INVALID'
    if value < min_distance_m or value > max_distance_m:
        return 'INVALID'
    return 'VALID'
