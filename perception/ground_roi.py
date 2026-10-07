import numpy as np
from camera.frame_align import require_aligned


class GroundROI:
    def __init__(self, roi: dict, depth: dict):
        self.roi, self.depth = roi, depth

    def extract(self, frame) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        require_aligned(frame)
        h, w = frame.color_image.shape[:2]
        x0, x1 = self.roi['x_start'], self.roi['x_end']
        y0, y1 = self.roi['y_start'], self.roi['y_end']
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("ground ROI is outside image")
        color = frame.color_image[y0:y1, x0:x1]
        depth = frame.aligned_depth[y0:y1, x0:x1]
        return color, depth, (x0, y0)

    def valid_depth(self, depth: np.ndarray) -> np.ndarray:
        return np.isfinite(depth) & (depth >= self.depth['min_distance_m']) & (depth <= self.depth['max_distance_m'])
