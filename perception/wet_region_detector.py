"""Reference-difference wet-region detector with connected-component extraction."""
import cv2
import numpy as np

from perception.calibration import build_reference
from perception.feature_extractor import feature_maps
from perception.pollution_region import PollutionRegion
from utils.errors import CalibrationError, PerceptionError

REQUIRED_KEYS = ('min_region_area_px', 'gaussian_kernel', 'morph_open_kernel',
                 'morph_close_kernel', 'candidate_threshold', 'brightness_weight',
                 'saturation_weight', 'texture_weight', 'reflection_weight',
                 'brightness_scale', 'saturation_scale', 'texture_scale',
                 'reflection_threshold')


class WetRegionDetector:
    """Turns an RGB-D ground ROI into :class:`PollutionRegion` candidates.

    Depth is only a validity gate: it never contributes to the wetness score,
    because a depth camera cannot measure moisture.
    """

    def __init__(self, cfg: dict, max_area_ratio: float, frame_count: int | None = None,
                 min_valid_depth_ratio: float = 0.0):
        missing = [key for key in REQUIRED_KEYS if key not in cfg]
        if missing:
            raise PerceptionError(f'perception config is missing keys: {", ".join(missing)}')
        if not 0 < max_area_ratio <= 1:
            raise PerceptionError('max_area_ratio must be in (0, 1]')
        if not 0 <= min_valid_depth_ratio <= 1:
            raise PerceptionError('min_valid_depth_ratio must be in [0, 1]')
        self.cfg, self.max_area_ratio = dict(cfg), float(max_area_ratio)
        self.frame_count = frame_count
        self.min_valid_depth_ratio = float(min_valid_depth_ratio)
        self.reference = None
        self.last_mask = None
        self.last_score = None
        self.last_feature_maps = None

    # ------------------------------------------------------------------ reference
    def build_reference(self, frames: list[np.ndarray]) -> np.ndarray:
        self.reference = build_reference(list(frames), self.frame_count)
        return self.reference

    # ------------------------------------------------------------------ detection
    def _feature_score(self, roi: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise CalibrationError('clean reference is not calibrated')
        if roi.shape != self.reference.shape:
            raise PerceptionError(
                f'ROI shape {roi.shape} differs from reference shape {self.reference.shape}')
        maps = feature_maps(roi, self.reference, self.cfg)
        weights = {k: self.cfg[f'{k}_weight'] for k in maps}
        self.last_feature_maps = maps
        return sum(weights[k] * maps[k] for k in maps)

    def _mask(self, score: np.ndarray) -> np.ndarray:
        mask = (score >= self.cfg['candidate_threshold']).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((self.cfg['morph_open_kernel'],) * 2, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((self.cfg['morph_close_kernel'],) * 2, np.uint8))
        return mask

    def detect(self, roi: np.ndarray, depth: np.ndarray, valid_depth: np.ndarray,
               offset: tuple[int, int] = (0, 0)) -> list[PollutionRegion]:
        roi = np.asarray(roi)
        depth = np.asarray(depth)
        if depth.shape != roi.shape[:2]:
            raise PerceptionError(
                f'depth shape {depth.shape} does not match ROI shape {roi.shape[:2]}')
        score = self._feature_score(roi)
        self.last_score = score
        mask = self._mask(score)
        self.last_mask = mask
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        regions: list[PollutionRegion] = []
        for idx in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area < self.cfg['min_region_area_px']:
                continue
            pixels = labels == idx
            valid_pixels = pixels & valid_depth
            ratio = float(np.count_nonzero(valid_pixels) / max(area, 1))
            if ratio < self.min_valid_depth_ratio or np.count_nonzero(valid_pixels) == 0:
                depth_m: float | None = None
            else:
                depth_m = float(np.median(depth[valid_pixels]))
            visual = float(np.mean(score[pixels]))
            area_score = min(area / mask.size / self.max_area_ratio, 1.0)
            regions.append(PollutionRegion(
                idx, (x + offset[0], y + offset[1], w, h),
                (int(round(centers[idx][0] + offset[0])), int(round(centers[idx][1] + offset[1]))),
                float(area), visual, area_score, depth_m, valid_depth_ratio=ratio))
        return regions
