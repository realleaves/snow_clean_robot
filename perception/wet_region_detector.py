import cv2
import numpy as np
from perception.feature_extractor import feature_maps
from perception.pollution_region import PollutionRegion


class WetRegionDetector:
    def __init__(self, cfg: dict, max_area_ratio: float):
        self.cfg, self.max_area_ratio = cfg, max_area_ratio
        self.reference = None
        self.last_mask = None

    def build_reference(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames or any(x.shape != frames[0].shape for x in frames):
            raise ValueError("reference frames missing or inconsistent")
        self.reference = np.median(np.stack(frames), axis=0).astype(np.uint8)
        return self.reference

    def detect(self, roi: np.ndarray, depth: np.ndarray, valid_depth: np.ndarray,
               offset: tuple[int, int]) -> list[PollutionRegion]:
        if self.reference is None:
            raise RuntimeError("clean reference is not calibrated")
        maps = feature_maps(roi, self.reference, self.cfg)
        weights = {k: self.cfg[f'{k}_weight'] for k in maps}
        score = sum(weights[k] * maps[k] for k in maps)
        mask = (score >= self.cfg['candidate_threshold']).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
            np.ones((self.cfg['morph_open_kernel'],) * 2, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
            np.ones((self.cfg['morph_close_kernel'],) * 2, np.uint8))
        self.last_mask = mask
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        regions = []
        for idx in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area < self.cfg['min_region_area_px']:
                continue
            pixels = labels == idx
            depth_values = depth[pixels & valid_depth]
            # Depth is only a validity gate; it never contributes to wetness score.
            if depth_values.size == 0:
                continue
            depth_m = float(np.median(depth_values))
            visual = float(np.mean(score[pixels]))
            area_score = min(area / mask.size / self.max_area_ratio, 1.0)
            regions.append(PollutionRegion(idx, (x + offset[0], y + offset[1], w, h),
                (int(centers[idx][0] + offset[0]), int(centers[idx][1] + offset[1])),
                float(area), visual, area_score, depth_m))
        return regions
