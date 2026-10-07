"""Ground pollution candidate produced by the perception layer."""
from dataclasses import dataclass, field


@dataclass
class PollutionRegion:
    region_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h in full-image pixels
    center_px: tuple[int, int]
    area_px: float
    visual_score: float
    area_score: float
    depth_m: float | None = None
    relative_x: float | None = None
    relative_y: float | None = None
    pollution_score: float | None = None
    pollution_level: str | None = None
    valid_depth_ratio: float | None = None
    world_x: float | None = None
    world_y: float | None = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    def center_inside_bbox(self) -> bool:
        x, y, w, h = self.bbox
        return x <= self.center_px[0] <= x + w and y <= self.center_px[1] <= y + h

    def validate(self) -> None:
        """Raise :class:`ValueError` when the region violates V1.0 invariants."""
        x, y, w, h = self.bbox
        if w <= 0 or h <= 0:
            raise ValueError(f'bbox {self.bbox} has non-positive size')
        if self.area_px <= 0:
            raise ValueError('area_px must be positive')
        if not self.center_inside_bbox():
            raise ValueError(f'center_px {self.center_px} outside bbox {self.bbox}')
        for name, value in (('visual_score', self.visual_score), ('area_score', self.area_score)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f'{name}={value} outside [0, 1]')
        if self.depth_m is not None and self.depth_m <= 0:
            raise ValueError('depth_m must be positive or None')
        if self.pollution_score is not None and not 0.0 <= float(self.pollution_score) <= 1.0:
            raise ValueError('pollution_score outside [0, 1]')
