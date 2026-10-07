from dataclasses import dataclass


@dataclass
class PollutionRegion:
    region_id: int
    bbox: tuple[int, int, int, int]
    center_px: tuple[int, int]
    area_px: float
    visual_score: float
    area_score: float
    depth_m: float | None = None
    relative_x: float | None = None
    relative_y: float | None = None
    pollution_score: float | None = None
    pollution_level: str | None = None
