from dataclasses import dataclass
import numpy as np


@dataclass
class FramePacket:
    timestamp: float
    color_image: np.ndarray  # BGR, OpenCV convention
    depth_image: np.ndarray  # metres, not RealSense raw units
    aligned_depth: np.ndarray  # metres in color pixel coordinates
    intrinsics: tuple[float, float, float, float] | None = None  # fx, fy, cx, cy
