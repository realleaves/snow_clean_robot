import cv2
import numpy as np


def render_map(grid_map, scale: int = 10):
    image = (1 - grid_map.cells) * 255
    return cv2.resize(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR),
                      None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
