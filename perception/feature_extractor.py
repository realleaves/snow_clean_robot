import cv2
import numpy as np
from perception.image_preprocess import preprocess


def feature_maps(current: np.ndarray, reference: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    if current.shape != reference.shape:
        raise ValueError("reference and current ROI shapes differ")
    a = preprocess(current, cfg['gaussian_kernel'])
    b = preprocess(reference, cfg['gaussian_kernel'])
    norm = lambda x, scale: np.clip(x.astype(np.float32) / scale, 0, 1)
    brightness = norm(cv2.absdiff(a['gray'], b['gray']), cfg['brightness_scale'])
    saturation = norm(cv2.absdiff(a['hsv'][:, :, 1], b['hsv'][:, :, 1]), cfg['saturation_scale'])
    texture = np.clip(np.abs(a['texture'] - b['texture']) / cfg['texture_scale'], 0, 1)
    reflection = ((a['hsv'][:, :, 2] >= cfg['reflection_threshold']) &
                  (b['hsv'][:, :, 2] < cfg['reflection_threshold'])).astype(np.float32)
    return dict(brightness=brightness, saturation=saturation, texture=texture, reflection=reflection)
