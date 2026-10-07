import cv2
import numpy as np


def preprocess(image: np.ndarray, kernel: int) -> dict[str, np.ndarray]:
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("Gaussian kernel must be positive and odd")
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    return {
        'gray': cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0]),
        'hsv': cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV),
        'texture': cv2.Laplacian(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), cv2.CV_32F),
    }
