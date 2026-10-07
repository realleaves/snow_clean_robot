def require_aligned(frame) -> None:
    """Reject any depth image whose pixel grid differs from color."""
    if frame.aligned_depth.shape != frame.color_image.shape[:2]:
        raise ValueError("color and aligned depth have different dimensions")
