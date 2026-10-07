"""Intel RealSense D435i adapter (real mode) and mock re-export.

The SDK is imported lazily so the fully virtual V1.0 environment never needs
``pyrealsense2``. ``MockCamera`` now lives in :mod:`camera.mock_camera` and is
re-exported here to keep existing imports working.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from camera.frame_packet import FramePacket
from camera.mock_camera import MockCamera  # noqa: F401  (re-exported for compatibility)
from utils.errors import CameraError

__all__ = ['RealSenseCamera', 'MockCamera']


class RealSenseCamera:
    """Return color-aligned metric depth; the SDK is loaded only in real mode."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width, self.height, self.fps = width, height, fps
        self.pipeline = None
        self._started = False
        self._last_time = None
        self.measured_fps = 0.0

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError('RealSense mode requires pyrealsense2 on this platform') from exc
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        try:
            profile = self.pipeline.start(config)
            self._started = True
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
        except Exception as exc:
            self.stop()
            raise CameraError(f'D435i start failed: {exc}') from exc

    def get_frame(self) -> FramePacket:
        if self.pipeline is None:
            raise CameraError('D435i is not started')
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            aligned = self.align.process(frames)
            color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color or not depth:
                raise CameraError('missing color or depth frame')
            intr = color.profile.as_video_stream_profile().intrinsics
            now = time.monotonic()
            if self._last_time is not None and now > self._last_time:
                self.measured_fps = 1 / (now - self._last_time)
            self._last_time = now
            rgb = np.asanyarray(color.get_data()).copy()
            depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
            return FramePacket(now, rgb, depth_m, depth_m,
                               (intr.fx, intr.fy, intr.ppx, intr.ppy))
        except CameraError:
            raise
        except Exception as exc:
            raise CameraError(f'D435i disconnected or frame acquisition failed: {exc}') from exc

    def save_test_frame(self, frame: FramePacket, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame.color_image)

    def stop(self) -> None:
        if self.pipeline is not None and self._started:
            self.pipeline.stop()
        self.pipeline = None
        self._started = False
