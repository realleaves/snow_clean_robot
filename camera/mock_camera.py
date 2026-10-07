"""Mock RGB-D camera for the fully virtual V1.0 acceptance environment.

The camera plays back one of several deterministic sources:

* a synthetic :class:`~camera.virtual_scene.VirtualWorld` scene (default);
* a single RGB image;
* a directory or explicit list of RGB images (a "video" sequence);
* a real video file decoded with OpenCV.

Depth is either ray-cast from the virtual world, derived from the RGB frame, or
loaded from ``.npy`` files placed next to an image sequence. Every frame is
shape-checked and time-stamped, and :meth:`MockCamera.stop` releases the source
so tests can assert on resource teardown.
"""
from pathlib import Path
import time

import cv2
import numpy as np

from camera.frame_align import require_aligned
from camera.frame_packet import FramePacket
from camera.virtual_scene import CAMERA_TO_ROBOT, DEFAULT_INTRINSICS, VirtualWorld
from utils.errors import CameraError, InvalidFrameError

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_SUFFIXES = {'.avi', '.mp4', '.mov', '.mkv', '.wmv'}


class MockCamera:
    """Deterministic RGB-D source used by every mock-mode test and demo."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 source: str | Path | None = None, image_paths: list[str | Path] | None = None,
                 depth_paths: list[str | Path] | None = None, world: VirtualWorld | None = None,
                 intrinsics=DEFAULT_INTRINSICS, depth_mode: str = 'world',
                 depth_value_m: float = 0.8, fps_window: int = 30):
        if width <= 0 or height <= 0:
            raise InvalidFrameError('camera width and height must be positive')
        if fps <= 0:
            raise CameraError('camera fps must be positive')
        if depth_mode not in ('world', 'rgb', 'constant'):
            raise CameraError(f'unknown depth mode {depth_mode!r}')
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.intrinsics = tuple(float(v) for v in intrinsics)
        self.depth_mode = depth_mode
        self.depth_value_m = float(depth_value_m)
        self.fps_window = int(fps_window)
        self.world = world if world is not None else VirtualWorld(self.width, self.height, self.intrinsics)
        self.camera_to_robot = CAMERA_TO_ROBOT.copy()

        self._source_path = Path(source) if source is not None else None
        self._images: list[np.ndarray] = []
        self._depths: list[np.ndarray | None] = []
        self._generator = None
        self.failure_hook = None
        self._capture = None
        self._index = 0
        self._started = False
        self._closed = False
        self._last_time: float | None = None
        self._intervals: list[float] = []
        self.measured_fps = 0.0
        self.frame_index = 0

        if image_paths:
            self._load_sequence([Path(p) for p in image_paths],
                                [Path(p) for p in depth_paths] if depth_paths else None)
        elif self._source_path is not None:
            self._load_source(self._source_path)

    # ------------------------------------------------------------------ loading
    def _load_source(self, path: Path) -> None:
        if not path.exists():
            raise CameraError(f'frame source does not exist: {path}')
        if path.is_dir():
            images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if not images:
                raise CameraError(f'no images found in {path}')
            self._load_sequence(images, None)
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise CameraError(f'cannot open video: {path}')
            self._capture = capture
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            self._load_sequence([path], None)
        else:
            raise CameraError(f'unsupported frame source: {path}')

    def _load_sequence(self, images: list[Path], depths: list[Path] | None) -> None:
        if not images:
            raise CameraError('image sequence is empty')
        if depths is not None and len(depths) != len(images):
            raise CameraError('depth sequence length differs from image sequence')
        loaded, loaded_depths = [], []
        for index, path in enumerate(images):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise CameraError(f'cannot read image: {path}')
            if image.shape[:2] != (self.height, self.width):
                image = cv2.resize(image, (self.width, self.height))
            loaded.append(image)
            if depths is not None:
                depth = self._to_metres(np.asarray(np.load(depths[index])))
                if depth.shape != image.shape[:2]:
                    raise CameraError(f'depth {depths[index]} does not match {path.name}')
            else:
                depth = self._read_depth(path)
            if depth is not None and depth.shape != image.shape[:2]:
                raise CameraError(f'depth sidecar for {path.name} does not match the image')
            loaded_depths.append(depth)
        self._images, self._depths = loaded, loaded_depths

    @staticmethod
    def _to_metres(depth: np.ndarray) -> np.ndarray:
        """Accept float metres or the RealSense z16 millimetre convention."""
        if np.issubdtype(depth.dtype, np.integer):
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    @classmethod
    def _read_depth(cls, image_path: Path) -> np.ndarray | None:
        candidate = image_path.with_suffix('.npy')
        if candidate.exists():
            return cls._to_metres(np.load(candidate))
        return None

    def set_generator(self, generator) -> None:
        """Use a callable returning :class:`FramePacket` instead of a file source."""
        self._generator = generator

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._closed:
            raise CameraError('camera was already released')
        self._started = True
        self._index = 0
        self.frame_index = 0

    def _check_started(self) -> None:
        if not self._started:
            raise CameraError('camera is not started')

    def _next_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        if self.failure_hook is not None:
            # Fault-injection point for the virtual exception scenarios.
            self.failure_hook(self.frame_index)
        if self._generator is not None:
            packet = self._generator()
            if not isinstance(packet, FramePacket):
                raise InvalidFrameError('generator must return a FramePacket')
            return packet.color_image, packet.aligned_depth
        if self._capture is not None:
            ok, image = self._capture.read()
            if not ok or image is None:
                raise CameraError('video stream ended or frame decode failed')
            depth = self._derive_depth(image)
            return image, depth
        if self._images:
            image = self._images[self._index % len(self._images)]
            stored = self._depths[self._index % len(self._depths)] if self._depths else None
            self._index += 1
            depth = stored if stored is not None else self._derive_depth(image)
            return image, depth
        image = self.world.render()
        depth = self.world.render_depth(self.depth_value_m) if self.depth_mode == 'world' \
            else self._derive_depth(image)
        return image, depth

    def _derive_depth(self, image: np.ndarray) -> np.ndarray:
        if self.depth_mode == 'constant':
            return np.full(image.shape[:2], self.depth_value_m, np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return np.clip(self.depth_value_m * (0.5 + gray / 255.0), 0.0, 10.0)

    def _validate(self, image: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image is None or depth is None:
            raise InvalidFrameError('empty RGB or depth frame')
        image = np.asarray(image)
        depth = np.asarray(depth)
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidFrameError(f'unexpected RGB shape {image.shape}')
        if image.shape[:2] != (self.height, self.width):
            raise InvalidFrameError(
                f'frame is {image.shape[1]}x{image.shape[0]}, expected {self.width}x{self.height}')
        if depth.shape != image.shape[:2]:
            raise InvalidFrameError(
                f'depth {depth.shape} does not match RGB {image.shape[:2]}')
        return image.copy(), depth.astype(np.float32, copy=True)

    # ------------------------------------------------------------------ frames
    def get_frame(self) -> FramePacket:
        self._check_started()
        try:
            image, depth = self._next_rgbd()
        except (CameraError, InvalidFrameError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise CameraError(f'frame acquisition failed: {exc}') from exc
        image, depth = self._validate(image, depth)
        now = time.monotonic()
        if self._last_time is not None:
            delta = now - self._last_time
            if delta > 0:
                self._intervals.append(delta)
                if len(self._intervals) > self.fps_window:
                    self._intervals.pop(0)
                self.measured_fps = 1.0 / (sum(self._intervals) / len(self._intervals))
        self._last_time = now
        self.frame_index += 1
        packet = FramePacket(now, image, depth, depth, self.intrinsics)
        require_aligned(packet)
        return packet

    def __iter__(self):
        return self

    def __next__(self) -> FramePacket:
        try:
            return self.get_frame()
        except CameraError as exc:
            raise StopIteration from exc

    # ------------------------------------------------------------------ mock API
    def clean(self, passes: int, target_id: int | None = None, retain: float = 0.45) -> None:
        self.world.clean(passes, target_id, retain)

    @property
    def pollution_fraction(self) -> float:
        return self.world.pollution_fraction

    @pollution_fraction.setter
    def pollution_fraction(self, value: float) -> None:
        self.world.pollution_fraction = value

    @property
    def clean_reference_mode(self) -> bool:
        return self.world.clean_reference_mode

    @clean_reference_mode.setter
    def clean_reference_mode(self, value: bool) -> None:
        self.world.clean_reference_mode = bool(value)

    @property
    def detection_hint(self):
        """Scenario-level ``(visual_score, area_score)`` injection, or ``None``."""
        return self.world.detection_hint

    @detection_hint.setter
    def detection_hint(self, value) -> None:
        if value is not None:
            visual, area = value
            if not 0 <= visual <= 1 or not 0 <= area <= 1:
                raise CameraError('detection_hint values must be within 0..1')
            value = (float(visual), float(area))
        self.world.set_detection_hint(value)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._started = False
        self._closed = True
        self._images = []
        self._depths = []
        self._generator = None

    @property
    def released(self) -> bool:
        return self._closed
