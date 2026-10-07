"""Deterministic builders and constants shared by the acceptance tests."""
from pathlib import Path

import numpy as np

from camera.frame_packet import FramePacket
from camera.mock_camera import MockCamera
from camera.virtual_scene import DEFAULT_INTRINSICS, VirtualWorld
from perception.ground_roi import GroundROI
from system import SnowCleanSystem
from utils.config_loader import load_config

ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES = ROOT / 'data' / 'test_images'

ROI_CFG = dict(x_start=40, x_end=600, y_start=180, y_end=480)
DEPTH_CFG = dict(min_distance_m=0.15, max_distance_m=2.0)
PERCEPTION_CFG = dict(load_config(ROOT)['perception']['perception'])
FUSION_CFG = dict(load_config(ROOT)['fusion']['fusion'])
CLEANING_CFG = dict(load_config(ROOT)['cleaning']['cleaning'])
CLEANING_FEEDBACK = dict(load_config(ROOT)['cleaning']['feedback'])
PRIORITY_CFG = dict(load_config(ROOT)['planner']['priority'])
HUMIDITY_CFG = dict(load_config(ROOT)['fusion']['humidity'])
LEVEL_CFG = dict(load_config(ROOT)['fusion']['pollution_level'])
MAP_CFG = dict(load_config(ROOT)['planner']['map'])


# ---------------------------------------------------------------------- camera
def make_world(radius_m: float = 0.05, fraction: float = 1.0, forward_m: float = 0.0,
               left_m: float = 0.0, depth_m: float = 0.8, **kwargs) -> VirtualWorld:
    world = VirtualWorld(640, 480)
    world.add_target(forward_m, left_m, radius_m, pollution_fraction=fraction,
                     depth_m=depth_m, **kwargs)
    return world


def make_camera(world: VirtualWorld | None = None, source=None, **kwargs) -> MockCamera:
    camera = MockCamera(640, 480, 30, world=world, source=source, **kwargs)
    camera.start()
    return camera


def frame_from_image(image: np.ndarray, depth_m: float = 0.8) -> FramePacket:
    depth = np.full(image.shape[:2], depth_m, np.float32)
    return FramePacket(0.0, image, depth, depth, DEFAULT_INTRINSICS)


def reference_for(detector, camera: MockCamera | None = None, frames: int = 3) -> None:
    """Build the detector's clean reference from the camera's clean viewpoint."""
    owned = camera is None
    camera = camera or make_camera(make_world())
    roi = GroundROI(ROI_CFG, DEPTH_CFG)
    image, _, _ = roi.extract(camera.get_frame())
    detector.build_reference([image.copy() for _ in range(frames)])
    if owned:
        camera.stop()


# ---------------------------------------------------------------------- system
def make_system(world: VirtualWorld | None = None, humidity_mode: str = 'wet',
                **kwargs) -> SnowCleanSystem:
    world = world if world is not None else VirtualWorld(640, 480)
    camera = MockCamera(640, 480, 30, world=world)
    return SnowCleanSystem(camera=camera, world=world, humidity_mode=humidity_mode,
                           headless=True, **kwargs)


def contaminated_system(fraction: float = 0.45, radius_m: float = 0.05,
                        humidity_mode: str = 'medium', visual: float = 0.46,
                        area: float = 0.05, retain: float = 0.45,
                        **kwargs) -> SnowCleanSystem:
    """System with one stain configured to need exactly one compensation pass."""
    world = make_world(radius_m=radius_m, fraction=fraction)
    world.cleaning_retain = retain
    world.detection_hint = (visual, area)
    return make_system(world, humidity_mode=humidity_mode, **kwargs)


def sandbox_root(tmp_path: Path) -> Path:
    """Copy ``config/`` next to *tmp_path* so a run can write logs in isolation."""
    import shutil
    target = Path(tmp_path)
    (target / 'config').mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'config').glob('*.yaml'):
        shutil.copy2(path, target / 'config' / path.name)
    return target


def run_until(system: SnowCleanSystem, predicate, max_steps: int = 60) -> dict:
    for _ in range(max_steps):
        if not system.running:
            break
        system.update()
        if predicate(system):
            break
    return system.summary()


class FrameSequenceSource:
    """Frame source for MockCamera that replays explicit RGB-D frames."""

    def __init__(self):
        self.frames: list[FramePacket] = []
        self.index = 0
        self.last_error: Exception | None = None

    def add(self, color_image: np.ndarray, depth: np.ndarray | None = None) -> None:
        depth = np.full(color_image.shape[:2], 0.8, np.float32) if depth is None else depth
        self.frames.append(FramePacket(0.0, color_image, depth, depth, DEFAULT_INTRINSICS))

    def frame(self) -> FramePacket:
        if self.last_error is not None:
            raise self.last_error
        packet = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return packet

    def fail_with(self, error: Exception) -> None:
        self.last_error = error
