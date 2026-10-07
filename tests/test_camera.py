"""tests/test_camera.py -- MockCamera sources, frame contract and teardown."""
from pathlib import Path

import numpy as np
import pytest

from camera.frame_packet import FramePacket
from camera.mock_camera import MockCamera
from camera.virtual_scene import VirtualWorld
from tests.helpers import TEST_IMAGES, make_camera, make_world
from utils.errors import CameraError, InvalidFrameError


def test_frame_packet_contract(camera):
    packet = camera.get_frame()
    assert isinstance(packet, FramePacket)
    assert packet.color_image.shape == (480, 640, 3)
    assert packet.color_image.dtype == np.uint8
    assert packet.aligned_depth.shape == (480, 640)
    assert packet.aligned_depth.dtype == np.float32
    assert packet.intrinsics is not None and len(packet.intrinsics) == 4
    assert packet.timestamp > 0
    # RGB and aligned depth must describe the same pixel grid.
    assert packet.aligned_depth.shape == packet.color_image.shape[:2]


def test_continuous_frames_and_fps(camera):
    frames = [camera.get_frame() for _ in range(5)]
    assert camera.frame_index == 5
    assert all(f.timestamp <= g.timestamp for f, g in zip(frames, frames[1:]))
    assert camera.measured_fps > 0
    assert camera.fps == 30


def test_iteration_protocol(camera):
    iterator = iter(camera)
    assert isinstance(next(iterator), FramePacket)


def test_single_image_source(tmp_path):
    import cv2
    image = np.full((480, 640, 3), (10, 20, 30), np.uint8)
    path = tmp_path / 'single.png'
    cv2.imwrite(str(path), image)
    cam = MockCamera(640, 480, 30, source=path)
    cam.start()
    try:
        packet = cam.get_frame()
        assert np.array_equal(packet.color_image, image)
        # A still image repeats with a fresh timestamp.
        assert cam.get_frame().color_image is not None
    finally:
        cam.stop()


def test_image_directory_as_sequence():
    camera = make_camera(source=TEST_IMAGES / 'heavy')
    try:
        first = camera.get_frame()
        second = camera.get_frame()
        assert first.color_image.shape == second.color_image.shape
        assert not np.array_equal(first.color_image, second.color_image)
        assert first.aligned_depth[0, 0] == pytest.approx(0.8, abs=1e-3)
    finally:
        camera.stop()


def test_video_file_source(tmp_path):
    import cv2
    path = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
    assert writer.isOpened()
    for value in (20, 90, 160):
        writer.write(np.full((480, 640, 3), value, np.uint8))
    writer.release()
    cam = MockCamera(640, 480, 30, source=path)
    cam.start()
    try:
        values = [int(cam.get_frame().color_image.mean()) for _ in range(3)]
        assert values != []
    finally:
        cam.stop()


def test_generator_source():
    cam = MockCamera(640, 480, 30)
    cam.set_generator(lambda: FramePacket(1.0, np.zeros((480, 640, 3), np.uint8),
                                          np.ones((480, 640), np.float32) * 0.5,
                                          np.ones((480, 640), np.float32) * 0.5,
                                          (500, 500, 320, 240)))
    cam.start()
    try:
        packet = cam.get_frame()
        assert packet.aligned_depth[10, 10] == pytest.approx(0.5)
    finally:
        cam.stop()


def test_failure_hook_injects_frame_error():
    cam = MockCamera(640, 480, 30)

    def boom(_index):
        raise InvalidFrameError('injected')

    cam.failure_hook = boom
    cam.start()
    with pytest.raises(InvalidFrameError):
        cam.get_frame()
    cam.stop()


# ------------------------------------------------------------------ error paths
def test_missing_source_raises():
    with pytest.raises(CameraError):
        MockCamera(640, 480, 30, source='/nonexistent/frames')


def test_empty_directory_raises(tmp_path):
    with pytest.raises(CameraError):
        MockCamera(640, 480, 30, source=tmp_path)


def test_unsupported_suffix_raises(tmp_path):
    path = tmp_path / 'notes.txt'
    path.write_text('nope')
    with pytest.raises(CameraError):
        MockCamera(640, 480, 30, source=path)


def test_unreadable_image_raises(tmp_path):
    path = tmp_path / 'broken.png'
    path.write_bytes(b'not an image')
    with pytest.raises(CameraError):
        MockCamera(640, 480, 30, source=path)


def test_depth_shape_mismatch_raises(tmp_path):
    import cv2
    import numpy as np
    cv2.imwrite(str(tmp_path / 'frame.png'), np.zeros((480, 640, 3), np.uint8))
    np.save(tmp_path / 'frame.npy', np.zeros((100, 100), np.float32))
    with pytest.raises(CameraError):
        MockCamera(640, 480, 30, image_paths=[tmp_path / 'frame.png'])


def test_generator_must_return_frame_packet():
    cam = MockCamera(640, 480, 30)
    cam.set_generator(lambda: 'not a packet')
    cam.start()
    with pytest.raises(InvalidFrameError):
        cam.get_frame()
    cam.stop()


def test_get_frame_before_start_raises():
    cam = MockCamera(640, 480, 30)
    with pytest.raises(CameraError):
        cam.get_frame()


def test_invalid_configuration_raises():
    with pytest.raises(InvalidFrameError):
        MockCamera(0, 480)
    with pytest.raises(CameraError):
        MockCamera(640, 480, 0)
    with pytest.raises(CameraError):
        MockCamera(640, 480, 30, depth_mode='nonsense')


def test_stop_releases_resources(camera):
    camera.stop()
    assert camera.released is True
    with pytest.raises(CameraError):
        camera.start()
    assert camera._images == [] and camera._capture is None


def test_stop_without_start_is_safe():
    cam = MockCamera(640, 480, 30)
    cam.stop()
    assert cam.released is True


def test_depth_modes():
    world = make_world()
    constant = make_camera(world, depth_mode='constant', depth_value_m=1.25)
    rgb = make_camera(world, depth_mode='rgb', depth_value_m=0.9)
    try:
        assert constant.get_frame().aligned_depth[0, 0] == pytest.approx(1.25)
        assert rgb.get_frame().aligned_depth[0, 0] == pytest.approx(0.9, rel=0.3)
    finally:
        constant.stop()
        rgb.stop()
