"""tests/test_calibration.py -- clean-reference construction, I/O and Calibrator."""
import unittest

import numpy as np
import pytest

from perception.calibration import (
    MIN_FRAMES, Calibrator, build_reference, load_reference, save_reference,
)
from utils.errors import CalibrationError


def clean_frame(value: int = 120, shape: tuple[int, int, int] = (8, 8, 3)) -> np.ndarray:
    return np.full(shape, value, np.uint8)


class CalibrationTests(unittest.TestCase):
    # ------------------------------------------------------------------ normal
    def test_thirty_identical_frames_reproduce_the_frame(self):
        frame = clean_frame(120, (6, 7, 3))
        frames = [frame.copy() for _ in range(30)]
        reference = build_reference(frames, frame_count=30)
        self.assertEqual(reference.dtype, np.uint8)
        self.assertEqual(reference.shape, frame.shape)
        self.assertTrue(np.array_equal(reference, frame))

    def test_exactly_min_frames_is_accepted(self):
        frame = clean_frame(90, (4, 4, 3))
        reference = build_reference([frame.copy() for _ in range(MIN_FRAMES)])
        self.assertTrue(np.array_equal(reference, frame))

    def test_noise_frames_stay_close_to_the_clean_frame(self):
        frame = clean_frame(100, (5, 5, 3))
        frames = []
        for offset in (-2, -1, 0, 1, 2):
            noisy = frame.astype(np.int16) + offset
            frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
        reference = build_reference(frames)
        self.assertLessEqual(int(np.abs(reference.astype(int) - 100).max()), 2)

    def test_isolated_bright_outlier_pixel_is_suppressed(self):
        """A single stray pixel in one frame must not survive the median."""
        frame = clean_frame(60, (5, 5, 3))
        frames = [frame.copy() for _ in range(5)]
        frames[2][3, 4] = (255, 255, 255)
        reference = build_reference(frames)
        self.assertTrue(np.array_equal(reference, frame))
        self.assertEqual(tuple(reference[3, 4]), (60, 60, 60))

    def test_brightness_drift_is_averaged_out(self):
        frame = clean_frame(120, (6, 6, 3))
        frames = [np.clip(frame.astype(np.int16) + offset, 0, 255).astype(np.uint8)
                  for offset in (-5, 0, 5) * 10]
        reference = build_reference(frames)
        self.assertEqual(reference.dtype, np.uint8)
        self.assertLessEqual(int(np.abs(reference.astype(int) - 120).max()), 3)

    # --------------------------------------------------------------- exception
    def test_too_few_frames_for_the_requested_count(self):
        frame = clean_frame()
        with self.assertRaises(CalibrationError):
            build_reference([frame.copy() for _ in range(3)], frame_count=10)

    def test_fewer_than_min_frames(self):
        frame = clean_frame()
        for frames in ([frame.copy()], [frame.copy(), frame.copy()]):
            with self.assertRaises(CalibrationError):
                build_reference(frames)

    def test_empty_and_none_input(self):
        with self.assertRaises(CalibrationError):
            build_reference([])
        with self.assertRaises(CalibrationError):
            build_reference(None)

    def test_inconsistent_shapes(self):
        frame = clean_frame(shape=(4, 4, 3))
        with self.assertRaises(CalibrationError):
            build_reference([frame.copy(), frame.copy(), clean_frame(shape=(5, 4, 3))])

    def test_non_bgr_frame(self):
        gray = np.full((4, 4), 100, np.uint8)
        with self.assertRaises(CalibrationError):
            build_reference([gray.copy() for _ in range(3)])


class CalibratorTests(unittest.TestCase):
    def test_add_build_reset_lifecycle(self):
        frame = clean_frame(77, (5, 6, 3))
        calibrator = Calibrator(frame_count=3)
        self.assertEqual(calibrator.frames, [])
        self.assertIsNone(calibrator.reference)
        for _ in range(3):
            calibrator.add_frame(frame)
        self.assertEqual(len(calibrator.frames), 3)
        reference = calibrator.build()
        self.assertIs(calibrator.reference, reference)
        self.assertTrue(np.array_equal(reference, frame))
        calibrator.reset()
        self.assertEqual(calibrator.frames, [])
        self.assertIsNone(calibrator.reference)

    def test_build_before_enough_frames_raises(self):
        calibrator = Calibrator(frame_count=3)
        calibrator.add_frame(clean_frame())
        calibrator.add_frame(clean_frame())
        with self.assertRaises(CalibrationError):
            calibrator.build()

    def test_build_after_reset_raises(self):
        calibrator = Calibrator(frame_count=3)
        for _ in range(3):
            calibrator.add_frame(clean_frame())
        calibrator.build()
        calibrator.reset()
        with self.assertRaises(CalibrationError):
            calibrator.build()

    def test_default_frame_count_requires_thirty_frames(self):
        calibrator = Calibrator()
        self.assertEqual(calibrator.frame_count, 30)
        for _ in range(5):
            calibrator.add_frame(clean_frame())
        with self.assertRaises(CalibrationError):
            calibrator.build()

    def test_frame_count_below_min_raises(self):
        for bad in (0, 1, 2, -5):
            with self.assertRaises(CalibrationError):
                Calibrator(frame_count=bad)


# ------------------------------------------------------------------- file I/O
def test_save_then_load_is_identical(tmp_path):
    reference = (np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3))
    path = tmp_path / 'nested' / 'reference.png'
    written = save_reference(reference, path)
    assert written == path and path.exists()
    loaded = load_reference(path)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, reference)


def test_save_empty_reference_raises(tmp_path):
    with pytest.raises(CalibrationError):
        save_reference(np.zeros((0, 0, 3), np.uint8), tmp_path / 'empty.png')
    with pytest.raises(CalibrationError):
        save_reference(None, tmp_path / 'none.png')


def test_load_missing_or_broken_path_raises(tmp_path):
    with pytest.raises(CalibrationError):
        load_reference(tmp_path / 'does_not_exist.png')
    broken = tmp_path / 'broken.png'
    broken.write_bytes(b'not a png')
    with pytest.raises(CalibrationError):
        load_reference(broken)


if __name__ == '__main__':
    unittest.main()
