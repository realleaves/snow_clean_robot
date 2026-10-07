"""tests/test_pose_provider.py -- mock pose source used by mapping/planning."""
import math

import pytest

from mapping.pose_provider import MockPoseProvider, PoseProvider, RobotPose, normalize_yaw
from utils.errors import ConfigError


def test_set_and_get_pose():
    provider = MockPoseProvider()
    provider.set_pose(2.0, 3.0, 0.0)
    pose = provider.get_pose()
    assert (pose.x, pose.y, pose.yaw) == (2.0, 3.0, 0.0)
    assert provider.updates == 1


@pytest.mark.parametrize('yaw', [0.0, math.pi / 2, math.pi, -math.pi / 2, 2 * math.pi,
                                 -math.pi, 3 * math.pi])
def test_yaw_is_wrapped_into_minus_pi_pi(yaw):
    provider = MockPoseProvider()
    provider.set_pose(0.0, 0.0, yaw)
    assert abs(provider.get_pose().yaw) <= math.pi + 1e-12
    assert provider.get_pose().yaw == pytest.approx(normalize_yaw(yaw))


def test_default_and_initial_pose():
    assert MockPoseProvider().get_pose() == RobotPose(0.0, 0.0, 0.0)
    assert MockPoseProvider(1, 2, math.pi / 4).get_pose().yaw == pytest.approx(math.pi / 4)


def test_get_pose_returns_a_copy():
    provider = MockPoseProvider()
    pose = provider.get_pose()
    pose.x = 99.0
    assert provider.get_pose().x == 0.0


def test_move_by_translates_along_heading():
    provider = MockPoseProvider(0.0, 0.0, 0.0)
    provider.move_by(2.0)
    assert provider.get_pose().x == pytest.approx(2.0)
    assert provider.get_pose().y == pytest.approx(0.0)
    provider = MockPoseProvider(0.0, 0.0, math.pi / 2)
    moved = provider.move_by(1.0)
    assert moved.x == pytest.approx(0.0, abs=1e-9)
    assert moved.y == pytest.approx(1.0)


def test_move_by_supports_lateral_offset():
    provider = MockPoseProvider(0.0, 0.0, 0.0)
    provider.move_by(0.0, 1.0)
    assert provider.get_pose().y == pytest.approx(1.0)


def test_reset_returns_to_origin():
    provider = MockPoseProvider(4.0, -2.0, 1.0)
    provider.reset()
    assert provider.get_pose() == RobotPose(0.0, 0.0, 0.0)


@pytest.mark.parametrize('values', [(float('nan'), 0, 0), (0, float('inf'), 0),
                                    (0, 0, float('nan'))])
def test_non_finite_pose_is_rejected(values):
    with pytest.raises(ConfigError):
        MockPoseProvider().set_pose(*values)


def test_normalize_yaw_rejects_non_finite():
    with pytest.raises(ConfigError):
        normalize_yaw(float('nan'))


def test_base_pose_provider_is_abstract():
    with pytest.raises(TypeError):
        PoseProvider()
    assert PoseProvider.is_valid(object()) is True  # default implementation
