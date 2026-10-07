"""接口层验收测试：:mod:`interface.mock_robot` + :mod:`interface.robot_interface`。

验证清洗动作顺序、互锁（cleaner 必须先放下、电机必须先启动）、故障后的
"安全动作仍然可用 / 危险动作被拒绝" 语义、动作日志的拷贝语义，以及 V1.0
不提供速度控制、ABC 抽象方法必须全部实现这两条契约。
"""
import unittest

import pytest

from interface.mock_robot import MockRobot
from interface.robot_interface import RobotInterface
from utils.errors import RobotError


def lowered_and_cleaning():
    robot = MockRobot()
    robot.cleaner_down()
    robot.start_cleaning()
    return robot


class MockRobotSequenceTests(unittest.TestCase):
    def test_required_action_sequence_is_recorded_in_order(self):
        robot = MockRobot()
        robot.cleaner_down()
        robot.start_cleaning()
        robot.execute_pass()
        robot.stop_cleaning()
        robot.cleaner_up()
        self.assertEqual(robot.action_log(), ['cleaner_down', 'start_cleaning',
                                              'cleaning_pass 1', 'stop_cleaning',
                                              'cleaner_up'])

    def test_execute_pass_is_counted_and_logged(self):
        robot = lowered_and_cleaning()
        robot.execute_pass()
        robot.execute_pass()
        self.assertEqual(robot.cleaning_passes, 2)
        self.assertEqual(robot.action_log()[-2:], ['cleaning_pass 1', 'cleaning_pass 2'])

    # ---------------------------------------------------------------- interlock
    def test_start_cleaning_while_raised_raises(self):
        robot = MockRobot()
        with pytest.raises(RobotError):
            robot.start_cleaning()
        self.assertFalse(robot.cleaning)
        self.assertEqual(robot.action_log(), [])

    def test_execute_pass_while_motor_off_raises(self):
        robot = MockRobot()
        robot.cleaner_down()
        with pytest.raises(RobotError):
            robot.execute_pass()
        self.assertEqual(robot.cleaning_passes, 0)
        self.assertNotIn('cleaning_pass 1', robot.action_log())

    def test_double_start_cleaning_raises(self):
        robot = lowered_and_cleaning()
        with pytest.raises(RobotError):
            robot.start_cleaning()
        self.assertEqual(robot.action_log().count('start_cleaning'), 1)

    # ---------------------------------------------------------------- stop/up
    def test_stop_cleaning_when_not_cleaning_is_idempotent(self):
        robot = MockRobot()
        robot.stop_cleaning()
        robot.stop_cleaning()
        self.assertFalse(robot.cleaning)
        self.assertEqual(robot.action_log(), [])

    def test_stop_cleaning_logs_once_and_clears_flag(self):
        robot = lowered_and_cleaning()
        robot.stop_cleaning()
        robot.stop_cleaning()
        self.assertFalse(robot.cleaning)
        self.assertEqual(robot.action_log().count('stop_cleaning'), 1)

    def test_cleaner_up_clears_lowered_flag(self):
        robot = MockRobot()
        robot.cleaner_down()
        self.assertTrue(robot.cleaner_lowered)
        robot.cleaner_up()
        self.assertFalse(robot.cleaner_lowered)
        self.assertEqual(robot.last_action(), 'cleaner_up')

    # ---------------------------------------------------------------- status
    def test_get_status_contents(self):
        robot = lowered_and_cleaning()
        robot.execute_pass()
        status = robot.get_status()
        self.assertEqual(set(status), {'cleaning', 'cleaner_lowered', 'cleaning_passes',
                                        'faulted', 'last_action'})
        self.assertTrue(status['cleaning'])
        self.assertTrue(status['cleaner_lowered'])
        self.assertEqual(status['cleaning_passes'], 1)
        self.assertFalse(status['faulted'])
        self.assertEqual(status['last_action'], 'cleaning_pass 1')

    # ---------------------------------------------------------------- faults
    def test_emergency_stop_blocks_motion_and_arming(self):
        robot = lowered_and_cleaning()
        robot.emergency_stop('motor overheat')
        self.assertTrue(robot.faulted)
        self.assertFalse(robot.cleaning)
        self.assertFalse(robot.cleaner_lowered)
        self.assertEqual(robot.last_action(), 'emergency_stop motor overheat')
        for action in (lambda: robot.move_forward(1.0),
                       lambda: robot.move_backward(1.0),
                       lambda: robot.turn_left(1.0),
                       lambda: robot.turn_right(1.0),
                       robot.cleaner_down,
                       robot.start_cleaning,
                       robot.execute_pass):
            with pytest.raises(RobotError):
                action()

    def test_stop_and_stop_cleaning_still_succeed_while_faulted(self):
        robot = lowered_and_cleaning()
        robot.emergency_stop('estop')
        robot.stop_cleaning()  # 故障时停止清洗必须安全（电机已经是关闭状态）
        robot.stop()
        self.assertEqual(robot.last_action(), 'stop')
        self.assertFalse(robot.cleaning)
        self.assertEqual(robot.action_log().count('stop'), 1)

    def test_clear_fault_restores_operation(self):
        robot = MockRobot()
        robot.emergency_stop('estop')
        robot.clear_fault()
        self.assertFalse(robot.faulted)
        robot.move_forward(0.5)
        robot.cleaner_down()
        robot.start_cleaning()
        robot.execute_pass()
        self.assertEqual(robot.cleaning_passes, 1)
        self.assertIn('move_forward 0.5m', robot.action_log())

    def test_reset_clears_state_and_action_log(self):
        robot = lowered_and_cleaning()
        robot.execute_pass()
        robot.emergency_stop('estop')
        robot.reset()
        self.assertFalse(robot.faulted)
        self.assertFalse(robot.cleaning)
        self.assertFalse(robot.cleaner_lowered)
        self.assertEqual(robot.cleaning_passes, 0)
        self.assertEqual(robot.action_log(), [])
        self.assertIsNone(robot.last_action())
        robot.cleaner_down()  # 故障已清除，可继续使用

    # ---------------------------------------------------------------- action log
    def test_action_log_helpers(self):
        robot = MockRobot()
        self.assertIsNone(robot.last_action())
        self.assertEqual(robot.actions_since(0), [])
        robot.cleaner_down()
        robot.start_cleaning()
        robot.execute_pass()
        self.assertEqual(robot.last_action(), 'cleaning_pass 1')
        self.assertEqual(robot.actions_since(0), ['cleaner_down', 'start_cleaning',
                                                  'cleaning_pass 1'])
        self.assertEqual(robot.actions_since(2), ['cleaning_pass 1'])
        self.assertEqual(robot.actions_since(99), [])
        robot.clear_action_log()
        self.assertEqual(robot.action_log(), [])
        self.assertIsNone(robot.last_action())

    def test_action_log_is_a_copy(self):
        robot = MockRobot()
        robot.cleaner_down()
        snapshot = robot.action_log()
        snapshot.append('tampered')
        self.assertEqual(robot.action_log(), ['cleaner_down'])
        since = robot.actions_since(0)
        since.clear()
        self.assertEqual(robot.action_log(), ['cleaner_down'])

    # ---------------------------------------------------------------- V1 scope
    def test_set_speed_is_not_implemented(self):
        robot = MockRobot()
        with pytest.raises(NotImplementedError):
            robot.set_speed(0.5)


class RobotInterfaceContractTests(unittest.TestCase):
    def test_incomplete_subclass_cannot_be_instantiated(self):
        class Incomplete(RobotInterface):
            def move_forward(self, distance=None):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_mock_robot_implements_the_full_interface(self):
        robot = MockRobot()
        self.assertIsInstance(robot, RobotInterface)
        for name in ('move_forward', 'move_backward', 'turn_left', 'turn_right',
                     'stop', 'cleaner_down', 'cleaner_up', 'start_cleaning',
                     'stop_cleaning', 'execute_pass', 'get_status', 'action_log',
                     'clear_action_log'):
            self.assertTrue(callable(getattr(robot, name)), name)


if __name__ == '__main__':
    unittest.main()
