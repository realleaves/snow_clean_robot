"""tests/test_state_machine.py -- system state machine, normal / compensation / error paths."""
import unittest

from camera.mock_camera import MockCamera
from camera.virtual_scene import VirtualWorld
from decision.cleaning_task import CleaningTask
from decision.state_machine import ALLOWED, CYCLE_END_STATES, State, StateMachine
from feedback.compensation import next_action
from system import SnowCleanSystem
from tests.helpers import contaminated_system, make_system
from utils.errors import InvalidFrameError

NORMAL_PATH = [State.CALIBRATION, State.PATROL, State.DETECT, State.APPROACH,
               State.HUMIDITY_CONFIRM, State.CLASSIFY, State.CREATE_TASK, State.PLANNING,
               State.NAVIGATING, State.CLEANING, State.RECHECK, State.PATROL]


def dirty_system(**kwargs):
    return contaminated_system(**kwargs)


class TransitionTests(unittest.TestCase):
    def test_normal_path_is_allowed(self):
        machine = StateMachine()
        for target in NORMAL_PATH:
            machine.transition(target)
        self.assertEqual(machine.state, State.PATROL)
        self.assertEqual(machine.history[0], State.INIT)
        self.assertIn(State.RECHECK, machine.history)

    def test_illegal_transitions_are_rejected(self):
        machine = StateMachine()
        with self.assertRaises(ValueError):
            machine.transition(State.CLEANING)
        machine.transition(State.CALIBRATION)
        with self.assertRaises(ValueError):
            machine.transition(State.NAVIGATING)
        with self.assertRaises(ValueError) as ctx:
            machine.transition(State.DETECT)
        self.assertIn('CALIBRATION -> DETECT', str(ctx.exception))

    def test_error_is_reachable_from_every_live_state(self):
        for state in ALLOWED:
            if state in (State.ERROR, State.SHUTDOWN):
                continue
            machine = StateMachine()
            machine.state = state
            machine.transition(State.ERROR)
            self.assertEqual(machine.state, State.ERROR)

    def test_error_only_leaves_for_shutdown(self):
        machine = StateMachine()
        machine.transition(State.ERROR)
        with self.assertRaises(ValueError):
            machine.transition(State.PATROL)
        machine.transition(State.SHUTDOWN)
        self.assertEqual(machine.state, State.SHUTDOWN)

    def test_shutdown_is_terminal(self):
        machine = StateMachine()
        machine.transition(State.ERROR)
        machine.transition(State.SHUTDOWN)
        for target in (State.PATROL, State.ERROR, State.CALIBRATION):
            with self.assertRaises(ValueError):
                machine.transition(target)

    def test_can_transition_and_fail_helper(self):
        machine = StateMachine()
        self.assertTrue(machine.can_transition(State.CALIBRATION))
        self.assertFalse(machine.can_transition(State.CLEANING))
        machine.fail('boom')
        self.assertEqual(machine.state, State.ERROR)
        self.assertEqual(machine.error_reason, 'boom')
        machine.reset()
        self.assertEqual(machine.state, State.INIT)
        self.assertIsNone(machine.error_reason)
        self.assertFalse(machine.visited(State.ERROR))

    def test_manual_check_state_exists(self):
        machine = StateMachine()
        machine.transition(State.CALIBRATION)
        machine.transition(State.PATROL)
        machine.transition(State.DETECT)
        machine.transition(State.APPROACH)
        machine.transition(State.HUMIDITY_CONFIRM)
        machine.transition(State.CLASSIFY)
        machine.transition(State.CREATE_TASK)
        machine.transition(State.PLANNING)
        machine.transition(State.NAVIGATING)
        machine.transition(State.CLEANING)
        machine.transition(State.RECHECK)
        machine.transition(State.MANUAL_CHECK)
        machine.transition(State.PATROL)
        self.assertIn(State.MANUAL_CHECK, machine.history)
        self.assertIn(State.PATROL, CYCLE_END_STATES)


class CompensationUnitTests(unittest.TestCase):
    def test_retry_limit(self):
        task = CleaningTask(1, 0, 0, .5, 'MEDIUM')
        self.assertEqual(next_action(task, 2), 'COMPENSATE')
        self.assertEqual(next_action(task, 2), 'COMPENSATE')
        self.assertEqual(next_action(task, 2), 'MANUAL_CHECK')
        self.assertEqual(task.retry_count, 2)


class SystemPathTests(unittest.TestCase):
    def test_mock_end_to_end_compensation(self):
        system = dirty_system()
        try:
            system.initialize()
            for _ in range(30):
                system.update()
                if system.task and system.task.status == 'COMPLETED':
                    break
            self.assertIsNotNone(system.task)
            self.assertEqual(system.task.status, 'COMPLETED')
            self.assertEqual(system.task.retry_count, 1)
            self.assertEqual(system.robot.cleaning_passes, 2)
            self.assertFalse(system.robot.cleaning)
            self.assertFalse(system.robot.cleaner_lowered)
            self.assertEqual(system.error_count, 0)
            visited = [state.value for state in system.machine.history]
            self.assertIn('COMPENSATE', visited)
            self.assertIn('RECHECK', visited)
        finally:
            system.shutdown()

    def test_clean_scene_patrols_without_creating_tasks(self):
        system = make_system()
        try:
            system.initialize()
            for _ in range(6):
                system.update()
            self.assertEqual(len(system.tasks.tasks), 0)
            self.assertEqual(system.machine.state, State.PATROL)
            self.assertEqual(system.error_count, 0)
            self.assertEqual(system.robot.cleaning_passes, 0)
        finally:
            system.shutdown()

    def test_manual_check_path_terminates(self):
        world = VirtualWorld(640, 480)
        world.add_target(0.0, 0.0, 0.05, depth_m=0.8)
        for target in world.targets:
            target.freeze_after_pass = True
        world.detection_hint = (0.60, 0.10)
        camera = MockCamera(640, 480, 30, world=world)
        system = SnowCleanSystem(camera=camera, world=world, humidity_mode='medium',
                                 headless=True)
        try:
            system.initialize()
            for _ in range(30):
                system.update()
                if system.task is not None and system.task.terminal:
                    break
            self.assertEqual(system.task.status, 'MANUAL_CHECK')
            self.assertEqual(system.task.retry_count, 2)
            self.assertIn('MANUAL_CHECK', [s.value for s in system.machine.history])
            self.assertLess(system.step_count, 30)
        finally:
            system.shutdown()

    def test_camera_error_moves_to_error_and_does_not_crash(self):
        system = dirty_system()
        try:
            system.initialize()
            system.camera.failure_hook = lambda index: (
                (_ for _ in ()).throw(InvalidFrameError('injected empty frame'))
                if index >= 30 else None)
            for _ in range(6):
                system.update()
            self.assertEqual(system.machine.state, State.ERROR)
            self.assertFalse(system.running)
            self.assertTrue(system.last_error)
            self.assertFalse(system.robot.cleaning)
            self.assertFalse(system.robot.cleaner_lowered)
        finally:
            system.shutdown()

    def test_raise_on_error_mode_propagates(self):
        system = dirty_system(raise_on_error=True)
        try:
            system.initialize()
            system.camera.failure_hook = lambda index: (
                (_ for _ in ()).throw(InvalidFrameError('injected empty frame'))
                if index >= 30 else None)
            with self.assertRaises(InvalidFrameError):
                for _ in range(4):
                    system.update()
        finally:
            system.shutdown()


if __name__ == '__main__':
    unittest.main()
