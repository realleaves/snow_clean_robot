import unittest
from decision.state_machine import State, StateMachine
from decision.cleaning_task import CleaningTask
from feedback.compensation import next_action
from system import SnowCleanSystem


class StateTests(unittest.TestCase):
    def test_invalid_transition_and_error(self):
        machine = StateMachine()
        with self.assertRaises(ValueError):
            machine.transition(State.CLEANING)
        machine.transition(State.ERROR)
        self.assertEqual(machine.state, State.ERROR)

    def test_retry_limit(self):
        task = CleaningTask(1, 0, 0, .5, 'MEDIUM')
        self.assertEqual(next_action(task, 2), 'COMPENSATE')
        self.assertEqual(next_action(task, 2), 'COMPENSATE')
        self.assertEqual(next_action(task, 2), 'MANUAL_CHECK')
        self.assertEqual(task.retry_count, 2)

    def test_mock_end_to_end(self):
        system = SnowCleanSystem(headless=True)
        try:
            system.initialize()
            for _ in range(30):
                system.update()
                if system.task and system.task.status == 'COMPLETED':
                    break
            self.assertEqual(system.task.status, 'COMPLETED')
            self.assertEqual(system.task.retry_count, 1)
            self.assertEqual(system.robot.cleaning_passes, 2)
            self.assertFalse(system.robot.cleaning)
            self.assertFalse(system.robot.cleaner_lowered)
        finally:
            system.shutdown()
