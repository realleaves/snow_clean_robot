import unittest
from decision.cleaning_task import CleaningTask
from decision.priority_scheduler import PriorityScheduler
from mapping.pose_provider import RobotPose


class PriorityTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = PriorityScheduler(dict(pollution_weight=.6, distance_weight=.25,
            waiting_weight=.15, max_distance_m=4, starvation_seconds=120))
        self.pose = RobotPose(0, 0, 0)

    def test_severity_distance_and_wait(self):
        a = CleaningTask(1, 1, 0, .9, 'HEAVY', created_at=100)
        b = CleaningTask(2, 1, 0, .3, 'LIGHT', created_at=100)
        self.assertGreater(self.scheduler.calculate(a, self.pose, now=100),
                           self.scheduler.calculate(b, self.pose, now=100))
        b.pollution_score = .9
        b.target_x = 3
        self.assertGreater(self.scheduler.calculate(a, self.pose, now=100),
                           self.scheduler.calculate(b, self.pose, now=100))
        self.assertGreater(self.scheduler.calculate(a, self.pose, now=500),
                           self.scheduler.calculate(a, self.pose, now=100))
