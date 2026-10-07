"""Composition root and deterministic mock closed-loop runner."""
import logging
from pathlib import Path
import cv2
import yaml
from camera.realsense_camera import MockCamera
from decision.cleaning_task import CleaningTask
from decision.cleaning_policy import CleaningPolicy
from decision.priority_scheduler import PriorityScheduler
from decision.state_machine import State, StateMachine
from decision.task_manager import TaskManager
from feedback.cleaning_evaluator import CleaningEvaluator
from feedback.compensation import next_action
from fusion.humidity_processor import normalize_humidity
from fusion.pollution_classifier import PollutionClassifier
from fusion.pollution_score import PollutionScorer
from interface.humidity_interface import MockHumiditySensor
from interface.mock_robot import MockRobot
from mapping.coordinate_transform import pixel_to_robot_ground, robot_to_world
from mapping.grid_map import GridMap
from mapping.pollution_map import PollutionMap
from mapping.pose_provider import MockPoseProvider
from perception.ground_roi import GroundROI
from perception.wet_region_detector import WetRegionDetector
from planning.astar import AStar
from planning.path_planner import PathPlanner
from utils.logger import configure_logger
from visualization.system_monitor import SystemMonitor


class SnowCleanSystem:
    def __init__(self, mode: str = 'mock', headless: bool | None = None, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parent
        self.config = {name: yaml.safe_load((self.root / 'config' / f'{name}.yaml').read_text())
                       for name in ('system', 'camera', 'perception', 'fusion', 'planner', 'cleaning')}
        self.mode = mode
        self.headless = self.config['system']['headless'] if headless is None else headless
        self.running = False
        self.machine = StateMachine()
        self.frame = None
        self.regions = []
        self.region = None
        self.task = None
        self.path = []
        self.log = logging.getLogger('system')

    def initialize(self) -> None:
        cfg = self.config
        configure_logger(self.root, cfg['system']['logging']['level'])
        self.log = logging.getLogger('system')
        if self.mode != 'mock':
            # A true pose source, camera extrinsics, humidity transport and robot
            # control protocol are required before movement can be enabled.
            raise RuntimeError('real mode requires calibrated PoseProvider, camera extrinsics, humidity adapter and RobotInterface; no hardware motion is enabled')
        camera_cfg = cfg['camera']
        self.camera = MockCamera(camera_cfg['camera']['width'], camera_cfg['camera']['height'])
        self.pose = MockPoseProvider()
        self.robot = MockRobot(self.pose)
        self.humidity = MockHumiditySensor('wet')
        self.roi = GroundROI(camera_cfg['ground_roi'], camera_cfg['depth'])
        self.scorer = PollutionScorer(cfg['fusion']['fusion'])
        self.classifier = PollutionClassifier(**cfg['fusion']['pollution_level'])
        self.detector = WetRegionDetector(cfg['perception']['perception'], cfg['fusion']['fusion']['max_area_ratio'])
        self.grid = GridMap.from_config(cfg['planner']['map'])
        self.planner = PathPlanner(self.grid, AStar(cfg['planner']['planner']['allow_diagonal']))
        self.pollution_map = PollutionMap(cfg['planner']['pollution_map']['merge_distance_m'])
        self.tasks = TaskManager(PriorityScheduler(cfg['planner']['priority']))
        self.policy = CleaningPolicy(cfg['cleaning']['cleaning'])
        self.evaluator = CleaningEvaluator(cfg['cleaning']['feedback']['success_threshold'])
        self.monitor = SystemMonitor(self.headless)
        self.camera.start()
        self.running = True
        self.machine.transition(State.CALIBRATION)
        self.log.info('mode=%s state=%s', self.mode, self.machine.state.value)

    def _detect(self):
        self.frame = self.camera.get_frame()
        image, depth, offset = self.roi.extract(self.frame)
        self.regions = self.detector.detect(image, depth, self.roi.valid_depth(depth), offset)
        self.monitor.show(self.frame, self.detector.last_mask, self.regions,
                          self.machine.state.value, self.task, self.camera.measured_fps)
        return self.regions

    def _clean(self) -> None:
        passes = self.policy.passes(self.task.pollution_level, self.task.retry_count > 0)
        self.robot.cleaner_down()
        try:
            self.robot.start_cleaning()
            for _ in range(passes):
                self.robot.execute_pass()
            self.camera.clean(passes)
        finally:
            self.robot.stop_cleaning()
            self.robot.cleaner_up()
        self.log.info('task_id=%s cleaning_passes=%s retry=%s', self.task.task_id, passes, self.task.retry_count)

    def _follow_mock_path(self, path: list[tuple[float, float]]) -> None:
        """Execute a planned path in the synthetic world and update mock pose."""
        import math
        for x, y in path[1:]:
            current = self.pose.get_pose()
            dx, dy = x - current.x, y - current.y
            distance = math.hypot(dx, dy)
            if distance < 1e-9:
                continue
            heading = math.atan2(dy, dx)
            delta = (heading - current.yaw + math.pi) % (2 * math.pi) - math.pi
            if delta >= 0:
                self.robot.turn_left(delta)
            else:
                self.robot.turn_right(-delta)
            self.robot.move_forward(distance)
            self.pose.set_pose(x, y, heading)
        self.robot.stop()

    def update(self) -> None:
        if not self.running:
            return
        try:
            self._update_state()
        except Exception:
            self.log.exception('state=%s task_id=%s', self.machine.state.value,
                               self.task.task_id if self.task else None)
            self.machine.transition(State.ERROR)
            self.robot.stop_cleaning()
            self.robot.cleaner_up()
            self.robot.stop()
            self.running = False
            raise

    def _update_state(self) -> None:
        state = self.machine.state
        if state == State.CALIBRATION:
            images = []
            for _ in range(self.config['camera']['calibration']['frame_count']):
                frame = self.camera.get_frame()
                images.append(self.roi.extract(frame)[0].copy())
            reference = self.detector.build_reference(images)
            path = self.root / self.config['camera']['calibration']['reference_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), reference)
            self.camera.clean_reference_mode = False
            self.machine.transition(State.PATROL)
        elif state == State.PATROL:
            self.machine.transition(State.DETECT)
        elif state == State.DETECT:
            regions = self._detect()
            if not regions:
                self.machine.transition(State.PATROL)
            else:
                self.region = max(regions, key=lambda r: r.area_px)
                self.machine.transition(State.APPROACH)
        elif state == State.APPROACH:
            # Mock extrinsics are known by construction. Real extrinsics must be measured.
            self.region.relative_x, self.region.relative_y = pixel_to_robot_ground(
                *self.region.center_px, self.region.depth_m, self.frame.intrinsics,
                self.camera.camera_to_robot)
            target_x, target_y = robot_to_world(self.region.relative_x,
                                                 self.region.relative_y, self.pose.get_pose())
            approach = CleaningTask(0, target_x, target_y, 0, 'LIGHT')
            path = self.planner.plan_to_task(self.pose.get_pose(), approach)
            if not path:
                raise RuntimeError('no safe path to humidity sampling point')
            self._follow_mock_path(path)
            self.machine.transition(State.HUMIDITY_CONFIRM)
        elif state == State.HUMIDITY_CONFIRM:
            self.raw_humidity = self.humidity.read_raw()
            hcfg = self.config['fusion']['humidity']
            self.humidity_score = normalize_humidity(self.raw_humidity, hcfg['dry_raw'], hcfg['wet_raw'])
            self.machine.transition(State.CLASSIFY)
        elif state == State.CLASSIFY:
            region = self.region
            region.pollution_score = self.scorer.score(region.visual_score, self.humidity_score, region.area_score)
            region.pollution_level = self.classifier.classify(region.pollution_score).name
            self.log.info('region=%s V=%.3f H=%.3f A=%.3f S=%.3f level=%s', region.region_id,
                          region.visual_score, self.humidity_score, region.area_score,
                          region.pollution_score, region.pollution_level)
            self.machine.transition(State.CREATE_TASK)
        elif state == State.CREATE_TASK:
            region = self.region
            x, y = robot_to_world(0.0, 0.0, self.pose.get_pose())
            target = self.pollution_map.upsert(x, y, region.pollution_score, region.pollution_level)
            if target.id not in self.tasks.tasks:
                self.tasks.add_task(CleaningTask(target.id, x, y, region.pollution_score,
                    region.pollution_level, before_area=region.area_px, region_id=region.region_id))
            self.machine.transition(State.PLANNING)
        elif state == State.PLANNING:
            self.task = self.tasks.get_next_task(self.pose.get_pose())
            if self.task is None:
                self.machine.transition(State.PATROL)
                return
            self.task.status = 'PLANNING'
            self.path = self.planner.plan_to_task(self.pose.get_pose(), self.task)
            if not self.path:
                self.tasks.fail(self.task.task_id)
                self.log.error('task_id=%s no path', self.task.task_id)
                self.machine.transition(State.PATROL)
            else:
                self.machine.transition(State.NAVIGATING)
        elif state == State.NAVIGATING:
            self.task.status = 'NAVIGATING'
            self._follow_mock_path(self.path)
            self.machine.transition(State.CLEANING)
        elif state == State.CLEANING:
            self.task.status = 'CLEANING'
            self._clean()
            self.machine.transition(State.RECHECK)
        elif state == State.RECHECK:
            self.task.status = 'RECHECK'
            after = max((r.area_px for r in self._detect()), default=0.0)
            result = self.evaluator.evaluate(self.task.before_area, after)
            self.task.after_area, self.task.cleaning_efficiency = after, result.efficiency
            self.log.info('task_id=%s before=%.0f after=%.0f efficiency=%s pass=%s',
                          self.task.task_id, self.task.before_area, after, result.efficiency, result.passed)
            if result.passed:
                self.tasks.complete(self.task.task_id)
                self.pollution_map.complete(self.task.task_id)
                self.camera.pollution_fraction = 0
                self.machine.transition(State.PATROL)
            else:
                self.machine.transition(State.COMPENSATE)
        elif state == State.COMPENSATE:
            action = next_action(self.task, self.config['cleaning']['cleaning']['max_retry'])
            self.log.info('task_id=%s action=%s retry=%s', self.task.task_id, action, self.task.retry_count)
            self.machine.transition(State.CLEANING if action == 'COMPENSATE' else State.PATROL)
        else:
            raise RuntimeError(f'unhandled state {state.value}')
        self.log.info('state=%s task_id=%s', self.machine.state.value,
                      self.task.task_id if self.task else None)

    def shutdown(self) -> None:
        self.running = False
        if hasattr(self, 'robot'):
            self.robot.stop_cleaning()
            self.robot.cleaner_up()
            self.robot.stop()
        if hasattr(self, 'camera'):
            self.camera.stop()
        if hasattr(self, 'monitor'):
            self.monitor.close()
        self.machine.state = State.SHUTDOWN
