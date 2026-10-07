"""Composition root and deterministic mock closed-loop runner."""
import logging
import math
from pathlib import Path

from camera.mock_camera import MockCamera
from camera.virtual_scene import VirtualWorld
from decision.cleaning_policy import CleaningPolicy
from decision.cleaning_task import CleaningTask, TaskStatus
from decision.priority_scheduler import PriorityScheduler
from decision.state_machine import State, StateMachine
from decision.task_manager import TaskManager
from feedback.cleaning_evaluator import CleaningEvaluator
from feedback.compensation import COMPENSATE, next_action
from fusion.humidity_processor import normalize_humidity
from fusion.pollution_classifier import PollutionClassifier
from fusion.pollution_score import PollutionScorer
from interface.humidity_interface import MockHumiditySensor
from interface.mock_robot import MockRobot
from mapping.coordinate_transform import pixel_to_robot_ground, robot_to_world
from mapping.grid_map import GridMap
from mapping.pollution_map import PollutionMap
from mapping.pose_provider import MockPoseProvider
from perception.calibration import save_reference
from perception.ground_roi import GroundROI
from perception.wet_region_detector import WetRegionDetector
from planning.astar import AStar
from planning.path_planner import PathPlanner
from utils.config_loader import load_config
from utils.errors import (CameraError, ConfigError, InvalidFrameError, NoPathError,
                          PerceptionError, RobotError, SnowCleanError, TaskError)
from utils.logger import configure_logger
from visualization.system_monitor import SystemMonitor

#: Fixed inspection station where the clean reference image stays valid.
DETECTION_STATION = (0.0, 0.0, 0.0)
#: Consecutive re-detections of a blacklisted area before the mock loop idles.
MAX_BLOCKED_REDETECTIONS = 2

#: Errors that mean "one task failed" rather than "the whole system is broken".
RECOVERABLE_ERRORS = (TaskError,)
#: Errors that stop the closed loop and move the machine to ERROR.
CRITICAL_ERRORS = (CameraError, InvalidFrameError, ConfigError, PerceptionError,
                   RobotError, NoPathError)


class SnowCleanSystem:
    """Wires every V1.0 module together and owns the mock closed loop."""

    def __init__(self, mode: str = 'mock', headless: bool | None = None,
                 root: Path | None = None, camera=None, world: VirtualWorld | None = None,
                 humidity_mode: str = 'wet', raise_on_error: bool = False,
                 setup_hook=None):
        self.root = Path(root) if root else Path(__file__).resolve().parent
        self.config = load_config(self.root)
        self.mode = mode
        self.headless = bool(self.config['system']['headless'] if headless is None else headless)
        self.raise_on_error = bool(raise_on_error)
        self._camera_override = camera
        self._world_override = world
        self._humidity_mode = humidity_mode
        self._setup_hook = setup_hook
        self.running = False
        self.initialized = False
        self.shutdown_complete = False
        self.step_count = 0
        self._healthy_cycles = 0
        self.last_error: str | None = None
        self.error_count = 0
        self.machine = StateMachine()
        self.frame = None
        self.regions = []
        self.region = None
        self.task = None
        self.path = []
        self.humidity_score = 0.0
        self.raw_humidity = None
        self._region_target: tuple[float, float] | None = None
        self._frame_pose = None
        #: World positions whose task already ended in FAILED/MANUAL_CHECK; the same
        #: floor area is not turned into a new task automatically.
        self.blacklisted_targets: list[tuple[float, float]] = []
        self.skipped_detections = 0
        self.blocked_streak = 0
        self.sleeping = False
        self.log = logging.getLogger('system')

    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        if self.initialized:
            raise RuntimeError('system is already initialized')
        cfg = self.config
        configure_logger(self.root, cfg['system']['logging']['level'])
        self.log = logging.getLogger('system')
        if self.mode != 'mock':
            # A real pose source, camera extrinsics, humidity transport and robot
            # control protocol are required before movement can be enabled.
            raise ConfigError(
                'real mode requires a calibrated PoseProvider, camera extrinsics, '
                'humidity adapter and RobotInterface; no hardware motion is enabled')
        camera_cfg = cfg['camera']
        self.world = self._world_override or VirtualWorld(
            camera_cfg['camera']['width'], camera_cfg['camera']['height'])
        self.camera = self._camera_override or MockCamera(
            camera_cfg['camera']['width'], camera_cfg['camera']['height'],
            camera_cfg['camera']['fps'], world=self.world)
        self.pose = MockPoseProvider()
        self.robot = MockRobot(self.pose)
        self.humidity = MockHumiditySensor(self._humidity_mode)
        self.roi = GroundROI(camera_cfg['ground_roi'], camera_cfg['depth'])
        self.depth_cfg = camera_cfg['depth']
        self.scorer = PollutionScorer(cfg['fusion']['fusion'])
        self.classifier = PollutionClassifier(**cfg['fusion']['pollution_level'])
        self.detector = WetRegionDetector(
            cfg['perception']['perception'], cfg['fusion']['fusion']['max_area_ratio'],
            frame_count=camera_cfg['calibration'].get('frame_count'))
        self.grid = GridMap.from_config(cfg['planner']['map'])
        self.planner = PathPlanner(self.grid, AStar(cfg['planner']['planner']['allow_diagonal']))
        self.pollution_map = PollutionMap(cfg['planner']['pollution_map']['merge_distance_m'])
        self.tasks = TaskManager(PriorityScheduler(cfg['planner']['priority']))
        self.policy = CleaningPolicy(cfg['cleaning']['cleaning'])
        self.evaluator = CleaningEvaluator(cfg['cleaning']['feedback']['success_threshold'])
        self.monitor = SystemMonitor(self.headless)
        self.camera.start()
        self.running = True
        self.initialized = True
        self.machine.transition(State.CALIBRATION)
        self._run_calibration()
        if self._setup_hook is not None:
            self._setup_hook(self)
        self.log.info('mode=%s state=%s', self.mode, self.machine.state.value)

    # ------------------------------------------------------------------ calibration
    def _run_calibration(self) -> None:
        """Collect clean ROI frames, build and persist the reference image."""
        frame_count = int(self.config['camera']['calibration']['frame_count'])
        images = []
        for _ in range(frame_count):
            frame = self.camera.get_frame()
            images.append(self.roi.extract(frame)[0].copy())
        reference = self.detector.build_reference(images)
        path = self.root / self.config['camera']['calibration']['reference_path']
        save_reference(reference, path)
        self.camera.clean_reference_mode = False
        self.reference_path = path
        self.machine.transition(State.PATROL)

    # ------------------------------------------------------------------ helpers
    def _capture(self):
        self.frame = self.camera.get_frame()
        return self.frame

    def region_world_coordinates(self, region):
        """Ground position of a region using depth, extrinsics and the current pose."""
        if region.depth_m is None:
            raise PerceptionError(
                f'region {region.region_id} has no valid depth and cannot be localised')
        forward, left = pixel_to_robot_ground(
            region.center_px[0], region.center_px[1], region.depth_m,
            self.frame.intrinsics, self.camera.camera_to_robot)
        region.relative_x, region.relative_y = forward, left
        pose = self._frame_pose if self._frame_pose is not None else self.pose.get_pose()
        x, y = robot_to_world(forward, left, pose)
        region.world_x, region.world_y = x, y
        return x, y

    def _localise_regions(self) -> None:
        """Fill in world coordinates for every region of the current frame.

        The pose is captured with the frame so later recomputation (for example
        after the robot has driven to the target) stays consistent: a region is
        always anchored to the world position observed from the viewpoint where
        the frame was taken.
        """
        self._frame_pose = self.pose.get_pose()
        for region in self.regions:
            try:
                self.region_world_coordinates(region)
            except PerceptionError:
                region.world_x = region.world_y = None

    def _camera_target_for(self, region):
        """Virtual target whose rendered centre is nearest to *region* (mock bookkeeping)."""
        best, best_key = None, None
        # Generous tolerance: as a stain shrinks its centroid drifts, and a missed
        # match would leave the target uncleaned.
        limit = max(region.width, region.height, 24)
        for target in getattr(self.world, 'targets', []):
            u, v = target.pixel_center(self.camera.intrinsics)
            distance = math.hypot(u - region.center_px[0], v - region.center_px[1])
            if distance > limit:
                continue
            # Prefer a target that still holds pollution: a nearly cleaned blob must
            # not shadow a neighbouring stain that produced the same component.
            key = (target.pollution_fraction <= 0.05, distance)
            if best_key is None or key < best_key:
                best, best_key = target, key
        return best

    def is_over_target(self, x: float, y: float, tolerance_m: float = 0.35) -> bool:
        return any(math.hypot(bx - x, by - y) <= tolerance_m
                   for bx, by in self.blacklisted_targets)

    def _reset_viewpoint(self) -> None:
        """Return the mock robot to the fixed inspection viewpoint.

        The clean reference image is only valid for the fixed camera viewpoint, so
        the mock robot re-checks the floor from the same station instead of from an
        arbitrary post-cleaning pose (see DESIGN_REVIEW.md).
        """
        pose = self.pose.get_pose()
        station_x, station_y, station_yaw = DETECTION_STATION
        if (abs(pose.x - station_x) > 1e-9 or abs(pose.y - station_y) > 1e-9
                or abs(pose.yaw - station_yaw) > 1e-9):
            self.robot.move_backward(math.hypot(pose.x - station_x, pose.y - station_y))
            self.robot.stop()
            # The mock has no return-to-home planner; the fixed inspection station
            # is reached directly so the clean reference image stays valid.
            self.pose.set_pose(station_x, station_y, station_yaw)

    def _detect(self) -> list:
        self._capture()
        image, depth, offset = self.roi.extract(self.frame)
        self.regions = self.detector.detect(image, depth, self.roi.valid_depth(depth), offset)
        self._apply_detection_hint()
        self._localise_regions()
        self.monitor.show(self.frame, self.detector.last_mask, self.regions,
                          self.machine.state.value, self.task, self.camera.measured_fps,
                          region=self.region,
                          path_length=self.planner.path_length_m(self.path) if self.path else None)
        return self.regions

    def _apply_detection_hint(self) -> None:
        """Scenario-only score injection; perception itself never reads this.

        The hint is attached to a virtual target and scales down as that target is
        cleaned, so the recheck still measures a real reduction.
        """
        if not self.regions:
            return
        for region in self.regions:
            target = self._camera_target_for(region)
            hint = self.world.hint_for(target) if target is not None else None
            if hint is None:
                hint = getattr(self.world, 'detection_hint', None)
            if hint is None:
                continue
            region.visual_score, region.area_score = float(hint[0]), float(hint[1])

    def _region_for_target(self, x: float | None = None, y: float | None = None,
                           tolerance_m: float = 0.15):
        """Detected region belonging to the task's own target, if any.

        The recheck must not mistake a *different* stain for leftover pollution, so
        only the region nearest to the target position within *tolerance_m* counts.
        """
        candidates = []
        for region in self.regions:
            if x is None or y is None or region.world_x is None or region.world_y is None:
                candidates.append((0.0, region))
                continue
            distance = math.hypot(region.world_x - x, region.world_y - y)
            if distance <= tolerance_m:
                candidates.append((distance, region))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _target_area(self, x: float | None = None, y: float | None = None,
                     tolerance_m: float = 0.15) -> float:
        region = self._region_for_target(x, y, tolerance_m)
        return float(region.area_px) if region is not None else 0.0

    def _clean(self) -> None:
        passes = self.policy.passes(self.task.pollution_level, self.task.retry_count > 0)
        target = self._camera_target_for(self.region) if self.region is not None else None
        self.robot.cleaner_down()
        try:
            self.robot.start_cleaning()
            for _ in range(passes):
                self.robot.execute_pass()
            self.task.record_cleaning(passes)
            if target is not None:
                self.camera.clean(passes, target.target_id,
                                  getattr(self.world, 'cleaning_retain', 0.45))
        finally:
            self.robot.stop_cleaning()
            self.robot.cleaner_up()
        self.log.info('task_id=%s cleaning_passes=%s retry=%s', self.task.task_id,
                      passes, self.task.retry_count)

    def _follow_mock_path(self, path) -> None:
        """Execute a planned path in the synthetic world and update the mock pose."""
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

    # ------------------------------------------------------------------ stepping
    def update(self) -> None:
        """Run one state machine step; never raises unless ``raise_on_error``."""
        if not self.running:
            return
        self.step_count += 1
        try:
            self._update_state()
        except RECOVERABLE_ERRORS as exc:
            self._handle_recoverable(exc)
        except CRITICAL_ERRORS as exc:
            self._handle_critical(exc)
        except SnowCleanError as exc:
            self._handle_critical(exc)
        except Exception as exc:  # unexpected bug: still must not crash the demo
            self._handle_critical(exc)

    def _handle_recoverable(self, exc: Exception) -> None:
        self.error_count += 1
        self.last_error = str(exc)
        self.log.error('recoverable error state=%s task_id=%s error=%s',
                       self.machine.state.value, self.task.task_id if self.task else None, exc)
        if self.task is not None and not self.task.terminal:
            try:
                self.tasks.fail(self.task.task_id, str(exc))
            except TaskError:
                self.task.status = TaskStatus.FAILED.value
        self._safe_robot_release()
        if self.machine.can_transition(State.PATROL):
            self.machine.transition(State.PATROL)

    def _handle_critical(self, exc: Exception) -> None:
        self.error_count += 1
        self.last_error = str(exc)
        self.log.exception('critical error state=%s task_id=%s',
                           self.machine.state.value, self.task.task_id if self.task else None)
        self._safe_robot_release()
        self.machine.fail(str(exc))
        self.running = False
        if self.raise_on_error:
            raise exc

    def _safe_robot_release(self) -> None:
        robot = getattr(self, 'robot', None)
        if robot is None:
            return
        for action in (robot.stop_cleaning, robot.cleaner_up, robot.stop):
            try:
                action()
            except Exception:  # pragma: no cover - defensive, must never mask the cause
                pass

    # ------------------------------------------------------------------ states
    def _update_state(self) -> None:
        state = self.machine.state
        if state == State.PATROL:
            self.machine.transition(State.DETECT)
        elif state == State.DETECT:
            # Reference differencing is only valid from the fixed station: never
            # take a detection frame from an arbitrary pose (see DESIGN_REVIEW.md).
            self._reset_viewpoint()
            regions = self._detect()
            if regions:
                self._healthy_cycles = 0
            else:
                self._healthy_cycles += 1
                self.blocked_streak = 0
            if not regions:
                self.machine.transition(State.PATROL)
            else:
                self.region = max(regions, key=lambda r: r.area_px)
                self._region_target = (self.region.world_x, self.region.world_y)
                self.machine.transition(State.APPROACH)
        elif state == State.APPROACH:
            approach = CleaningTask(0, self.region.world_x, self.region.world_y, 0.0, 'LIGHT')
            self.path = self.planner.plan_to_task(self.pose.get_pose(), approach)
            if not self.path:
                raise NoPathError('no safe path to the humidity sampling point')
            self._follow_mock_path(self.path)
            self.machine.transition(State.HUMIDITY_CONFIRM)
        elif state == State.HUMIDITY_CONFIRM:
            self.raw_humidity = self.humidity.read_raw()
            humidity_cfg = self.config['fusion']['humidity']
            self.humidity_score = normalize_humidity(
                self.raw_humidity, humidity_cfg['dry_raw'], humidity_cfg['wet_raw'])
            self.log.info('raw_humidity=%s humidity_score=%.3f', self.raw_humidity,
                          self.humidity_score)
            self.machine.transition(State.CLASSIFY)
        elif state == State.CLASSIFY:
            region = self.region
            self.scorer.score_region(region, self.humidity_score)
            self.classifier.classify_region(region)
            region.validate()
            self.log.info('region=%s V=%.3f H=%.3f A=%.3f S=%.3f level=%s', region.region_id,
                          region.visual_score, self.humidity_score, region.area_score,
                          region.pollution_score, region.pollution_level)
            self.machine.transition(State.CREATE_TASK)
        elif state == State.CREATE_TASK:
            region = self.region
            if (self.tasks.is_known_bad_target(region.world_x, region.world_y)
                    or self.is_over_target(region.world_x, region.world_y)):
                self.log.warning('region=%s skipped: an earlier task here already failed',
                                 region.region_id)
                self.pollution_map.upsert(region.world_x, region.world_y,
                                          region.pollution_score, region.pollution_level)
                self.skipped_detections += 1
                self.blocked_streak += 1
                if self.blocked_streak >= MAX_BLOCKED_REDETECTIONS:
                    # The same unreachable/dirty area keeps being detected: idle until
                    # an operator intervenes instead of spinning forever.
                    self.sleeping = True
                    self.running = False
                    self.log.warning('all remaining detections are blocked; system idling')
                self.machine.transition(State.PATROL)
                return
            target, action = self.pollution_map.upsert(
                region.world_x, region.world_y, region.pollution_score, region.pollution_level)
            existing = self.tasks.find_by_target(target.x, target.y)
            if existing is None:
                self.tasks.add_task(CleaningTask(
                    target.id, target.x, target.y, region.pollution_score,
                    region.pollution_level, before_area=region.area_px,
                    region_id=region.region_id))
            self.log.info('pollution_map action=%s target_id=%s total=%s', action, target.id,
                          len(self.pollution_map))
            self.machine.transition(State.PLANNING)
        elif state == State.PLANNING:
            self.blocked_streak = 0
            self.task = self.tasks.get_next_task(self.pose.get_pose())
            if self.task is None:
                self.machine.transition(State.PATROL)
                return
            self.tasks.set_status(self.task.task_id, TaskStatus.PLANNING)
            self.path = self.planner.plan_to_task(self.pose.get_pose(), self.task)
            if not self.path:
                raise NoPathError(self.planner.last_failure or 'no path to task')
            self.machine.transition(State.NAVIGATING)
        elif state == State.NAVIGATING:
            self.tasks.set_status(self.task.task_id, TaskStatus.NAVIGATING)
            self._follow_mock_path(self.path)
            self.machine.transition(State.CLEANING)
        elif state == State.CLEANING:
            self.tasks.set_status(self.task.task_id, TaskStatus.CLEANING)
            self._clean()
            self.machine.transition(State.RECHECK)
        elif state == State.RECHECK:
            if self.task.terminal:
                # Already closed (e.g. the recheck budget was exhausted earlier).
                self.machine.transition(State.PATROL)
                return
            self.tasks.set_status(self.task.task_id, TaskStatus.RECHECK)
            # The clean reference image is only valid at the fixed inspection
            # viewpoint, so return there *before* taking the recheck measurement.
            self._reset_viewpoint()
            self._detect()
            target_x, target_y = self._region_target or (None, None)
            after = self._target_area(target_x, target_y)
            result = self.evaluator.evaluate(self.task.before_area, after)
            self.task.record_recheck(self.task.before_area, after, result.efficiency)
            self.log.info('task_id=%s before=%s after=%.0f efficiency=%s verdict=%s',
                          self.task.task_id, self.task.before_area, after,
                          result.efficiency, result.reason)
            if result.passed:
                self.tasks.complete(self.task.task_id)
                self.pollution_map.complete(self.task.task_id)
                self.machine.transition(State.PATROL)
            else:
                action = next_action(self.task, self.config['cleaning']['cleaning']['max_retry'])
                self.log.info('task_id=%s action=%s retry=%s', self.task.task_id,
                              action, self.task.retry_count)
                if action == COMPENSATE:
                    self.machine.transition(State.COMPENSATE)
                else:
                    # next_action already set the MANUAL_CHECK status; do not
                    # attempt an illegal MANUAL_CHECK -> MANUAL_CHECK transition.
                    self.machine.transition(State.MANUAL_CHECK)
        elif state == State.COMPENSATE:
            self.machine.transition(State.PLANNING)
        elif state == State.MANUAL_CHECK:
            self.log.warning('task_id=%s requires manual inspection', self.task.task_id)
            self.blacklisted_targets.append((self.task.target_x, self.task.target_y))
            self.machine.transition(State.PATROL)
        elif state == State.SHUTDOWN:
            self.running = False
        else:
            raise TaskError(f'unhandled state {state.value}')
        self.log.info('state=%s task_id=%s task_status=%s', self.machine.state.value,
                      self.task.task_id if self.task else None,
                      self.task.status if self.task else None)

    # ------------------------------------------------------------------ runners
    def run(self, max_steps: int | None = None, stop_on_terminal: bool = True) -> dict:
        """Run the closed loop and return a compact summary dictionary."""
        if not self.initialized:
            self.initialize()
        limit = int(max_steps if max_steps is not None else
                    self.config['system']['loop']['max_steps'])
        for _ in range(limit):
            if not self.running:
                break
            self.update()
            if stop_on_terminal and self.task is not None and self.task.terminal:
                break
        return self.summary()

    def summary(self) -> dict:
        tasks = list(self.tasks.tasks.values()) if hasattr(self, 'tasks') else []
        return dict(
            mode=self.mode,
            state=self.machine.state.value,
            steps=self.step_count,
            running=self.running,
            sleeping=self.sleeping,
            errors=self.error_count,
            last_error=self.last_error,
            tasks=len(tasks),
            completed=sum(1 for t in tasks if t.status == TaskStatus.COMPLETED.value),
            manual_check=sum(1 for t in tasks if t.status == TaskStatus.MANUAL_CHECK.value),
            failed=sum(1 for t in tasks if t.status == TaskStatus.FAILED.value),
            cleaning_passes=getattr(getattr(self, 'robot', None), 'cleaning_passes', 0),
            task_status=self.task.status if self.task else None,
            task_retry=self.task.retry_count if self.task else None,
            task_level=self.task.pollution_level if self.task else None,
            task_score=self.task.pollution_score if self.task else None,
            task_efficiency=self.task.cleaning_efficiency if self.task else None,
            reference=str(getattr(self, 'reference_path', '')),
        )

    def shutdown(self) -> None:
        self.running = False
        self._safe_robot_release()
        camera = getattr(self, 'camera', None)
        if camera is not None:
            try:
                camera.stop()
            except Exception:  # pragma: no cover - defensive
                pass
        monitor = getattr(self, 'monitor', None)
        if monitor is not None:
            monitor.close()
        if self.initialized:
            self.machine.state = State.SHUTDOWN
        self.shutdown_complete = True
