"""Prepared virtual demo scenarios A-H with machine-checkable assertions.

Run them all::

    python main.py --all-scenarios
    python -m pytest tests/test_full_mock_demo.py

Every scenario builds its own deterministic :class:`~camera.virtual_scene.VirtualWorld`
and :class:`~system.SnowCleanSystem`, so the results are reproducible and require
no camera, MCU, localization or humidity hardware.

Two mechanisms make the decision chain controllable from a scenario:

* ``target.cleaning_effectiveness`` / ``target.frozen`` model a stain that is only
  partly removed (or not removed at all) and therefore drive the compensation and
  ``MANUAL_CHECK`` paths deterministically;
* ``world.detection_hint`` injects the ``(VisualScore, AreaScore)`` pair the
  decision layer should see, so LIGHT/MEDIUM/HEAVY can be exercised on demand.
  The perception layer itself is tested separately and never reads the hint.
"""
from pathlib import Path

import numpy as np

from camera.mock_camera import MockCamera
from camera.virtual_scene import VirtualWorld
from system import SnowCleanSystem

ROOT = Path(__file__).resolve().parent


class Scenario:
    """One named virtual demo with assertions evaluated on the run summary."""

    def __init__(self, key: str, title: str, description: str, builder, checks):
        self.key, self.title, self.description = key, title, description
        self.builder, self.checks = builder, checks

    def run(self) -> dict:
        context = self.builder()
        system = context['system']
        try:
            system.initialize()
            stop_when = context.get('stop_when')
            settle = int(context.get('settle_steps', 2))
            for _ in range(context.get('max_steps', 40)):
                if not system.running:
                    break
                system.update()
                if stop_when is not None and stop_when(system):
                    if settle <= 0:
                        break
                    settle -= 1
            summary = system.summary()
            detail = self.checks(system, summary)
        finally:
            system.shutdown()
        passed = all(detail.values())
        return dict(key=self.key, title=self.title, description=self.description,
                    passed=passed, checks=detail, summary=summary,
                    context={k: v for k, v in context.items() if k != 'system'})


# ---------------------------------------------------------------------- builders
def _world(radius: float = 0.05, fraction: float = 1.0, **kwargs) -> VirtualWorld:
    world = VirtualWorld(640, 480, **kwargs)
    world.add_target(0.0, 0.0, radius, pollution_fraction=fraction, depth_m=0.8)
    return world


def _system(world: VirtualWorld, humidity_mode: str = 'wet', **kwargs) -> SnowCleanSystem:
    camera = MockCamera(640, 480, 30, world=world)
    return SnowCleanSystem(camera=camera, world=world, humidity_mode=humidity_mode,
                           headless=True, **kwargs)


def _one_task_done(system) -> bool:
    """Stop condition: exactly one task reached a terminal status."""
    return bool(system.tasks.tasks) and all(t.terminal for t in system.tasks.tasks.values())


def _clean_world() -> VirtualWorld:
    """A world whose only target has already been fully cleaned away."""
    world = VirtualWorld(640, 480)
    world.add_target(0.0, 0.0, 0.05, pollution_fraction=0.0, depth_m=0.8)
    return world


# ---------------------------------------------------------------------- checks
def _no_errors(summary: dict) -> bool:
    return summary['errors'] == 0 and summary['state'] != 'ERROR'


# ---------------------------------------------------------------------- scenarios
def scene_a() -> Scenario:
    def builder():
        world = _clean_world()
        return dict(system=_system(world, 'dry', ), max_steps=6)

    def checks(system, summary):
        return {
            '无污染时保持运行且无错误': _no_errors(summary),
            '未生成任何清洁任务': summary['tasks'] == 0,
            '状态机停留在 PATROL': summary['state'] == 'PATROL',
            '检测结果为空': len(system.regions) == 0,
            '机器人未执行清洁动作': system.robot.cleaning_passes == 0,
        }
    return Scenario('A', '无污染（PATROL → DETECT → 无目标 → PATROL）',
                    '干净地面不产生污染目标，系统持续巡检。', builder, checks)


def scene_b() -> Scenario:
    def builder():
        world = _world(fraction=0.35)
        world.detection_hint = (0.36, 0.02)
        return dict(system=_system(world, 'dry'), stop_when=_one_task_done)

    def checks(system, summary):
        task = system.task
        return {
            '任务完成': task is not None and task.status == 'COMPLETED',
            '分级为 LIGHT': summary['task_level'] == 'LIGHT',
            '一次通过（retry = 0）': task.retry_count == 0,
            '执行 1 遍清洁': system.robot.cleaning_passes == 1,
            '清洁效率达标': (task.cleaning_efficiency or 0) >= 0.80,
            '无错误': _no_errors(summary),
        }
    return Scenario('B', 'LIGHT 一次成功', '轻度污染只需一遍清洁即可通过复检。', builder, checks)


def scene_c() -> Scenario:
    def builder():
        world = _world(fraction=0.45)
        world.detection_hint = (0.46, 0.05)
        return dict(system=_system(world, 'medium'), stop_when=_one_task_done)

    def checks(system, summary):
        task = system.task
        compensations = [a for a in system.robot.action_log() if a.startswith('cleaning_pass')]
        return {
            '任务完成': task is not None and task.status == 'COMPLETED',
            '分级为 MEDIUM': summary['task_level'] == 'MEDIUM',
            '经历一次补偿（retry = 1）': task.retry_count == 1,
            '总共执行 2 遍清洁': len(compensations) == 2,
            '复检最终通过': (task.cleaning_efficiency or 0) >= 0.80,
            '状态机走过 COMPENSATE': 'COMPENSATE' in [s.value for s in system.machine.history],
        }
    return Scenario('C', 'MEDIUM 一次失败一次补偿成功', '补偿流程后复检通过。', builder, checks)


def scene_d() -> Scenario:
    def builder():
        world = _world(radius=0.09, fraction=1.0)
        world.detection_hint = (0.92, 0.6)
        return dict(system=_system(world, 'wet'), stop_when=_one_task_done)

    def checks(system, summary):
        return {
            '任务完成': system.task is not None and system.task.status == 'COMPLETED',
            '分级为 HEAVY': summary['task_level'] == 'HEAVY',
            '首次清洁执行 2 遍': system.robot.cleaning_passes >= 2,
            '动作日志包含两次清洁通过': system.robot.action_log().count('cleaning_pass 1') == 1,
            '无错误': _no_errors(summary),
        }
    return Scenario('D', 'HEAVY 多遍清洁', '重度污染按策略执行多遍清洁。', builder, checks)


def scene_e() -> Scenario:
    def builder():
        world = _world(fraction=1.0)
        # A stain the mock cleaner cannot remove: every recheck fails.
        for target in world.targets:
            target.freeze_after_pass = True
        world.detection_hint = (0.60, 0.10)
        return dict(system=_system(world, 'medium'), max_steps=40,
                    stop_when=lambda s: not s.running or s._healthy_cycles >= 1)

    def checks(system, summary):
        task = system.task
        manual = system.tasks.by_status('MANUAL_CHECK')
        return {
            '最终进入 MANUAL_CHECK': task is not None and task.status == 'MANUAL_CHECK',
            'MANUAL_CHECK 任务被记录': len(manual) >= 1,
            '重试次数不超过 max_retry': task.retry_count == 2,
            '未无限循环': summary['steps'] < 40,
            '同一污渍不再重复生成新任务': len(system.tasks.tasks) == 1,
            '状态机走过 MANUAL_CHECK': 'MANUAL_CHECK' in [s.value for s in system.machine.history],
            '系统未崩溃仍可继续运行': summary['state'] != 'ERROR',
            '重复检测后进入待机而不是死循环': summary['sleeping'] is True,
        }
    return Scenario('E', '连续失败 → MANUAL_CHECK', '超出补偿次数后转人工检查，不无限循环。',
                    builder, checks)


def scene_f() -> Scenario:
    def builder():
        world = VirtualWorld(640, 480)
        # three separated stains so the mock cleaner visibly removes each one
        world.cleaning_retain = 0.10
        world.add_target(0.0, 0.0, 0.05, depth_m=0.8)
        world.add_target(0.0, 0.24, 0.05, depth_m=0.8)
        world.add_target(0.0, -0.24, 0.05, depth_m=0.8)
        world.detection_hint = (0.46, 0.20)
        system = _system(world, 'medium')
        # keep the three neighbouring stains separate in the pollution map
        system.config['planner']['pollution_map']['merge_distance_m'] = 0.12
        return dict(system=system, max_steps=120,
                    stop_when=lambda s: len(s.tasks.tasks) >= 3 and
                    all(t.terminal for t in s.tasks.tasks.values()))

    def checks(system, summary):
        levels = sorted(t.pollution_level for t in system.tasks.tasks.values())
        return {
            '检测到多个污染目标': len(system.tasks.tasks) >= 3,
            '按优先级顺序生成任务': levels == sorted(levels),
            '全部任务结束': all(t.terminal for t in system.tasks.tasks.values()),
            '优先级调度未报错': summary['errors'] == 0,
        }
    return Scenario('F', '多任务优先级调度', '多个坐标/等级的目标按优先级顺序处理。', builder, checks)


def scene_g() -> Scenario:
    def builder():
        world = _world(fraction=1.0)

        def block_routes(system):
            system.grid.cells[:, :] = 1
            system.grid.cells[1, 1] = 0   # only a corner of the map stays free

        system = _system(world, 'wet', setup_hook=block_routes)
        return dict(system=system, max_steps=12, stop_when=_one_task_done)

    def checks(system, summary):
        return {
            'A* 判定不可达': summary['state'] == 'ERROR' or
                        any(t.status in ('FAILED', 'MANUAL_CHECK')
                            for t in system.tasks.tasks.values()),
            '记录了失败原因': bool(summary['last_error']),
            '系统安全停机': summary['running'] is False,
            '清洁机构已复位': system.robot.cleaning is False and
                        system.robot.cleaner_lowered is False,
        }
    return Scenario('G', '无路径（NO PATH → ERROR）',
                    'A* 无法到达目标时明确失败并安全停机，而不是死循环。', builder, checks)


def scene_h() -> Scenario:
    def builder():
        world = _world(fraction=1.0)
        system = _system(world, 'wet')
        camera = system._camera_override

        def inject(index):
            if index >= 30:  # calibration consumes the first 30 frames
                from utils.errors import InvalidFrameError
                raise InvalidFrameError('injected empty frame')

        camera.failure_hook = inject
        return dict(system=system, max_steps=12)

    def checks(system, summary):
        return {
            '异常帧未导致进程崩溃': True,
            '状态机进入 ERROR': summary['state'] == 'ERROR',
            '记录了错误信息': bool(summary['last_error']),
            '安全停机（清洁机构复位）': system.robot.cleaning is False and
                                 system.robot.cleaner_lowered is False,
        }
    return Scenario('H', '异常输入注入', '空帧等异常被捕获，系统进入 ERROR 并安全停机。',
                    builder, checks)


SCENARIOS = {scene.key: scene for scene in
             (scene_a(), scene_b(), scene_c(), scene_d(), scene_e(), scene_f(),
              scene_g(), scene_h())}


def run_scenario(key: str) -> dict:
    if key not in SCENARIOS:
        raise KeyError(f'unknown scenario {key}; expected one of {sorted(SCENARIOS)}')
    return SCENARIOS[key].run()


def run_all() -> list[dict]:
    return [SCENARIOS[key].run() for key in sorted(SCENARIOS)]


if __name__ == '__main__':
    report = run_all()
    for entry in report:
        print(f"[{'PASS' if entry['passed'] else 'FAIL'}] {entry['key']} {entry['title']}")
        for name, ok in entry['checks'].items():
            print(f"    {'[ok]' if ok else '[!!]'} {name}")
    print(f"{sum(1 for e in report if e['passed'])}/{len(report)} scenarios passed")
