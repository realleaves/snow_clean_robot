# V1.0 方案核查、实现边界与虚拟验收核对表

本文件对应《全虚拟功能实现与验收测试方案》第 32 节，逐条记录当前代码的验收状态。
复核命令：

```bash
cd /home/real/Desktop/大创/snow_clean_robot
.venv/bin/python -m pytest -q            # 478 passed
.venv/bin/python main.py --all-scenarios # 8/8 场景通过
```

## 1. 项目理解

这是树莓派上位机的感知与决策软件，不是底盘驱动项目。视觉负责提出疑似湿污候选；机器人到
目标附近后，脚下湿度传感器参与分级；规划器使用人工地图和位置源生成路线；清洁机构按级别
执行固定次数；复检失败时有限次补偿。当前 V1.0 已在不依赖实物的条件下跑通并自动验证完整
软件闭环，真机闭环仍依赖标定、定位和通信协议。

## 2. 虚拟验收核对表

### Camera

```text
[x] Mock RGB 正常                 tests/test_camera.py
[x] Mock Depth 正常               tests/test_camera.py::test_depth_modes
[x] FramePacket 正常              tests/test_camera.py::test_frame_packet_contract
[x] RGB / Depth 尺寸一致          tests/test_camera.py（含不一致即报错）
[x] 异常帧处理正常                tests/test_camera.py（空源/坏图/尺寸不符/生成器非法返回值）
[x] 图片 / 序列 / 视频 / 生成器   tests/test_camera.py
[x] FPS 统计、stop 释放           tests/test_camera.py
```

### Perception

```text
[x] Ground ROI                    tests/test_ground_roi.py（越界、空 ROI、配置错误）
[x] Calibration                   tests/test_calibration.py（中值抑制异常、帧数不足、保存/读取）
[x] Preprocess                    tests/test_perception.py、tests/test_test_data.py
[x] WetRegionDetector             tests/test_perception.py
[x] PollutionRegion               tests/test_pollution_region.py（bbox/中心/面积/评分域校验）
[x] 小区域过滤                    tests/test_test_data.py（light 面积下限）
[x] Depth 有效性                  tests/test_depth.py（0.05/0.15/0.80/2.00/3.00 表）
```

### Fusion

```text
[x] Humidity 正向标定             tests/test_humidity.py
[x] Humidity 反向标定             tests/test_humidity.py
[x] VisualScore / AreaScore       tests/test_perception.py、tests/test_depth.py
[x] PollutionScore                tests/test_fusion.py（V=0.6,H=0.8,A=0.4 → 0.62）
[x] 权重校验                      tests/test_fusion.py（和不为 1、负权重均报错）
[x] LIGHT / MEDIUM / HEAVY        tests/test_classifier.py（0.349/0.350/0.699/0.700 边界）
```

### Mapping

```text
[x] PollutionMap 新增/更新/合并/删除      tests/test_pollution_map.py
[x] 合并距离边界（1.00,1.00 与 1.10,1.08）tests/test_pollution_map.py
[x] MockPoseProvider              tests/test_pose_provider.py（yaw 0/π/2/π/-π/2）
[x] CoordinateTransform           tests/test_coordinate_transform.py（Pixel→Camera→Robot→World→Grid）
[x] GridMap                       tests/test_grid_map.py（世界/栅格互转、障碍、边界、加载）
```

### Planning

```text
[x] A* 正常路径                   tests/test_astar.py、tests/test_path_planner.py
[x] A* 绕障                       tests/test_astar.py::test_obstacle_and_no_corner_cutting
[x] A* 无路径                     tests/test_astar.py、tests/test_path_planner.py
[x] A* 边界 / 起点终点非法 / 障碍 tests/test_astar.py（12 类场景全覆盖）
[x] A* 斜向与窄通道               tests/test_astar.py
[x] PathPlanner                   tests/test_path_planner.py
```

### Decision

```text
[x] CleaningTask                  tests/test_cleaning_task.py（全部合法状态迁移）
[x] TaskManager                   tests/test_task_manager.py（空/单/多/完成/失败/重试/取消/重复目标）
[x] PriorityScheduler             tests/test_priority.py（P = ws·S - wd·D + wt·T + 有界老化项）
[x] 防任务饿死                    tests/test_priority.py（老化后 LIGHT 超越 HEAVY）
[x] CleaningPolicy                tests/test_cleaning_policy.py（LIGHT 1 / MEDIUM 1 / HEAVY 2）
```

### Interface

```text
[x] RobotInterface                tests/test_mock_robot.py（抽象方法齐全、无速度/压力接口）
[x] MockRobot                     tests/test_mock_robot.py（动作顺序、联锁、急停、日志）
[x] MockHumidity                  tests/test_humidity.py
```

### Feedback

```text
[x] CleaningEvaluator             tests/test_cleaning_evaluator.py（η 表、0.799 FAIL / 0.800 PASS）
[x] PASS / FAIL                   tests/test_full_mock_demo.py
[x] Compensation / max_retry      tests/test_compensation.py（1→2→MANUAL_CHECK）
[x] MANUAL_CHECK                  tests/test_state_machine.py、tests/test_full_mock_demo.py
```

### System

```text
[x] StateMachine 正常路径         tests/test_state_machine.py
[x] StateMachine 补偿路径         tests/test_state_machine.py（经 COMPENSATE 重新规划）
[x] StateMachine 失败路径         tests/test_state_machine.py
[x] StateMachine ERROR            tests/test_state_machine.py、tests/test_system.py
[x] Logging                       tests/test_logging.py（时间/级别/模块/状态/评分/动作/异常栈/不覆盖）
[x] Headless                      tests/test_visualization.py（子进程无 DISPLAY 运行）
[x] GUI                           tests/test_visualization.py（面板渲染与降级）
[x] 完整 Mock Demo                tests/test_full_mock_demo.py（场景 A–H）
```

### Automation

```text
[x] pytest 全部通过               478 passed
[x] 无测试失败
[x] 无未处理异常                  严重异常统一进入 ERROR 状态并安全停机
[x] 无无限循环                    所有状态迁移有界；黑名单 + 待机机制防止重复任务
```

## 3. 本阶段完成的工作

1. `camera/virtual_scene.py`：把合成场景抽象为可配置的 `VirtualWorld`（目标位置、半径、
   污染度、外观、清洁保留率、冻结/耐久设置、检测提示），使 Mock 闭环的几何与
   `mapping/coordinate_transform.py` 使用同一套外参，检测坐标与任务坐标自洽。
2. `camera/mock_camera.py`：MockCamera 支持虚拟场景、单图、图片序列、视频、生成器与
   故障注入，统一校验帧尺寸、统计 FPS、可释放资源；z16 毫米深度自动转米。
3. `perception/calibration.py`：独立标定模块（中值参考图、帧数校验、保存/读取）。
4. `perception/ground_roi.py`：ROI 越界/空/配置错误显式报错，新增 `valid_depth_ratio`、
   `depth_status`；`PollutionRegion` 增加 `validate()` 不变量校验。
5. `fusion/`：湿度端点校验、`HumidityCalibration`、权重与输入域校验、分类器边界修正
   （原实现不接受 `medium_max == 1`）。
6. `mapping/`：`PollutionMap` 变为完整 CRUD（含 `observations`、`expire`、合并计数、
   包含边界的合并半径），`GridMap` 增加膨胀/复制/夹取/加载校验，坐标变换补齐
   `world_to_robot` 与 `pixel_to_world` 全链。
7. `planning/`：A* 增补越界保护、扩展计数、`plan_or_raise`、代价计算；`PathPlanner` 在
   起点被占时自动夹取最近可通行格并记录失败原因。
8. `decision/`：`CleaningTask` 带合法状态迁移表；`TaskManager` 完整生命周期与
   `is_known_bad_target`；`PriorityScheduler` 保留原公式并给出有界老化项、
   `rank/next_task/is_starving`；状态机新增 `MANUAL_CHECK` 与 `fail()`。
9. `feedback/`：`Evaluation` 带原因，边界按 0.799 FAIL / 0.800 PASS 校验。
10. `interface/`：`MockRobot` 记录动作日志、清洁联锁、急停与故障复位；湿度 Mock 可注入。
11. `system.py`：统一异常分层（可恢复 → 任务 FAILED / 严重 → ERROR 并安全停机）、
    固定检测站复检、按目标匹配的复检测量、黑名单与待机保护、运行摘要。
12. `tools/generate_test_data.py`：生成 clean/light/medium/heavy/interference 虚拟图像集。
13. `scenario_runner.py` + `main.py --all-scenarios`：场景 A–H 可执行断言报告。
14. `tests/`：27 个测试文件、478 项用例，覆盖正常 / 边界 / 异常三类路径。

## 4. 必须补足的方案条件（真机接入前）

| 优先级 | 问题 | 影响与处理 |
| --- | --- | --- |
| 高 | 固定干净参考图与移动相机不配准 | 移动后同一像素对应不同地面，差分与面积复检失效。真机应固定复检视角，或加入地面图像配准/地图锚定。当前 Mock 通过 `DETECTION_STATION` 固定视角。 |
| 高 | 没有真实定位 | 人工地图与 A* 只能算路线，不能证明车辆到位。真实 `PoseProvider` 应提供位置、朝向、有效期与失效状态；到位由底层反馈确认。 |
| 高 | 相机像素、地面湿度探头和清洁机构坐标未标定 | 需标定相机外参与三个器件在机器人上的位置，才能把候选、湿度读数与清洁覆盖对应到同一地面区域。Mock 使用已知合成坐标。 |
| 高 | 清洁动作语义不完整 | `move_forward`、清洁一遍、升降机构需要动作完成回执和超时/急停语义。未知协议不接通真机。 |
| 中 | A* 路线未考虑车体宽度与动态障碍 | 真机需要障碍膨胀（`GridMap.inflate_obstacles` 已就绪）、局部避障与碰撞停机。 |
| 中 | 视觉规则与湿度权重没有实验依据 | 强反光、阴影、曝光变化可能误报（`interference` 数据集可复现）；按实际地面与污渍采集数据，测 Precision/Recall/F1 后调参。 |
| 中 | `eta` 只用面积 | 污渍变浅但面积不变时会判失败；真实复检应同时评估强度与区域匹配。 |

## 5. 完成判定

上述核对表全部通过，当前版本可冻结为：

```text
室内雪迹清除机器人多模态感知与智能决策系统 V1.0
```

可以准确描述为：系统 V1.0 软件功能模块全部完成，已在虚拟测试环境下完成 RGB-D 数据输入、
湿态污染识别、多模态融合、污染等级判定、污染地图、任务生成、优先级调度、A* 路径规划、
清洁策略、模拟机器人执行、清洁复检、补偿清洁、状态机和运行监控等完整软件闭环，全部
V1.0 虚拟功能测试及自动化测试（478 项）通过。

不得描述为：真实机器人性能全部验证通过。因为真实硬件性能验证不属于本阶段范围。
