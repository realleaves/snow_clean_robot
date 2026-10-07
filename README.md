# 室内雪迹清除机器人多模态感知与智能决策系统 V1.0

本工程按《室内雪迹清除机器人多模态感知与智能决策系统 V1.0 全虚拟功能实现与验收测试方案》实现并验收上位机软件。

当前版本为**全虚拟闭环**：不接真实 D435i、单片机、定位系统与机器人本体，全部功能以 Mock 相机 / 虚拟 RGB-D 场景 / 测试图片集 / Mock 湿度 / Mock 位姿 / Mock 机器人驱动，已完成

```text
污染检测 → 空间确认 → 湿度确认 → 多模态评分 → 污染分类 → 任务生成
→ 优先级调度 → A* 规划 → 清洁策略 → 模拟执行 → 复检 → 补偿清洁
→ 完成 / 人工检查 → 状态机 / 日志 / 可视化
```

全部软件闭环，并通过 478 项自动化测试与 8 个完整 Mock 演示场景。

`real` 模式会明确报错，因为真实定位、相机外参、湿度通信和机器人控制协议尚未提供。

---

## 1. 安装与运行

建议 Python 3.10+：

```bash
cd /home/real/Desktop/大创/snow_clean_robot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 默认 Mock 闭环（窗口由 config/system.yaml 的 headless 决定）
.venv/bin/python main.py --mode mock --headless

# 运行单个 / 全部完整演示场景 A–H，并逐项打印验收结果
.venv/bin/python main.py --scenario C
.venv/bin/python main.py --all-scenarios

# 全量自动化测试（478 项）
.venv/bin/python -m pytest -q
```

`--steps` 控制最大状态机步数；`--headless` 禁用 OpenCV 窗口；`--json` 输出运行摘要。
运行日志写入 `logs/`（文件名带微秒后缀，多次运行不覆盖），干净参考帧写入
`data/clean_reference/reference.png`。

`pytest.ini` 已显式禁用 ROS 2 的 `launch_testing` / `launch_ros` 两个 pytest 插件
（其依赖 `lark` 且声明了与本机 pytest 不兼容的钩子）。若在其它环境仍需绕过：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q
```

未安装 pytest 时可退回 `unittest`（仅覆盖以 `unittest.TestCase` 编写的部分）：

```bash
.venv/bin/python -m unittest discover -s tests -q
```

## 2. 验收结论

| 验收域 | 内容 | 状态 |
| --- | --- | --- |
| Camera | Mock RGB-D、FramePacket、尺寸一致、多来源（图片/序列/视频/生成器）、FPS、资源释放、异常帧 | 通过 |
| Perception | Ground ROI、Calibration、Preprocess、WetRegionDetector、PollutionRegion、小区域过滤、Depth 有效性 | 通过 |
| Fusion | 湿度正反标定、VisualScore、AreaScore、PollutionScore、权重校验、LIGHT/MEDIUM/HEAVY | 通过 |
| Mapping | PollutionMap 新增/更新/合并/删除/过期、MockPoseProvider、CoordinateTransform、GridMap | 通过 |
| Planning | A* 正常/绕障/无路径/边界/窄道/斜向、PathPlanner | 通过 |
| Decision | CleaningTask、TaskManager、PriorityScheduler、防任务饿死、CleaningPolicy | 通过 |
| Interface | RobotInterface、MockRobot（动作日志与联锁）、MockHumidity | 通过 |
| Feedback | CleaningEvaluator、PASS/FAIL、Compensation、max_retry、MANUAL_CHECK | 通过 |
| System | StateMachine 正常/补偿/失败/ERROR、Logging、Headless、GUI 面板、完整 Mock Demo | 通过 |
| Automation | 478 项 pytest 全绿、无未处理异常、无无限循环 | 通过 |

详细的逐条核对见 [DESIGN_REVIEW.md](DESIGN_REVIEW.md)。

不得据此宣称“真实机器人性能全部验证通过”：真实硬件性能验证不属于本阶段范围。

## 3. 工程结构

```text
camera/          D435i 适配器、MockCamera（多来源）、VirtualWorld（虚拟场景）
perception/      ROI、标定、预处理、特征图、候选区域、污染区域数据类
fusion/          湿度归一化、多模态评分、三级分类
mapping/         人工栅格地图、污染地图去重、坐标变换链、Mock 位姿
planning/        八邻域 A*、世界坐标路径规划
decision/        任务状态、优先级调度、任务管理、清洁策略、系统状态机
feedback/        清洁效率评估、有限次数补偿
interface/       硬件抽象、Mock 机器人、Mock 湿度传感器
visualization/   OpenCV 监控面板、相机视图、地图视图
tests/           27 个测试文件，478 项用例（正常 / 边界 / 异常）
tools/           虚拟测试图像数据集生成器
scenario_runner.py  场景 A–H 定义与断言
config/*.yaml    全部可调参数（ROI、深度、阈值、权重、级别、清洁、调度、地图）
data/test_images/{clean,light,medium,heavy,interference}/  虚拟测试图像集
```

所有业务参数均在 `config/*.yaml`；业务代码不直接操作 GPIO/PWM，只调用
`interface/robot_interface.py`。

## 4. 虚拟测试数据

```bash
.venv/bin/python -m tools.generate_test_data --per-class 8
```

按 `clean / light / medium / heavy / interference` 生成确定性（带随机种子）640×480
RGB 图与 z16 毫米深度侧车文件；`interference` 覆盖强反光、阴影、灯光变化、桌腿、
鞋子、地面纹理、亮斑与颜色变化。生成结果已提交在 `data/test_images/`，可直接被
MockCamera 作为图片序列回放。

## 5. 完整 Mock 演示场景

| 场景 | 内容 | 关键断言 |
| --- | --- | --- |
| A | 无污染 | 不生成任务，持续巡检 |
| B | LIGHT 一次成功 | retry=0，清洁 1 遍，效率 ≥ 0.80 |
| C | MEDIUM 一次失败一次补偿成功 | retry=1，共 2 遍清洁，走过 COMPENSATE |
| D | HEAVY 多遍清洁 | 首次执行 2 遍 |
| E | 连续失败 | retry 不超过 max_retry，进入 MANUAL_CHECK 且不死循环 |
| F | 多任务优先级调度 | 多目标顺序执行且无错误 |
| G | 无路径 | A* 失败 → ERROR 并安全停机 |
| H | 异常输入注入 | 空帧被捕获 → ERROR，清洁机构复位 |

```bash
.venv/bin/python main.py --all-scenarios
# 8/8 场景通过
```

## 6. 真机接入前必须完成

1. 标定相机内参和相机到机器人坐标系的外参，并确定清洁机构相对于机器人中心的偏移。
   `mapping/coordinate_transform.py` 已提供可单测的四段坐标链，但状态机仍使用已知的
   Mock 场景坐标。
2. 提供能连续更新且带不确定度/失效指示的真实 `PoseProvider`。仅有人工地图和 A* 不能
   构成自主导航，还需机器人尺寸、障碍膨胀和移动障碍避让。
3. 定义脚下湿度探头实际测量位置、采样时机及干湿 ADC 标定；机器人必须移动到目标处
   再将该读数与该目标融合。
4. 实现底层通信协议、动作完成确认、急停和故障反馈后，再接通 `RobotInterface`。禁止仅靠
   发出前进命令推定机器人已到达。
5. 建立可重复观测同一块地面的复检方式。机器人移动后固定像素参考图不再对齐；需要地图
   锚定/图像配准，或像当前 Mock 一样在固定检测站复检。
6. 用真实地面材料、融雪/水渍/泥污及阴影反光干扰集评估 Precision、Recall、F1，并重新
   标定所有阈值。深度仅验证几何距离，不能识别湿度。

## 7. 已知限制

Mock 相机是固定视角合成图：`DETECT` / `RECHECK` 一律在固定检测站 `(0, 0, 0)` 取帧，
这是固定参考图差分成立的唯一视角（`system.DETECTION_STATION`）。场景 A–H 通过
`VirtualWorld` 的 `detection_hint` / `cleaning_effectiveness` / `freeze_after_pass`
控制分级与清洁效果，用于确定性地驱动决策链路；感知算法本身不读取这些注入值，其真实
（未注入）表现由 `tests/test_perception.py`、`tests/test_test_data.py` 单独验证。

参考图差分对强阴影、镜面反光、曝光变化和视角变化会误报（`interference` 数据集中即可
观察到误报），需真实数据重新调参。无真实地图障碍膨胀、现场避障和安全停机验证。D435i
模块已写，但本机无 `pyrealsense2` 和硬件，尚未做采集实测或 30 分钟稳定性测试。
