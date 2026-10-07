# 室内雪迹清除机器人上位机 V1.0

本工程按 [原始规格说明](../Downloads/室内雪迹清除机器人_V1.0_Agent开发规格说明%20(4).md) 实现上位机软件原型。当前可运行的是固定视角的 Mock 闭环：参考帧校准 → 疑似湿污检测 → 接近目标 → 脚下湿度确认 → 评分分级 → 任务与 A* → 清洁 → 复检 → 补偿。`real` 模式会明确报错，因为定位、相机外参、湿度通信和机器人控制协议尚未提供。

## 安装与运行

建议 Python 3.10+：

```bash
cd /home/real/snow_clean_robot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --mode mock --headless
.venv/bin/python -m pytest -q
```

本机已有 OpenCV、NumPy 和 PyYAML 时，也可直接运行 `python3 main.py --mode mock --headless`。`--steps` 控制最大状态机步数；`--headless` 禁用窗口。运行日志写入 `logs/`，参考帧写入 `data/clean_reference/reference.png`。
尚未安装 pytest 时，可用 `python3 -m unittest discover -s tests -v` 运行同一组测试。

有 D435i 和可用的 `pyrealsense2` 时，可只检查相机：`python3 camera_check.py --seconds 600 --headless --save-dir data/test_images/camera_check`。该诊断只采集图像，不启动机器人。方案问题与验收状态见 [DESIGN_REVIEW.md](DESIGN_REVIEW.md)。

## 结构

- `camera/`：D435i 对齐 RGB-D 适配器及合成相机。深度统一为米。
- `perception/`：固定 ROI、参考图差分、特征融合、连通域。
- `fusion/`：湿度正反极性标定、融合得分和三级分类。
- `mapping/` 与 `planning/`：人工地图、污染目标去重、八邻域 A*。
- `decision/`：任务优先级、清洁次数、状态机。
- `feedback/`：清洁效率与有限次数补偿。
- `interface/`：硬件抽象、Mock 湿度与机器人。
- `visualization/`：OpenCV 监控及无界面模式。

配置均在 `config/*.yaml`。默认阈值、权重、ROI、湿度端点均为待实验验证的起始值。

## 真机接入前必须完成

1. 标定相机内参和相机到机器人坐标系的外参，并确定清洁机构相对于机器人中心的偏移。`mapping/coordinate_transform.py` 提供投影函数，但当前状态机只使用已知的 Mock 场景坐标。
2. 提供能连续更新且带不确定度/失效指示的真实 `PoseProvider`。单有人工地图和 A* 不能构成自主导航。还需机器人尺寸、地图障碍膨胀和移动障碍避让。
3. 定义脚下湿度探头实际测量位置、采样时机及干湿 ADC 标定；机器人必须移动到目标处再将读数与目标融合。
4. 实现底层通信协议、动作完成确认、急停和故障反馈后，再接通 `RobotInterface`。禁止仅靠发出前进命令推定机器人已到达。
5. 建立可重复观测同一块地面的复检方式。机器人移动后，固定像素参考图不再对齐；需要地图锚定/图像配准或在固定检测站进行复检。
6. 用真实地面材料、融雪/水渍/泥污及阴影反光干扰集评估 Precision、Recall、F1，并重新调所有阈值。深度仅验证几何距离，不能识别湿度。

## 已知限制

Mock 相机是固定视角合成图。`APPROACH` 的目标相对位置和清洁效果也由合成场景给定，用于软件联调，不表示真实机器人定位或清洁能力。当前感知采用参考图差分，强阴影、镜面反光、曝光变化及视角变化会误报。无真实地图障碍膨胀、现场避障和安全停机验证。D435i 模块已写，但本机无 `pyrealsense2` 和硬件，尚未做采集实测或 30 分钟稳定性测试。
