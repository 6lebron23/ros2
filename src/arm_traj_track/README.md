# arm_traj_track —— MuJoCo 3 关节机械臂连续轨迹跟踪

对 **3 关节平面机械臂** 做物理仿真，串起一条 **感知 → 神经网络 → 闭环扭矩控制** 的连续轨迹跟踪闭环：

- 感知：3 关节编码器（三角编码）+ 轨迹前瞻点相对末端坐标差 + 末端朝向误差，共 **9 维**；
- 网络：纯 NumPy 实现的 MLP `[9, 128, 128, 3]`，直接把感知状态映射为 **3 关节扭矩**；
- 学习：先用 **IK + PD 专家** 采集 20 万步监督演示，训练 MLP，再用 **DAgger** 缓解分布偏移；
- 轨迹：圆（circle）/ 8 字（eight）两种连续闭合轨迹；
- 引擎：MuJoCo（`integrator=implicitfast`），**无需图形界面与显卡**，可无头复现。

## 环境与依赖

- Ubuntu 20.04 / ROS 2 Humble（或 Foxy），Python 3.8+
- `pip install mujoco numpy matplotlib`（mujoco 2.3.x 验证通过）

## 目录结构

```
src/arm_traj_track/
├── package.xml / setup.py / setup.cfg
├── launch/arm_traj_track.launch.py   # launch 入口（部署评估）
├── models/arm3.xml                   # 3 关节机械臂 MuJoCo 模型
└── arm_traj_track/
    ├── arm3_env.py        # 环境：正/逆运动学、IK+PD 专家、9 维感知、轨迹
    ├── nn.py              # 纯 NumPy MLP + Adam
    ├── collect_demo.py    # IK+PD 专家采集监督演示（demo.npz）
    ├── train_mlp.py       # 监督学习训练 MLP（mlp.npz + 归一化统计）
    ├── dagger_iter.py     # DAgger 迭代（部署分布采集 + 专家标注 + 重训）
    ├── deploy_eval.py     # MLP 闭环部署评估 + 误差曲线 + 性能 CSV
    ├── render_gif.py      # matplotlib(Agg) 渲染闭环跟踪动图
    └── check_expert.py    # 专家控制器性能验证
```

## 运行步骤（在包目录 `src/arm_traj_track/arm_traj_track/`）

```bash
# 1) 专家采集监督演示（20 万步，圆/8字交替 400 回合）
python3 collect_demo.py --episodes 400

# 2) 监督学习训练 MLP
python3 train_mlp.py          # 10 万样本子集，300 轮，输出 mlp.npz

# 3) DAgger 迭代（可选，缓解分布偏移）
python3 dagger_iter.py

# 4) MLP 闭环部署评估（性能表 eval_summary.csv + 误差曲线）
python3 deploy_eval.py

# 5) 渲染闭环跟踪动图（圆 / 8 字）
python3 render_gif.py --kind circle
python3 render_gif.py --kind eight

# launch 启动（包目录）
cd ..
ros2 launch arm_traj_track arm_traj_track.launch.py
```

## 性能（无头复现，30 回合随机初始）

| 方法 | 平均跟踪误差 (m) | 中位数 (m) | 成功率 (<0.05 m) |
|---|---|---|---|
| MLP 闭环 | 0.0086 | 0.0088 | **100%** |
| IK+PD 专家 | 0.0083 | 0.0074 | **100%** |

MLP 在闭环部署下追平 IK+PD 专家，成功率均达 100%。

> 说明：本模块在仿真环境（MuJoCo）中完成，感知、规划与控制算法为神经网络（MLP）。动图与曲线见 `docs/ground/arm_traj_track.md`。
