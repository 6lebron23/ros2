"""arm_traj_track 启动入口。

MuJoCo 3 关节机械臂连续轨迹跟踪：感知 → MLP 神经网络 → 闭环扭矩控制。
本 launch 在包源码目录启动 MLP 闭环部署评估（deploy_eval），
对连续轨迹（圆 / 8 字）统计闭环跟踪误差并生成误差曲线与性能 CSV。

用法（在包目录）：
    cd src/arm_traj_track
    ros2 launch arm_traj_track arm_traj_track.launch.py
其他脚本（采集 / 训练 / DAgger / 渲染动图）按文档手动执行。
"""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    # launch 位于 <pkg>/launch，包根在 launch 的上一级，脚本在 <pkg>/arm_traj_track/
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_dir = os.path.join(pkg_dir, 'arm_traj_track')
    deploy = os.path.join(script_dir, 'deploy_eval.py')

    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', deploy],
            cwd=script_dir,
            output='screen',
            name='arm_traj_track_deploy',
        ),
    ])
