from setuptools import setup
from glob import glob
import os

package_name = 'arm_traj_track'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='6lebron23',
    maintainer_email='6lebron23@users.noreply.github.com',
    description='MuJoCo 3 关节机械臂连续轨迹跟踪：感知 → MLP → 闭环扭矩控制',
    license='MIT',
    entry_points={
        'console_scripts': [
            'collect_demo = arm_traj_track.collect_demo:main',
            'train_mlp = arm_traj_track.train_mlp:main',
            'dagger_iter = arm_traj_track.dagger_iter:main',
            'deploy_eval = arm_traj_track.deploy_eval:main',
            'render_gif = arm_traj_track.render_gif:main',
        ],
    },
)
