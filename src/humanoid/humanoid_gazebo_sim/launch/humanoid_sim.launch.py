#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_gazebo_sim')
    world_file = os.path.join(pkg_share, 'worlds', 'humanoid_world.world')
    urdf_file = os.path.join(pkg_share, 'urdf', 'simple_humanoid.urdf')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    gzserver = ExecuteProcess(
        cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen')

    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen')

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'simple_humanoid', '-topic', 'robot_description'],
        output='screen')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}])

    teleop = Node(
        package='humanoid_gazebo_sim', executable='humanoid_teleop',
        name='humanoid_teleop', output='screen')
    perception = Node(
        package='humanoid_gazebo_sim', executable='humanoid_perception',
        name='humanoid_perception', output='screen')
    controller = Node(
        package='humanoid_gazebo_sim', executable='humanoid_controller',
        name='humanoid_controller', output='screen')
    bridge = Node(
        package='humanoid_gazebo_sim', executable='humanoid_bridge',
        name='humanoid_bridge', output='screen')

    return LaunchDescription([
        gzserver, gzclient, spawn_entity, robot_state_publisher,
        teleop, perception, controller, bridge,
    ])
