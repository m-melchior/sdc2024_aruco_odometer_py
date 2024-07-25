from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

# requires depthai-ros
# to install:
# apt install ros-humble-depthai-ros

def generate_launch_description():
    depthai_ros_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('depthai_ros_driver'),
            '/launch',
            '/camera.launch.py'
        ])
    )

    aruco_detector_node = Node(
        package = 'sdc2024_aruco_odometer_py',
        executable = 'sdc2024_aruco_odometer_py',
        name = 'sdc2024_aruco_odometer'
    )

    return LaunchDescription([
        depthai_ros_driver_launch,
        aruco_detector_node
    ])
