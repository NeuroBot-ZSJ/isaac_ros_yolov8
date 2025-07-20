# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # ---------- 新增：开关 rqt_image_view ----------
    declare_use_rqt = DeclareLaunchArgument(
        'use_rqt',
        default_value='false',          # 默认不开
        description='Launch rqt_image_view if true'
    )
    use_rqt = LaunchConfiguration('use_rqt')
    # ------------------------------------------------

    declare_num_classes = DeclareLaunchArgument(
        'num_classes',
        default_value='6',
        description='Number of classes for YOLOv8 decoder'
    )
    num_classes = LaunchConfiguration('num_classes')

    my_package_dir = get_package_share_directory('isaac_ros_yolov8')

    return LaunchDescription([
        declare_use_rqt,                # 把新参数加入 LD
        declare_num_classes,

        # Include main YOLOv8 + TensorRT pipeline
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(my_package_dir, 'launch',
                             'yolov8_tensor_rt.launch.py')),
            launch_arguments={'num_classes': num_classes}.items()
        ),

        # Visualizer node
        Node(
            package='isaac_ros_yolov8',
            executable='isaac_ros_yolov8_visualizer.py',
            name='yolov8_visualizer',
            output='screen'
        ),

        # Image viewer —— 只有 use_rqt:=true 时才会启动
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='image_view',
            output='screen',
            arguments=['/yolov8_processed_image'], 
            condition=IfCondition(use_rqt)
        )
    ])
