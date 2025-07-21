# isaac_ros_yolov8
isaac_ros_yolov8项目：在docker中使用自定义yolov8模型（经过TensorRT加速推理）

### 使用教程

本人的部署环境是Jetson orin nano 8GB、ROS2 humble、D435i

##### 1.部署相机环境

本人的相机环境部署在宿主机中

###### 1.1从 ROS 服务器安装Intel RealSense SDK 2.0（librealsense2）

**作用**：提供D435i的底层驱动、API和工具（如深度计算、IMU同步等）。

**SDK安装方法参考**https://github.com/jetsonhacks/jetson-orin-librealsense

下面命令行直接安装**亲测会出问题**！！！

```
sudo apt install ros-humble-librealsense2*
```

###### 1.2从 ROS 服务器安装realsense相机ROS驱动包 (realsense2_camera)

**作用**：将SDK功能封装为ROS节点，发布图像、深度、IMU等话题。

```
sudo apt install ros-humble-realsense2-camera
```

###### 1.3启动相机ROS节点

```
ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:=640,480,30 \
  depth_module.depth_profile:=640,480,30 
```

##### 2.docker官方安装教程

https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

如果有用代理还需为 Docker 服务配置代理，此处略

##### 3.运行容器isaac_yolov8_ros并挂载代码

```
docker run -it \
  --network host \
  --runtime nvidia \
  --ipc host \
  --privileged \
  -v /home/nvidia/isaac_ros_ws:/workspaces/isaac_ros-dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev:/dev \
  -e DISPLAY=$DISPLAY \
  --name isaac_yolov8_ros \
  isaac_ros_dev \
  /bin/bash
```

将宿主机目录 `/home/nvidia/isaac_ros_ws` 挂载到容器内路径 `/workspaces/isaac_ros-dev`，方便容器内访问和修改宿主机上的代码或数据。（此处要将目录替换成自身放置isaac_ros_ws工作空间的目录）

##### 4.然后，在宿主机执行： 

```
xhost +local:root
```

作用就是允许宿主机本地 root 用户访问 X11 显示服务器，方便容器内以 root 运行的 GUI 程序显示图形界面。

##### 5.容器内环境搭建：

```
sudo apt-get update
sudo apt-get install -y ros-humble-isaac-ros-dnn-image-encoder ros-humble-isaac-ros-tensor-rt
```

安装项目其他依赖包，得在容器内编译ROS工作空间下的源码

##### 6.容器和宿主机都设置：

```
sudo apt update
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

```
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
source ~/.bashrc
```

ROS 2 在跨容器和宿主机通信时，默认的中间件实现不匹配或者缺少依赖，会导致话题能看到但数据无法传输。

##### 7.运行指令：

启动相机ROS节点后，再在docker中启动以下命令：

```
ros2 launch isaac_ros_yolov8 isaac_ros_yolov8_visualize.launch.py \
  model_file_path:=/workspaces/isaac_ros-dev/src/isaac_ros_assets/models/yolov8/best.onnx \
  engine_file_path:=/workspaces/isaac_ros-dev/src/isaac_ros_assets/models/yolov8/best.engine \
  input_binding_names:='[images]' \
  output_binding_names:='[output0]' \
  input_image_width:=640 \
  input_image_height:=480 \
  network_image_width:=640 \
  network_image_height:=640 \
  force_engine_update:=false \
  image_mean:='[0.0, 0.0, 0.0]' \
  image_stddev:='[1.0, 1.0, 1.0]' \
  confidence_threshold:=0.5 \
  nms_threshold:=0.4 \
  num_classes:=6
```

这里面的参数适配官方基本都给出了办法。

##### 8.关机重启，容器不用重新配置

```
docker start -ai isaac_yolov8_ros
```

由于是持久创建的容器，故而再次加载用docker start

### 问题记录

这里主要是记录对于官方源代码的修改。

##### 1.num_classes:=6传参传不过去

参考：[Cuda bindings mismatch error using custom trained model · Issue #32 · NVIDIA-ISAAC-ROS/isaac_ros_object_detection](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection/issues/32)

isaac_ros_yolov8_visualize.launch.py文件修改

```
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# …（上面头部许可证不变）…

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch argument for num_classes with default value 6
    declare_num_classes = DeclareLaunchArgument(
        'num_classes',
        default_value='6',
        description='Number of classes for YOLOv8 decoder'
    )
    num_classes = LaunchConfiguration('num_classes')

    my_package_dir = get_package_share_directory('isaac_ros_yolov8')

    return LaunchDescription([
        declare_num_classes,

        # Include main YOLOv8 + TensorRT pipeline
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(my_package_dir, 'launch', 'yolov8_tensor_rt.launch.py')
            ]),
            launch_arguments={
                'num_classes': num_classes
            }.items()
        ),

        # Visualizer node
        Node(
            package='isaac_ros_yolov8',
            executable='isaac_ros_yolov8_visualizer.py',
            name='yolov8_visualizer',
            output='screen'
        ),

        # Image viewer
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='image_view',
            arguments=['/yolov8_processed_image'],
            output='screen'
        )
    ])
```

yolov8_tensor_rt.launch.py文件修改

```
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# …（上面头部许可证不变）…

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for TensorRT ROS 2 node."""
    # By default loads and runs mobilenetv2-1.0 included in isaac_ros_dnn_inference/models
    launch_args = [
        DeclareLaunchArgument(
            'model_file_path',
            default_value='',
            description='The absolute file path to the ONNX file'),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute file path to the TensorRT engine file'),
        DeclareLaunchArgument(
            'input_tensor_names',
            default_value='["input_tensor"]',
            description='A list of tensor names to bound to the specified input binding names'),
        DeclareLaunchArgument(
            'input_binding_names',
            default_value='[""]',
            description='A list of input tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'output_tensor_names',
            default_value='["output_tensor"]',
            description='A list of tensor names to bound to the specified output binding names'),
        DeclareLaunchArgument(
            'output_binding_names',
            default_value='[""]',
            description='A list of output tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'verbose',
            default_value='False',
            description='Whether TensorRT should verbosely log or not'),
        DeclareLaunchArgument(
            'force_engine_update',
            default_value='False',
            description='Whether TensorRT should update the TensorRT engine file or not'),
        DeclareLaunchArgument(
	    'num_classes',
	    default_value='6',  # 或你默认想用的数字，比如 '6'
	    description='Number of classes for YOLOv8 decoder'
	),
    ]

    # DNN Image Encoder parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')

    # TensorRT parameters
    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')
    input_tensor_names = LaunchConfiguration('input_tensor_names')
    input_binding_names = LaunchConfiguration('input_binding_names')
    output_tensor_names = LaunchConfiguration('output_tensor_names')
    output_binding_names = LaunchConfiguration('output_binding_names')
    verbose = LaunchConfiguration('verbose')
    force_engine_update = LaunchConfiguration('force_engine_update')

    # YOLOv8 Decoder parameters
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    nms_threshold = LaunchConfiguration('nms_threshold')
    num_classes = LaunchConfiguration('num_classes')

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    yolov8_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': image_mean,
            'image_stddev': image_stddev,
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'tensor_rt_container',
            'dnn_image_encoder_namespace': 'yolov8_encoder',
            'image_input_topic': '/image',
            'camera_info_input_topic': '/camera_info',
            'tensor_output_topic': '/tensor_pub',
        }.items(),
    )

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'output_binding_names': output_binding_names,
            'output_tensor_names': output_tensor_names,
            'input_tensor_names': input_tensor_names,
            'input_binding_names': input_binding_names,
            'verbose': verbose,
            'force_engine_update': force_engine_update
        }]
    )

    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder_node',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
            'num_classes': num_classes,
        }]
    )

    tensor_rt_container = ComposableNodeContainer(
        name='tensor_rt_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[tensor_rt_node, yolov8_decoder_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO'],
        namespace=''
    )

    final_launch_description = launch_args + [tensor_rt_container, yolov8_encoder_launch]
    return launch.LaunchDescription(final_launch_description)
```

上述两个launch.py文件修改后，num_classes:=6才能传给yolov8_decoder_node.cpp

##### 2.要实现自己的模型部署，不仅得修改num_classes，还得修改类别名

在isaac_ros_yolov8_visualizer.py中，修改names字典

```
names = {
    0: 'blue',
    1: 'green',
    2: 'orange',
    3: 'purple',
    4: 'red',
    5: 'yellow',
}
```

##### 3.修改yolov8_tensor_rt.launch.py，对应D435i发布的图像话题

'image_input_topic': '/camera/camera/color/image_raw',（从/image修改至/camera/camera/color/image_raw）
'camera_info_input_topic': '/camera/camera/color/camera_info',（从/camera_info修改至/camera/camera/color/camera_info）

```
yolov8_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': image_mean,
            'image_stddev': image_stddev,
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'tensor_rt_container',
            'dnn_image_encoder_namespace': 'yolov8_encoder',
            'image_input_topic': '/camera/camera/color/image_raw',
            'camera_info_input_topic': '/camera/camera/color/camera_info',
            'tensor_output_topic': '/tensor_pub',
        }.items(),
    )
```

注意：如果你的图像话题不是'/camera/camera/color/image_raw'和'/camera/camera/color/camera_info'，可以用relay功能包将其你提供的话题转发至它们上面

##### 4.在docker中开rqt_image_view浪费资源，故实现rqt_image_view开关

在原文件里加一个布尔参数 `use_rqt`，默认为false，并用 `IfCondition` 控制 `rqt_image_view` 是否启动。不过要想查看图像时，最好在宿主机开rqt_image_view。

isaac_ros_yolov8_visualize.launch.py文件修改

```
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# …（上面头部许可证不变）…

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
```

### 自定义模型使用方法

如果想要clone该项目然后使用自己训练的yolo模型

##### 1.替换模型.onnx

在isaac_ros_ws/src/isaac_ros_assets/models/yolov8目录下，替换成自己训练的模型参数文件

##### 2.修改类别数

num_classes参数修改成自己的类别数

```
ros2 launch isaac_ros_yolov8 isaac_ros_yolov8_visualize.launch.py \
  ... \
  num_classes:=修改成自己的类别数
```

##### 3.修改类别名

在isaac_ros_yolov8_visualizer.py中，修改names字典，以下是我的示例

```
names = {
    0: 'blue',
    1: 'green',
    2: 'orange',
    3: 'purple',
    4: 'red',
    5: 'yellow',          待修改的names字典
}
```
