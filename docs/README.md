# OceanSim: ROS2 Bridge #

<!-- website link to https://umfieldrobotics.github.io/OceanSim/ -->
<!-- arxiv https://arxiv.org/abs/2503.01074 -->
<!-- subscription form https://docs.google.com/forms/d/e/1FAIpQLSfKWMhE4L6R4jjvEw_bfMtLigXbv5WZeijDah5vk2SpQZW1hA/viewform -->
[![Website](https://img.shields.io/website?down_color=red&down_message=offline&up_color=blue&up_message=online&url=https%3A%2F%2Fumfieldrobotics.github.io%2FOceanSim%2F)](https://umfieldrobotics.github.io/OceanSim/)
[![Subscription Form](https://img.shields.io/badge/Subscribe-Form-blue.svg)](https://docs.google.com/forms/d/e/1FAIpQLSfKWMhE4L6R4jjvEw_bfMtLigXbv5WZeijDah5vk2SpQZW1hA/viewform)
[![arXiv](https://img.shields.io/badge/arXiv-2503.01074-b31b1b.svg)](https://arxiv.org/abs/2503.01074)
[![IsaacSim 5.0.0](https://img.shields.io/badge/IsaacSim-5.0.0-brightgreen.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
<!-- add and scale media/oceansim_demo.gif to full width-->
<!-- ![OceanSim Demo](../media/oceansim_demo.gif) \ -->
<a href="https://umfieldrobotics.github.io/OceanSim/">
  <img src="../media/oceansim_demo.gif" alt="OceanSim Demo" style="width:100%;">
</a>

## Instructions: 
This is the ROS2 version of OceanSim which requires user to set up their ROS2 workspace with Isaac Sim following their official [tutorial](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_ros.html).

Note --[PR14](https://github.com/umfieldrobotics/OceanSim/pull/14#issue-3190565204): 

Before OceanSim extension being activated, the extension isaacsim.ros2.bridge should be activated, otherwise rclpy will fail to be loaded.

We suggest that make sure the extension isaacsim.ros2.bridge is being setup to "AUTOLOADED" in Window->Extension. 


## Usage:
### ros2 control:
We provided an exmaple util located at `isaacsim/oceansim/utils/ros2_control.py` for user to consult and develop on.

This util extends the control mode to ros control in the **sensor_example** extension. 

### ros2 publish uw image:
We add ros2 publish uw image function in the UW_Camera class, located at `isaacsim/oceansim/sensors/UW_Camera.py`.

For testing, we provide a ros2 subscriber example located at `isaacsim/oceansim/utils/ros2_image_subscriber.py`.

Test steps:
1. check the Underwater Camera checkbox in the **sensor_example** extension.
2. run the simulation.
3. open a terminal and run the ros2_image_subscriber.py.
```
cd /path/to/oceansim/utils
python3 ros2_image_subscriber.py
```

### GPU-first ROS2 publishing (UW camera + imaging sonar)
OceanSim now supports GPU-first publication for the processed underwater camera image and imaging sonar fan image. The publishing path uses the ROS2 bridge `ROS2PublishImage` node with `dataPtr` + `cudaDeviceIndex`, so no per-frame CPU copies are required.

What changed:
- **UW_Camera**: processed image can be published from a preallocated GPU buffer (`dataPtr`) with a consistent `bufferSize`, `width`, `height`, and `encoding`.
- **ImagingSonarSensor**: the fan image is produced on GPU and published from a preallocated GPU RGBA buffer.
- **Helper**: `isaacsim/oceansim/pipeline/ros2_graphs.py` provides a small builder to create the ROS2 image graph and set the correct inputs for GPU publishing.

These paths are designed to avoid per-frame `annotator.get_data()` polling and reduce Python/Numpy churn in the main loop.

## Latest Updates
- `[2025/9]` OceanSim is now compatible with Isaac Sim 5.0 GA.
- `[2025/4]` OceanSim is featured by [NVIDIA Robotics](https://www.linkedin.com/posts/nvidiarobotics_robotics-underwaterrobotics-simulation-activity-7313986055894880257-Dfmq?utm_source=share&utm_medium=member_desktop&rcm=ACoAACB8Y7sB7ikB6wVGPL5NrxYkNwk8RTEJ-3Y)!
- `[2025/4]` 🔥 Beta version of OceanSim is released!
- `[2025/3]` 🎉 OceanSim will be presented at [AQ²UASIM](https://sites.google.com/view/aq2uasim/home?authuser=0) and the late-breaking poster session at [ICRA 2025](https://2025.ieee-icra.org/)!
- `[2025/3]` OceanSim paper is available on arXiv. Check it out [here](https://arxiv.org/abs/2503.01074).

## TODO
- [x] Documentation for OceanSim provided example
- [x] Built your own digital twin documentation
- [x] Code release
- [x] ROS2 Example release, contributed from [Tang-JingWei](https://github.com/Tang-JingWei)

## Acknowledgement:
Great appreciation to [Tang-JingWei](https://github.com/Tang-JingWei) for contributng the ros bridge example for OceanSim.
