# OceanSim: ROS2 Bridge #

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

## Acknowledgement:
Great appreciation to [Tang-JingWei](https://github.com/Tang-JingWei) for contributng the ros bridge example for OceanSim.
