"""Helpers to build ROS2 image publication graphs inside Isaac Sim."""

from __future__ import annotations

from typing import Any, Dict, Optional

import omni.graph.core as og
import omni.kit.commands
import omni.usd


def create_ros2_image_graph(
    graph_path: str,
    topic_name: str,
    frame_id: str,
    ros_namespace: str,
    queue_size: int,
    encoding: str,
    cuda_device_index: int = -1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    buffer_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a ROS2PublishImage graph and return attribute handles.

    Returns the graph, publish impulse, and image input attribute handles.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is not None and stage.GetPrimAtPath(graph_path).IsValid():
        # A saved scene may still contain the former OnPlaybackTick graph. Editing
        # it in place leaves that trigger connected and republishes stale frames.
        omni.kit.commands.execute("DeletePrims", paths=[graph_path])

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("PublishImpulse", "omni.graph.action.OnImpulseEvent"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishImage", "isaacsim.ros2.bridge.ROS2PublishImage"),
            ],
            keys.CONNECT: [
                ("PublishImpulse.outputs:execOut", "PublishImage.inputs:execIn"),
                ("ROS2Context.outputs:context", "PublishImage.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("PublishImage.inputs:topicName", topic_name),
                ("PublishImage.inputs:frameId", frame_id),
                ("PublishImage.inputs:queueSize", queue_size),
                ("PublishImage.inputs:nodeNamespace", ros_namespace),
                ("PublishImage.inputs:encoding", encoding),
                ("PublishImage.inputs:cudaDeviceIndex", int(cuda_device_index)),
            ],
        },
    )

    graph = og.Controller.graph(graph_path)
    graph.change_pipeline_stage(og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND)
    publish_impulse_attr = og.Controller.attribute(f"{graph_path}/PublishImpulse.state:enableImpulse")
    timestamp_attr = og.Controller.attribute(f"{graph_path}/PublishImage.inputs:timeStamp")
    data_attr = og.Controller.attribute(f"{graph_path}/PublishImage.inputs:data")
    data_ptr_attr = og.Controller.attribute(f"{graph_path}/PublishImage.inputs:dataPtr")
    buffer_size_attr = og.Controller.attribute(f"{graph_path}/PublishImage.inputs:bufferSize")
    width_attr = og.Controller.attribute(f"{graph_path}/PublishImage.inputs:width")
    height_attr = og.Controller.attribute(f"{graph_path}/PublishImage.inputs:height")

    if width_attr is not None and width is not None:
        width_attr.set(int(width))
    if height_attr is not None and height is not None:
        height_attr.set(int(height))
    if buffer_size_attr is not None and buffer_size is not None:
        buffer_size_attr.set(int(buffer_size))

    return {
        "graph": graph,
        "publish_impulse_attr": publish_impulse_attr,
        "timestamp_attr": timestamp_attr,
        "data_attr": data_attr,
        "data_ptr_attr": data_ptr_attr,
        "buffer_size_attr": buffer_size_attr,
        "width_attr": width_attr,
        "height_attr": height_attr,
    }
