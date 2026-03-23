# Omniverse Import
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_image
import omni.ui as ui

# Isaac sim import
from isaacsim.sensors.camera import Camera
import numpy as np
import warp as wp
import yaml
import carb

# Custom import
from isaacsim.oceansim.utils.UWrenderer_utils import UW_render


@wp.kernel
def _rgba_to_rgb(
    src: wp.array(ndim=3, dtype=wp.uint8),
    dst: wp.array(ndim=3, dtype=wp.uint8),
):
    i, j = wp.tid()
    dst[i, j, 0] = src[i, j, 0]
    dst[i, j, 1] = src[i, j, 1]
    dst[i, j, 2] = src[i, j, 2]
from isaacsim.oceansim.pipeline.ros2_graphs import create_ros2_image_graph

class UW_Camera(Camera):

    def __init__(self, 
                 prim_path, 
                 name = "UW_Camera", 
                 frequency = None, 
                 dt = None, 
                 resolution = None, 
                 position = None, 
                 orientation = None, 
                 translation = None, 
                 render_product_path = None):
        
        """Initialize an underwater camera sensor.
    
        Args:
            prim_path (str): prim path of the Camera Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "UW_Camera".
            frequency (Optional[int], optional): Frequency of the sensor (i.e: how often is the data frame updated).
                                                Defaults to None.
            dt (Optional[str], optional): dt of the sensor (i.e: period at which a the data frame updated). Defaults to None.
            resolution (Optional[Tuple[int, int]], optional): resolution of the camera (width, height). Defaults to None.
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            render_product_path (str): path to an existing render product, will be used instead of creating a new render product
                                    the resolution and camera attached to this render product will be set based on the input arguments.
                                    Note: Using same render product path on two Camera objects with different camera prims, resolutions is not supported
                                    Defaults to None
        """
        self._name = name
        self._prim_path = prim_path
        self._res = resolution
        self._writing = False
        self._last_uw_rgba_cpu = None
        self._uw_image_buffer = None
        self._uw_rgb_buffer = None
        self._processed_graph_path = None
        self._processed_data_attr = None
        self._processed_data_ptr_attr = None
        self._processed_buffer_size_attr = None
        self._processed_width_attr = None
        self._processed_height_attr = None
        self._processed_use_gpu = False
        self._processed_cuda_device_index = -1

        super().__init__(prim_path, name, frequency, dt, resolution, position, orientation, translation, render_product_path)

    def initialize(self, 
                   UW_param: np.ndarray = np.array([0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05 ]),
                   viewport: bool = True,
                   writing_dir: str = None,
                   UW_yaml_path: str = None,
                   physics_sim_view=None):
        
        """Configure underwater rendering properties and initialize pipelines.
    
        Args:
            UW_param (np.ndarray, optional): Underwater parameters array:
                [0:3] - Backscatter value (RGB)
                [3:6] - Attenuation coefficients (RGB)
                [6:9] - Backscatter coefficients (RGB)
                Defaults to typical coastal water values.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
            writing_dir (str, optional): Directory to save rendered images. Defaults to None.
            UW_yaml_path (str, optional): Path to YAML file with water properties. Defaults to None.
            physics_sim_view (_type_, optional): _description_. Defaults to None.
    
        """
        self._id = 0
        self._viewport = viewport
        self._device = wp.get_preferred_device()
        super().initialize(physics_sim_view)

        if UW_yaml_path is not None:
            with open(UW_yaml_path, 'r') as file:
                try:
                    # Load the YAML content
                    yaml_content = yaml.safe_load(file)
                    self._backscatter_value = wp.vec3f(*yaml_content['backscatter_value'])
                    self._atten_coeff = wp.vec3f(*yaml_content['atten_coeff'])
                    self._backscatter_coeff = wp.vec3f(*yaml_content['backscatter_coeff'])
                    print(f"[{self._name}] On {str(self._device)}. Using loaded render parameters:")
                    print(f"[{self._name}] Render parameters: {yaml_content}")
                except yaml.YAMLError as exc:
                    carb.log_error(f"[{self._name}] Error reading YAML file: {exc}")
        else:
            self._backscatter_value = wp.vec3f(*UW_param[0:3])
            self._atten_coeff = wp.vec3f(*UW_param[6:9])
            self._backscatter_coeff = wp.vec3f(*UW_param[3:6])
            print(f'[{self._name}] On {str(self._device)}. Using default render parameters.')

        
        self._rgba_annot = rep.AnnotatorRegistry.get_annotator('LdrColor', device=str(self._device))
        self._depth_annot = rep.AnnotatorRegistry.get_annotator('distance_to_camera', device=str(self._device))

        self._rgba_annot.attach(self._render_product_path)
        self._depth_annot.attach(self._render_product_path)

        if self._viewport:
            self.make_viewport()

        if writing_dir is not None:
            self._writing = True
            self._writing_backend = rep.BackendDispatch({"paths": {"out_dir": writing_dir}})

        print(f'[{self._name}] Initialized successfully. Data writing: {self._writing}')

    def setup_processed_ros2_publisher(
        self,
        topic_name: str,
        frame_id: str,
        ros_namespace: str,
        queue_size: int,
        graph_path: str | None = None,
    ) -> None:
        """Create a ROS2PublishImage graph for the processed camera output."""
        device_str = str(self._device)
        self._processed_use_gpu = device_str.startswith("cuda:")
        if self._processed_use_gpu:
            self._processed_cuda_device_index = int(device_str.split(":", 1)[1])
        else:
            self._processed_cuda_device_index = -1

        graph_path = graph_path or f"/ROS2{self._name}ProcessedGraph"
        attrs = create_ros2_image_graph(
            graph_path=graph_path,
            topic_name=topic_name,
            frame_id=frame_id,
            ros_namespace=ros_namespace,
            queue_size=queue_size,
            encoding="rgb8",
            cuda_device_index=self._processed_cuda_device_index,
            width=self.get_resolution()[0],
            height=self.get_resolution()[1],
            buffer_size=int(self.get_resolution()[0] * self.get_resolution()[1] * 3),
        )
        self._processed_graph_path = graph_path
        self._processed_data_attr = attrs["data_attr"]
        self._processed_data_ptr_attr = attrs["data_ptr_attr"]
        self._processed_buffer_size_attr = attrs["buffer_size_attr"]
        self._processed_width_attr = attrs["width_attr"]
        self._processed_height_attr = attrs["height_attr"]

        # Ensure the node uses dataPtr when available
        if self._processed_use_gpu:
            height, width = self.get_resolution()[1], self.get_resolution()[0]
            if self._uw_rgb_buffer is None:
                self._uw_rgb_buffer = wp.zeros(
                    shape=(height, width, 3),
                    dtype=wp.uint8,
                    device=self._device,
                )
            if self._processed_data_ptr_attr is not None:
                self._processed_data_ptr_attr.set(int(self._uw_rgb_buffer.ptr))
        else:
            if self._processed_data_ptr_attr is not None:
                self._processed_data_ptr_attr.set(0)
        if self._processed_data_attr is not None:
            self._processed_data_attr.set([])

    def _process_underwater_frame(self):
        """Fetch sensor inputs and run the underwater Warp kernel once.

        Returns:
            Optional[wp.array]: Processed RGBA frame on the preferred Warp device.
        """
        raw_rgba = self._rgba_annot.get_data()
        depth = self._depth_annot.get_data()
        if raw_rgba.size == 0 or depth.size == 0:
            return None

        if self._uw_image_buffer is None or self._uw_image_buffer.shape != raw_rgba.shape:
            self._uw_image_buffer = wp.zeros_like(raw_rgba)
        else:
            self._uw_image_buffer.zero_()

        uw_image = self._uw_image_buffer
        wp.launch(
            dim=np.flip(self.get_resolution()),
            kernel=UW_render,
            inputs=[
                raw_rgba,
                depth,
                self._backscatter_value,
                self._atten_coeff,
                self._backscatter_coeff
            ],
            outputs=[
                uw_image
            ]
        )

        if self._processed_use_gpu:
            height, width = uw_image.shape[0], uw_image.shape[1]
            if (
                self._uw_rgb_buffer is None
                or self._uw_rgb_buffer.shape[0] != height
                or self._uw_rgb_buffer.shape[1] != width
            ):
                self._uw_rgb_buffer = wp.zeros(
                    shape=(height, width, 3),
                    dtype=wp.uint8,
                    device=self._device,
                )
            else:
                self._uw_rgb_buffer.zero_()
            wp.launch(
                dim=(height, width),
                kernel=_rgba_to_rgb,
                inputs=[uw_image],
                outputs=[self._uw_rgb_buffer],
            )
        return uw_image

    def render(self):
        """Process and display a single frame with underwater effects.
    
        Note:
            - Updates viewport display if enabled
            - Saves image to disk if writing_dir was specified
        
        Returns:
            Optional[np.ndarray]: Latest processed RGBA image on host memory.
                                 Callers that do not need the image can ignore the return value.
        """
        uw_image = self._process_underwater_frame()
        if uw_image is not None:
            if self._viewport:
                self._provider.set_bytes_data_from_gpu(uw_image.ptr, self.get_resolution())
            if self._writing:
                self._writing_backend.schedule(write_image, path=f'UW_image_{self._id}.png', data=uw_image)
                print(f'[{self._name}] [{self._id}] Rendered image saved to {self._writing_backend.output_dir}')

            self._last_uw_rgba_cpu = uw_image.numpy()
            self._id += 1
            return self._last_uw_rgba_cpu
        self._last_uw_rgba_cpu = None
        return None

    def render_rgb(self):
        """Render one frame and return an RGB host image."""
        uw_rgba = self.render()
        if uw_rgba is None:
            return None
        return np.ascontiguousarray(uw_rgba[:, :, :3])

    def publish_processed_frame(self, frame_rgb: np.ndarray) -> None:
        """Push a processed RGB frame into the ROS2PublishImage graph."""
        if (
            frame_rgb is None
            or self._processed_data_attr is None
            or self._processed_buffer_size_attr is None
        ):
            return
        if self._processed_use_gpu:
            return
        if self._processed_width_attr is not None:
            self._processed_width_attr.set(int(frame_rgb.shape[1]))
        if self._processed_height_attr is not None:
            self._processed_height_attr.set(int(frame_rgb.shape[0]))
        self._processed_buffer_size_attr.set(int(frame_rgb.size))
        self._processed_data_attr.set(frame_rgb.reshape(-1))

    def step_processed(self) -> None:
        """Render and publish one processed frame if ROS2 is configured."""
        if self._processed_use_gpu:
            uw_image = self._process_underwater_frame()
            if uw_image is None or self._uw_rgb_buffer is None:
                return
            # Ensure GPU kernels complete before ROS2 reads the pointer.
            wp.synchronize()
            if self._processed_width_attr is not None:
                self._processed_width_attr.set(int(self._uw_rgb_buffer.shape[1]))
            if self._processed_height_attr is not None:
                self._processed_height_attr.set(int(self._uw_rgb_buffer.shape[0]))
            if self._processed_buffer_size_attr is not None:
                self._processed_buffer_size_attr.set(int(self._uw_rgb_buffer.size))
            if self._processed_data_ptr_attr is not None:
                self._processed_data_ptr_attr.set(int(self._uw_rgb_buffer.ptr))
            if self._processed_data_attr is not None:
                self._processed_data_attr.set([])
            return

        frame_rgb = self.render_rgb()
        if frame_rgb is None:
            return
        self.publish_processed_frame(frame_rgb)

    def get_last_uw_rgba(self):
        """Return the latest processed RGBA frame on host memory.

        This is a transitional API that lets application code consume processed
        underwater frames without reaching into private Replicator annotators.
        """
        return self._last_uw_rgba_cpu

    def make_viewport(self):
        """Create a viewport window for real-time visualization.
    
        Note:
            - Window size fixed at 1280x760 pixels
        """
    
        self.wrapped_ui_elements = []
        self.window = ui.Window(self._name, width=1280, height=720 + 40, visible=True)
        self._provider = ui.ByteImageProvider()
        with self.window.frame:
            with ui.ZStack(height=720):
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.Label('Run the scenario for image to be received',
                         style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                         word_wrap=True)
                image_provider = ui.ImageWithProvider(self._provider, width=1280, height=720,
                                     style={'fill_policy': ui.FillPolicy.PRESERVE_ASPECT_FIT,
                                    'alignment' :ui.Alignment.CENTER})
        
        self.wrapped_ui_elements.append(image_provider)
        self.wrapped_ui_elements.append(self._provider)
        self.wrapped_ui_elements.append(self.window)

    # Detach the annotator from render product and clear the data cache
    def close(self):
        """Clean up resources by detaching annotators and clearing caches.
    
        Note:
            - Required for proper shutdown when done using the sensor
            - Also closes viewport window if one was created
        """
        self._rgba_annot.detach(self._render_product_path)
        self._depth_annot.detach(self._render_product_path)

        rep.AnnotatorCache.clear(self._rgba_annot)
        rep.AnnotatorCache.clear(self._depth_annot)

        if self._viewport:
            self.ui_destroy()
            
        print(f'[{self._name}] Annotator detached. AnnotatorCache cleaned.')
    
    
    def ui_destroy(self):
        """Explicitly destroy viewport UI elements.
    
        Note:
            - Called automatically by close()
            - Only needed if manually managing UI lifecycle
        """
        for elem in self.wrapped_ui_elements:
            elem.destroy()

        
       
