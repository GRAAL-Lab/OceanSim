from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import omni.ui as ui
import numpy as np
from omni.replicator.core.scripts.functional import write_np
import warp as wp
from isaacsim.oceansim.utils.ImagingSonar_kernels import *
from isaacsim.oceansim.pipeline.ros2_graphs import create_ros2_image_graph


@wp.kernel
def _sonar_fan_rgba(
    src: wp.array(ndim=3, dtype=wp.uint8),
    dst: wp.array(ndim=3, dtype=wp.uint8),
    half_fov_rad: float,
    min_radius_norm: float,
):
    i, j = wp.tid()
    dst_h = dst.shape[0]
    dst_w = dst.shape[1]
    src_h = src.shape[0]
    src_w = src.shape[1]

    x = wp.float32(j)
    y = wp.float32(i)
    cx = wp.float32(dst_w - 1) * 0.5
    cy = wp.float32(dst_h - 1)
    dx = x - cx
    dy = cy - y

    valid = False
    if dy >= 0.0:
        theta = wp.atan2(dx, wp.max(dy, wp.float32(1.0e-6)))
        radius_norm = wp.sqrt(dx * dx + dy * dy) / wp.max(cy, wp.float32(1.0))
        if (
            wp.abs(theta) <= half_fov_rad
            and radius_norm >= min_radius_norm
            and radius_norm <= 1.0
        ):
            theta_norm = (theta + half_fov_rad) / wp.max(2.0 * half_fov_rad, wp.float32(1.0e-6))
            radius_scaled = (radius_norm - min_radius_norm) / wp.max(1.0 - min_radius_norm, wp.float32(1.0e-6))
            src_j = wp.min(wp.int32(theta_norm * wp.float32(src_w - 1) + 0.5), src_w - 1)
            src_i = wp.min(wp.int32(radius_scaled * wp.float32(src_h - 1) + 0.5), src_h - 1)
            for c in range(4):
                dst[i, j, c] = src[src_i, src_j, c]
            valid = True

    if not valid:
        dst[i, j, 0] = wp.uint8(0)
        dst[i, j, 1] = wp.uint8(0)
        dst[i, j, 2] = wp.uint8(0)
        dst[i, j, 3] = wp.uint8(255)


# Future TODO
# In future release, wrap this class around RTX lidar

class ImagingSonarSensor(Camera):
    def __init__(self, 
                 prim_path, 
                 name = "ImagingSonar", 
                 frequency = None, 
                 dt = None, 
                 position = None, 
                 orientation = None, 
                 translation = None, 
                 render_product_path = None,
                 physics_sim_view = None,
                 min_range: float = 0.2, # m
                 max_range: float = 3.0, # m
                 range_res: float = 0.008, # deg
                 hori_fov: float = 130.0, # deg
                 vert_fov: float = 20.0, # deg
                 angular_res: float = 0.5, # deg
                 hori_res: int = 3000 # isaac camera render product only accepts square pixel, 
                                      # for now vertical res is automatically set with ratio of hori_fov vs.vert_fov 
                 ):
        
    
        """Initialize an imaging sonar sensor with physical parameters.
    
        Args:
            prim_path (str): prim path of the Camera Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "ImagingSonar".
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

            physics_sim_view (_type_, optional): _description_. Defaults to None.            
            min_range (float, optional): Minimum detection range in meters. Defaults to 0.2.
            max_range (float, optional): Maximum detection range in meters. Defaults to 3.0.
            range_res (float, optional): Range resolution in meters. Defaults to 0.008.
            hori_fov (float, optional): Horizontal field of view in degrees. Defaults to 130.0.
            vert_fov (float, optional): Vertical field of view in degrees. Defaults to 20.0.
            angular_res (float, optional): Angular resolution in degrees. Defaults to 0.5.
            hori_res (int, optional): Horizontal pixel resolution. Defaults to 3000.
    
        Note:
            - Vertical resolution is automatically calculated to maintain aspect ratio
            - Uses Warp for GPU-accelerated sonar image generation
            - Creates polar coordinate meshgrid for sonar returns processing
        """


        self._name = name
        # Raw parameters from Oculus M370s\MT370s\MD370s
        self.max_range = max_range # m (max is 200 m in datasheet )
        self.min_range = min_range # m (min is 0.2 m in datasheet)
        self.range_res = range_res # m (datasheet is 0.008 m)
        self.hori_fov = hori_fov # degree (hori_fov is 130 degrees in datasheet)
        self.vert_fov = vert_fov # degree (vert_fov is 20 degrees in datasheet)
        self.angular_res = angular_res # degree (datasheet is 2 deg)
        self.hori_res= hori_res

        # self.beam_separation = 0.5 # degree (Not USED FOR NOW)!!
        # self.num_beams = 256 # (max number of beams) (NOT USED FOR NOW)!!
        # self.update_rate = 40 # Hz (max update rate) (NOT USED FOR NOW)!!


        # Generate sonar map's r and z meshgrid
        self.min_azi = np.deg2rad(90-self.hori_fov/2)
        r, azi = np.meshgrid(np.arange(self.min_range,self.max_range,self.range_res),
                                       np.arange(np.deg2rad(90-self.hori_fov/2), np.deg2rad(90+self.hori_fov/2), np.deg2rad(self.angular_res)),
                                       indexing='ij')
        self.r = wp.array(r, shape=r.shape, dtype=wp.float32)
        self.azi = wp.array(azi, shape=r.shape, dtype=wp.float32)

        # Load array that doesn't change shapes to cuda for reusage memory
        # Users can also automatically see if they have set a reasonable parameter 
        # for sonar map bin size\resolution once load the sensor
        self.bin_sum = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.bin_count = wp.zeros(shape=self.r.shape, dtype=wp.int32)
        self.binned_intensity = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.sonar_map = wp.zeros(shape=self.r.shape, dtype=wp.vec3)
        self.sonar_image = wp.zeros(shape=(self.r.shape[0], self.r.shape[1], 4), dtype=wp.uint8)
        self.gau_noise = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.range_dependent_ray_noise = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self._fan_rgba_buffer = None

        self.AR = self.hori_fov / self.vert_fov
        self.vert_res = int(self.hori_res / self.AR)
        # By doing this, I am assuming the vertical beam separation
        # is the same as the beam horizontal separation. 
        # This is bacause replicator raytracing is specified as resolutions
        # while non-squre pixel is not supported in Isaac sim. See details below.
        
        super().__init__(prim_path=prim_path, 
                         name=name, 
                         frequency=frequency,
                         dt=dt, 
                         resolution=[self.hori_res, self.vert_res],
                         position=position, 
                         orientation=orientation, 
                         translation=translation, 
                         render_product_path=render_product_path)

        self.set_clipping_range(
            near_distance=self.min_range,
            far_distance=self.max_range
        )
        # This is a bug. Needs to call initialize() before changing aperture
        # https://forums.developer.nvidia.com/t/error-when-setting-a-cameras-vertical-horizontal-aperture/271314
        # This line initialize the camera
        self.initialize(physics_sim_view)

        # Assume the default focal length to compute the desired horizontal aperture
        # The reason why we are doing this is because Isaac sim will fix vertical aperture
        # given aspect ratio for mandating square pixles
        # https://forums.developer.nvidia.com/t/how-to-modify-the-cameras-field-of-view/278427/5
        self.focal_length = self.get_focal_length()
        horizontal_aper = 2 * self.focal_length * np.tan(np.deg2rad(self.hori_fov) / 2)
        self.set_horizontal_aperture(horizontal_aper)
        # Notice if you would like to observe sonar view from linked viewport.
        # Only horizontal fov is displayed correctly while the vertical fov is
        # followed by your viewport aspect ratio settings.
        

    # Initialize the sensor so that annotator is 
    # loaded on cuda and ready to acquire data
    # Data is generated per simulation tick

    # do_array_copy: If True, retrieve a copy of the data array. 
    # This is recommended for workflows using asynchronous
    # backends to manage the data lifetime. 
    # Can be set to False to gain performance if the data is 
    # expected to be used immediately within the writer. Defaults to True.

    def sonar_initialize(
        self,
        output_dir: str = None,
        viewport: bool = True,
        include_unlabelled=False,
        if_array_copy: bool = True,
        fetch_on_device: bool = False,
    ):
        """Initialize sonar data processing pipeline and annotators.
    
        Args:
            output_dir (str, optional): Directory to save sonar data. Defaults to None.
                                        If set to None, sonar will not write data.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
                                        Set to False for Sonar running without visualization.
            include_unlabelled (bool, optional): Include unlabelled objects to be scanned into sonar view. Defaults to False.
            if_array_copy (bool, optional): If True, retrieve a copy of the data array. 
                                            This is recommended for workflows using asynchronous backends to manage the data lifetime. 
                                            Can be set to False to gain performance if the data is expected to be used immediately within the writer. 
                                            Defaults to True.
            fetch_on_device (bool, optional): If True, request Replicator pointcloud data directly
                                            on the Warp device. Defaults to False to favor stability by
                                            fetching on host and explicitly copying into Warp arrays.
                                            
        Note:
            - Attaches pointcloud, camera params, and semantic segmentation annotators
            - Sets up Warp arrays for sonar image processing
            - Can optionally write data to disk if output_dir specified
        """
        self.writing = False
        self._viewport = viewport
        self._device = str(wp.get_preferred_device())
        self._fetch_on_device = bool(fetch_on_device)
        annotator_device = self._device if self._fetch_on_device else "cpu"
        self._device_array_cache = {}
        self._index_to_prop_cache = {}
        self._temp_array_cache = {}
        self.scan_data = {}
        self.id = 0
        self._ros2_graph_path = None
        self._ros2_data_attr = None
        self._ros2_data_ptr_attr = None
        self._ros2_buffer_size_attr = None
        self._ros2_width_attr = None
        self._ros2_height_attr = None
        self._ros2_use_gpu = False
        self._ros2_cuda_device_index = -1

        self.pointcloud_annot = rep.AnnotatorRegistry.get_annotator(
            name="pointcloud",
            init_params={"includeUnlabelled": include_unlabelled},
            do_array_copy=if_array_copy,
            device=annotator_device
            )
        
        self.cameraParams_annot = rep.AnnotatorRegistry.get_annotator(
            name="CameraParams",
            do_array_copy=if_array_copy,
            device=annotator_device
            )
        
        self.semanticSeg_annot = rep.AnnotatorRegistry.get_annotator(
            name='semantic_segmentation',
            init_params={"colorize": False},
            do_array_copy=if_array_copy,
            device=annotator_device
        )

        print(f'[{self._name}] Using {self._device} (annotator fetch: {"device" if self._fetch_on_device else "host"})' )
        print(f'[{self._name}] Render query res: {self.hori_res} x {self.vert_res}. Binning res: {self.r.shape[0]} x {self.r.shape[1]}')

        self.pointcloud_annot.attach(self._render_product_path)
        self.cameraParams_annot.attach(self._render_product_path)
        self.semanticSeg_annot.attach(self._render_product_path)
        
        if output_dir is not None:
            self.writing = True
            self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})
        if self._viewport:
            self.make_sonar_viewport()
        
        print(f'[{self._name}] Initialized successfully. Data writing: {self.writing}')

        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()
        self.sonar_map.zero_()
        self.sonar_image.zero_()
        self.range_dependent_ray_noise.zero_()
        self.gau_noise.zero_()

    def setup_ros2_publisher(
        self,
        topic_name: str,
        frame_id: str,
        ros_namespace: str,
        queue_size: int,
        stream_width: int,
        stream_height: int,
        graph_path: str | None = None,
    ) -> None:
        """Create a ROS2PublishImage graph for the sonar RGBA output."""
        device_str = str(self._device)
        self._ros2_use_gpu = device_str.startswith("cuda:")
        if self._ros2_use_gpu:
            self._ros2_cuda_device_index = int(device_str.split(":", 1)[1])
        else:
            self._ros2_cuda_device_index = -1

        graph_path = graph_path or f"/ROS2{self._name}Graph"
        attrs = create_ros2_image_graph(
            graph_path=graph_path,
            topic_name=topic_name,
            frame_id=frame_id,
            ros_namespace=ros_namespace,
            queue_size=queue_size,
            encoding="rgba8",
            cuda_device_index=self._ros2_cuda_device_index,
            width=stream_width,
            height=stream_height,
            buffer_size=int(stream_width * stream_height * 4),
        )
        self._ros2_graph_path = graph_path
        self._ros2_data_attr = attrs["data_attr"]
        self._ros2_data_ptr_attr = attrs["data_ptr_attr"]
        self._ros2_buffer_size_attr = attrs["buffer_size_attr"]
        self._ros2_width_attr = attrs["width_attr"]
        self._ros2_height_attr = attrs["height_attr"]

        if self._ros2_use_gpu:
            if (
                self._fan_rgba_buffer is None
                or self._fan_rgba_buffer.shape[0] != stream_height
                or self._fan_rgba_buffer.shape[1] != stream_width
            ):
                self._fan_rgba_buffer = wp.zeros(
                    shape=(stream_height, stream_width, 4),
                    dtype=wp.uint8,
                    device=self._device,
                )
            if self._ros2_data_ptr_attr is not None:
                self._ros2_data_ptr_attr.set(int(self._fan_rgba_buffer.ptr))
        else:
            if self._ros2_data_ptr_attr is not None:
                self._ros2_data_ptr_attr.set(0)
        if self._ros2_data_attr is not None:
            self._ros2_data_attr.set([])

    def _to_warp_array(self, data, dtype, cache_key=None):
        """Move host-fetched annotator data onto the configured Warp device.

        Replicator host fetches may return flattened buffers for vector-valued
        outputs such as point positions and normals. Warp's vec3 constructors
        require an inner dimension of 3, so normalize the array shape here.
        """
        if isinstance(data, wp.array):
            return data

        array = np.asarray(data)

        if dtype in (wp.vec3, wp.vec3f):
            if array.size == 0:
                array = np.empty((0, 3), dtype=np.float32)
            elif array.ndim == 1:
                if array.size % 3 != 0:
                    raise RuntimeError(
                        f"Cannot convert flattened host array of length {array.size} to {dtype}: expected a multiple of 3"
                    )
                array = array.reshape((-1, 3))
            elif array.shape[-1] == 4:
                array = array[..., :3]
            elif array.shape[-1] != 3:
                if array.size % 3 != 0:
                    raise RuntimeError(
                        f"Cannot convert host array with shape {array.shape} to {dtype}: expected trailing dimension 3"
                    )
                array = array.reshape((-1, 3))
            array = np.ascontiguousarray(array, dtype=np.float32)
        elif dtype in (wp.float32,):
            if array.size == 0:
                array = np.empty((0, 3), dtype=np.float32)
            elif array.ndim == 1:
                if array.size % 3 != 0:
                    raise RuntimeError(
                        f"Cannot convert flattened host array of length {array.size} to float32 point array: expected a multiple of 3"
                    )
                array = array.reshape((-1, 3))
            elif array.shape[-1] == 4:
                array = array[..., :3]
            elif array.shape[-1] != 3:
                if array.size % 3 != 0:
                    raise RuntimeError(
                        f"Cannot convert host array with shape {array.shape} to float32 point array: expected trailing dimension 3"
                    )
                array = array.reshape((-1, 3))
            array = np.ascontiguousarray(array, dtype=np.float32)
        elif dtype in (wp.uint32,):
            array = np.ascontiguousarray(array.reshape(-1), dtype=np.uint32)
        else:
            array = np.ascontiguousarray(array)

        if cache_key is None:
            return wp.array(array, dtype=dtype, device=self._device)

        cached = self._device_array_cache.get(cache_key)
        if cached is None or cached.shape != array.shape or cached.dtype != dtype:
            cached = wp.empty(shape=array.shape, dtype=dtype, device=self._device)
            self._device_array_cache[cache_key] = cached

        src = wp.from_numpy(array, dtype=dtype, device="cpu")
        wp.copy(cached, src)
        return cached

    def _get_temp_array(self, cache_key, shape, dtype):
        cached = self._temp_array_cache.get(cache_key)
        if cached is None or cached.shape != shape or cached.dtype != dtype:
            cached = wp.empty(shape=shape, dtype=dtype, device=self._device)
            self._temp_array_cache[cache_key] = cached
        return cached

    def _build_index_to_prop_key(self, id_to_labels: dict, query_property: str):
        items = []
        for id_key in sorted(id_to_labels.keys(), key=lambda x: int(x)):
            labels = id_to_labels.get(id_key, {})
            items.append((str(id_key), labels.get(query_property, 1.0)))
        return query_property, tuple(items)

    def _get_index_to_prop_array(self, id_to_labels: dict, query_property: str):
        cache_key = self._build_index_to_prop_key(id_to_labels, query_property)
        cached = self._index_to_prop_cache.get(cache_key)
        if cached is not None:
            return cached

        max_id = max(id_to_labels.keys(), default=-1)
        index_to_prop_array = np.ones((int(max_id) + 1,), dtype=np.float32)
        for id_key, labels in id_to_labels.items():
            if query_property in labels:
                index_to_prop_array[int(id_key)] = labels[query_property]

        cached = wp.from_numpy(index_to_prop_array, dtype=wp.float32, device=self._device)
        self._index_to_prop_cache = {cache_key: cached}
        return cached

        
    def _fetch_scan_frame(self):
        """Fetch a single consistent annotator snapshot for the current frame.

        The sonar processing path consumes pointcloud, semantic, and camera-parameter
        data from Replicator. Fetching the same annotator repeatedly within one scan
        causes unnecessary cache reshaping/copies and makes the frame lifetime harder
        to reason about. Consolidating the fetch here keeps the public API unchanged
        while ensuring each scan reuses one snapshot per annotator.
        """
        semantic_frame = self.semanticSeg_annot.get_data()
        id_to_labels = semantic_frame.get('info', {}).get('idToLabels', {})
        if len(id_to_labels) == 0:
            return None

        if self._fetch_on_device:
            pointcloud_frame = self.pointcloud_annot.get_data(device=self._device)
            pcl = pointcloud_frame['data']
            normals = pointcloud_frame['info']['pointNormals']
            semantics = pointcloud_frame['info']['pointSemantic']
        else:
            pointcloud_frame = self.pointcloud_annot.get_data()
            pcl = self._to_warp_array(pointcloud_frame['data'], wp.float32, cache_key="pcl")
            normals = self._to_warp_array(pointcloud_frame['info']['pointNormals'], wp.float32, cache_key="normals")
            semantics = self._to_warp_array(
                pointcloud_frame['info']['pointSemantic'], wp.uint32, cache_key="semantics"
            )

        camera_params_frame = self.cameraParams_annot.get_data()
        return {
            'pcl': pcl,
            'normals': normals,
            'semantics': semantics,
            'viewTransform': camera_params_frame['cameraViewTransform'].reshape(4, 4).T,
            'idToLabels': id_to_labels,
        }

    def scan(self):

        """Capture a single sonar scan frame and store the raw data.
    
        Returns:
            bool: True if scan was successful (valid data received), False otherwise
    
        Note:
            - Stores pointcloud, normals, semantics, and camera transform in scan_data dict
            - First few frames may be empty due to CUDA initialization
            - Automatically skips frames with no detected objects
        """
        # Due to the time to load annotator to cuda, the first few simulation tick gives no annotation in memory.
        # This would also reult error when no mesh within the sonar fov
        # NOTE: Isaac Sim annotator output has squeezed the first dimention after 5.0 update: (1,N,3) -> (N,3)
        scan_frame = self._fetch_scan_frame()
        if scan_frame is None:
            self.scan_data.clear()
            return False
        self.scan_data = scan_frame
        return True


    def make_sonar_data(self, 
                        binning_method: str = "sum", 
                        normalizing_method: str = "range",
                        query_prop: str ='reflectivity', # Do not modify this if not developing the sensor.
                        attenuation: float = 0.1, # Control the attentuation along distance when computing attenuation
                        gau_noise_param: float = 0.2, # multiplicative noise coefficient 
                        ray_noise_param: float = 0.05, # additive noise parameter
                        intensity_offset: float = 0.0, # offset intensity after normalization
                        intensity_gain: float = 1.0, # scale intensity after normalization
                        central_peak: float = 2, # control the strength of the streak
                        central_std: float = 0.001, # control the spread of the streak
                        ):
        """Process raw scan data into a sonar image with configurable parameters.

        Args:
            binning_method (str): "sum" or "mean" for intensity accumulation
                                Remember to adjust your noise scale accordingly after changing this.
            normalizing_method (str): "all" (global max) or "range" (per-range max)
                                Remember to adjust your noise scale accordingly after changing this.
            query_prop (str): Material property to query (default 'reflectivity')
                            Don't modify this if not for development.
            attenuation (float): Distance attenuation coefficient (0-1)
            gau_noise_param (float): Gaussian noise multiplier
            ray_noise_param (float): Rayleigh noise scale factor
            intensity_offset (float): Post-normalization intensity offset
            intensity_gain (float): Post-normalization intensity multiplier
            central_peak (float): Central beam streak intensity
            central_std (float): Central beam streak width
    
        """



        if self.scan():
            num_points = self.scan_data['pcl'].shape[0]
            indexToRefl = self._get_index_to_prop_array(
                id_to_labels=self.scan_data['idToLabels'],
                query_property=query_prop,
            )
            viewTransform=wp.mat44(self.scan_data['viewTransform'])
            # directly use warp array loaded on cuda
            pcl = self.scan_data['pcl']
            normals = self.scan_data['normals']
            semantics = self.scan_data['semantics']
        else:
            return

        # Compute intensity for each ray query     
        intensity = self._get_temp_array("intensity", (num_points,), wp.float32)
        wp.launch(kernel=compute_intensity,
                  dim=num_points,
                  inputs=[
                      pcl,
                      normals,
                      viewTransform,
                      semantics,
                      indexToRefl,
                      attenuation,
                  ],
                  outputs=[
                      intensity
                  ]
                )
                
        # Transform pointcloud from world cooridates to sonar local
        pcl_local = self._get_temp_array("pcl_local", (num_points,), wp.vec3)
        pcl_spher = self._get_temp_array("pcl_spher", (num_points,), wp.vec3)
        wp.launch(kernel=world2local,
                  dim=num_points,
                  inputs=[
                      viewTransform,
                      pcl
                  ],
                    outputs=[
                      pcl_local,
                      pcl_spher
                    ]
                )
        
        # Collapse three dimensional intensity data to 2D
        # Simply sum intensity return and compute number of return that falls into the same bin
        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()

        
        wp.launch(kernel=bin_intensity,
                  dim=num_points,
                  inputs=[
                      pcl_spher,
                      intensity,
                      self.min_range,
                      self.min_azi,
                      self.range_res,
                      wp.radians(self.angular_res),
                  ],
                  outputs=[
                      self.bin_sum,
                      self.bin_count
                  ]
                  )
        
        # Process intensity data by either sum as it is or averaging
        if binning_method == "mean":
            wp.launch(
                kernel=average,
                dim=self.bin_sum.shape,
                inputs=[
                    self.bin_sum,
                    self.bin_count
                ],
                outputs=[
                    self.binned_intensity,
                ]
                )
        
        if binning_method == "sum":
            self.binned_intensity = self.bin_sum


        self.range_dependent_ray_noise.zero_()
        self.gau_noise.zero_()
        self.sonar_map.zero_()

        # Calculate multiplicative gaussian noise
        
        wp.launch(
            kernel=normal_2d,
            dim=self.bin_sum.shape,
            inputs=[
                self.id,   # use frame num for RNG seed increment
                0.0,
                gau_noise_param
            ],
            outputs=[
                self.gau_noise
            ]
        )

        # Calculate additive rayleigh noise (range dependent and mimic central beam)

        wp.launch(
            kernel=range_dependent_rayleigh_2d,
            dim=self.bin_sum.shape,
            inputs=[
                self.id,   # use frame num for RNG seed increment
                self.r,
                self.azi,
                self.max_range,
                ray_noise_param,
                central_peak,
                central_std,
            ],
            outputs=[
                self.range_dependent_ray_noise

            ]
        )

        
        
        # Normalizing intensity at each bin either by global maximum or rangewise maximum
        # Compute global maximum
        if normalizing_method == "all":
            maximum = self._get_temp_array("maximum_all", (1,), wp.float32)
            maximum.zero_()
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=all_max,
                inputs=[
                    self.binned_intensity,
                ],
                outputs=[
                    maximum # wp.array of shape (1,), max value is stored at maximum[0]
                ]
            )
            
            # Apply noise, normalize by global maximum, and convert (r, azi) to (x,y) for plotting
            wp.launch(
                  kernel=make_sonar_map_all,
                  dim=self.sonar_map.shape,
                  inputs=[
                      self.r,
                      self.azi,
                      self.binned_intensity,
                      maximum,
                      self.gau_noise,
                      self.range_dependent_ray_noise,
                      intensity_offset,
                      intensity_gain
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
            
        if normalizing_method == "range":
            # Compute rangewise maximum
            maximum = self._get_temp_array("maximum_range", (self.r.shape[0],), wp.float32)
            maximum.zero_()
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=range_max,
                inputs=[
                    self.binned_intensity,
                ],
                outputs=[
                    maximum      # wp.array of shape (number of range bins, )
                ]
            )
            # Apply noise, normalize by range maximum, and convert (r, azi) to (x,y) for plotting
            wp.launch(
                  kernel=make_sonar_map_range,
                  dim=self.sonar_map.shape,
                  inputs=[
                      self.r,
                      self.azi, 
                      self.binned_intensity,
                      maximum,
                      self.gau_noise,
                      self.range_dependent_ray_noise,
                      intensity_offset,
                      intensity_gain
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
        
        
        # Write data to the dir
        if self.writing:
            # self.backend.schedule(write_np, f"intensity_{self.id}.npy", data=intensity)
            # self.backend.schedule(write_np, f'pcl_local_{self.id}.npy', data=pcl_local)
            self.backend.schedule(write_np, f'sonar_data_{self.id}.npy', data=self.sonar_map)
            print(f"[{self._name}] [{self.id}] Writing sonar data to {self.backend.output_dir}")
        
        if self._viewport:
            self._sonar_provider.set_bytes_data_from_gpu(self.make_sonar_image().ptr, 
                                                    [self.sonar_map.shape[1], self.sonar_map.shape[0]])
            # self.backend.schedule(write_image, f'sonar_{self.id}.png', data = self.make_sonar_image())        
            
        self.id += 1
    

    def make_sonar_image(self):
        """Convert processed sonar data to a viewable grayscale image.
    
        Returns:
            wp.array: GPU array containing the sonar image (RGBA format)
    
        Note:
            - Used internally for viewport display
            - Image dimensions match the sonar's polar binning resolution
        """
        self.sonar_image.zero_()
        wp.launch(
            dim=self.sonar_map.shape,
            kernel=make_sonar_image,
            inputs=[
                self.sonar_map
            ],
            outputs=[
                self.sonar_image
            ]
        )
        return self.sonar_image

    def render_rgba(self, **kwargs):
        """Run one sonar processing step and return the resulting RGBA image.

        Keyword arguments are forwarded to ``make_sonar_data`` so callers can
        configure processing without reaching into lower-level internals.
        """
        self.make_sonar_data(**kwargs)
        return self.make_sonar_image()

    def render_fan_rgba(
        self,
        stream_width: int,
        stream_height: int,
        min_range_m: float,
        max_range_m: float,
        horizontal_fov_deg: float,
        **kwargs,
    ) -> wp.array:
        """Render a sonar frame and map it into a fan-shaped RGBA image."""
        sonar_rgba = self.render_rgba(**kwargs)
        if sonar_rgba is None:
            return None

        if (
            self._fan_rgba_buffer is None
            or self._fan_rgba_buffer.shape[0] != stream_height
            or self._fan_rgba_buffer.shape[1] != stream_width
        ):
            self._fan_rgba_buffer = wp.zeros(
                shape=(stream_height, stream_width, 4),
                dtype=wp.uint8,
                device=self._device,
            )
        else:
            self._fan_rgba_buffer.zero_()

        half_fov_rad = 0.5 * np.deg2rad(float(horizontal_fov_deg))
        min_radius_norm = 0.0
        if max_range_m > min_range_m and min_range_m > 0.0:
            min_radius_norm = float(np.clip(min_range_m / max_range_m, 0.0, 0.99))

        wp.launch(
            dim=(stream_height, stream_width),
            kernel=_sonar_fan_rgba,
            inputs=[
                sonar_rgba,
                self._fan_rgba_buffer,
                half_fov_rad,
                min_radius_norm,
            ],
        )
        return self._fan_rgba_buffer

    def publish_rgba(self, rgba_image: wp.array, stream_width: int, stream_height: int) -> None:
        """Publish an RGBA sonar frame to ROS2 using host data."""
        if (
            rgba_image is None
            or self._ros2_data_attr is None
            or self._ros2_buffer_size_attr is None
        ):
            return
        if self._ros2_width_attr is not None:
            self._ros2_width_attr.set(int(stream_width))
        if self._ros2_height_attr is not None:
            self._ros2_height_attr.set(int(stream_height))
        if self._ros2_use_gpu:
            # Ensure GPU kernels complete before ROS2 reads the pointer.
            wp.synchronize()
            self._ros2_buffer_size_attr.set(int(stream_width * stream_height * 4))
            if self._ros2_data_ptr_attr is not None:
                self._ros2_data_ptr_attr.set(int(rgba_image.ptr))
            self._ros2_data_attr.set([])
            return
        rgba_host = rgba_image.numpy()
        self._ros2_buffer_size_attr.set(int(rgba_host.size))
        self._ros2_data_attr.set(rgba_host.reshape(-1))

    def step_render_publish(
        self,
        stream_width: int,
        stream_height: int,
        min_range_m: float,
        max_range_m: float,
        horizontal_fov_deg: float,
        **kwargs,
    ) -> bool:
        """Render a sonar frame and publish it to ROS2."""
        rgba_image = self.render_fan_rgba(
            stream_width=stream_width,
            stream_height=stream_height,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            horizontal_fov_deg=horizontal_fov_deg,
            **kwargs,
        )
        if rgba_image is None:
            return False
        self.publish_rgba(rgba_image, stream_width, stream_height)
        return True
    

    def make_sonar_viewport(self):
        """Create an interactive viewport window for real-time sonar visualization.
    
        Note:
            - Displays live sonar images when simulation is running
            - Includes range and azimuth tick marks
            - Window size is fixed at 800x800 pixels
        """
        self.wrapped_ui_elements = []

        range_tick_num = 10
        range_tick = np.round(np.linspace(self.min_range, self.max_range, range_tick_num), 2)

        azi_tick_num = 10
        azi_tick = np.round(np.linspace(90-self.hori_fov/2, 90+self.hori_fov/2, azi_tick_num))
        self._sonar_provider = ui.ByteImageProvider()
        self._window = ui.Window(self._name, width=800, height=800, visible=True)
        
        with self._window.frame:
            with ui.ZStack(height=720, width = 720):
                ui.Rectangle(widthstyle={"background_color": 0xFF000000})
                ui.Label('Run the scenario for image to be received',
                         style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                         word_wrap=True)
                sonar_image_provider = ui.ImageWithProvider(self._sonar_provider, 
                                    style={"width": 720, 
                                        "height": 720, 
                                        "fill_policy" : ui.FillPolicy.STRETCH,
                                        'alignment': ui.Alignment.CENTER})
                
                # ui.Line(alignment=ui.Alignment.LEFT,
                #         style={'border_width': 2,
                #                 'color':ui.color.white })
                # with ui.VGrid(row_height = 720/(range_tick_num-1)):
                #     for i in range(range_tick_num-1):
                #         with ui.ZStack():
                #             ui.Rectangle(style={'border_color': ui.color.white, 'background_color': ui.color.transparent,'border_width': 0.05, 'margin': 0})
                #             ui.Label(str(range_tick[i]) + ' m',style={'font_size': 15,'alignment': ui.Alignment.LEFT, 'margin':2})
                # with ui.HGrid(column_width = 720/(azi_tick_num-1), direction=ui.Direction.RIGHT_TO_LEFT):
                #     for i in range(azi_tick_num-1):
                #         with ui.ZStack():
                #             ui.Rectangle(style={'border_color': ui.color.white, 'background_color': ui.color.transparent,'border_width': 0.05, 'margin': 0})
                #             ui.Label(str(azi_tick[i]) + "°",style={'font_size': 15,'alignment': ui.Alignment.RIGHT, 'margin':2})                           
                # ui.Label(str(range_tick[-1]) +" m", style={'font_size': 15, "alignment":ui.Alignment.LEFT_BOTTOM, 'margin':2})
        
        self.wrapped_ui_elements.append(sonar_image_provider)
        self.wrapped_ui_elements.append(self._sonar_provider)
        self.wrapped_ui_elements.append(self._window)

    def get_range(self) -> list[float]:
        """Get the configured operating range of the sonar.
    
        Returns:
            list[float]: [min_range, max_range] in meters
        """
        return [self.min_range, self.max_range]
    
    def get_fov(self) -> list[float]:
        """Get the configured field of view angles.
    
        Returns:
            list[float]: [horizontal_fov, vertical_fov] in degrees
        """
        return [self.hori_fov, self.vert_fov]
    

    
    def close(self):
        """Clean up resources by detaching annotators and clearing caches.
    
        Note:
            - Required for proper shutdown when done using the sensor
            - Also closes viewport window if one was created
        """
        self.pointcloud_annot.detach(self._render_product_path)
        self.cameraParams_annot.detach(self._render_product_path)
        self.semanticSeg_annot.detach(self._render_product_path)

        rep.AnnotatorCache.clear(self.pointcloud_annot)
        rep.AnnotatorCache.clear(self.cameraParams_annot)
        rep.AnnotatorCache.clear(self.semanticSeg_annot)


        print(f'[{self._name}] Annotator detached. AnnotatorCache cleaned.')

        if self._viewport:
            self.ui_destroy()


    def ui_destroy(self):
        """Explicitly destroy viewport UI elements.
    
        Note:
            - Called automatically by close()
            - Only needed if manually managing UI lifecycle
        """
        for elem in self.wrapped_ui_elements:
            elem.destroy()
