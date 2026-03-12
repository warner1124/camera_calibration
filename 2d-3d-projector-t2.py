import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
import numpy as np
import tf2_ros
from tf2_geometry_msgs import PointStamped

class YoloRoiTo3DFrustum(Node):
    def __init__(self):
        super().__init__('yolo_roi_to_3d_projector')

        # Parameters
        self.target_frame = 'sensor_kit_base_link'
        self.frustum_length = 20.0  # How far the visualization extends (meters)

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        # Change these to match your actual topics
        self.roi_sub = self.create_subscription(
            DetectedObjectsWithFeature,
            '/perception/object_recognition/detection/rois6',
            self.roi_callback,
            10)
        
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/sensing/camera/camera6/camera_info',
            self.info_callback,
            10)

        # Publisher for RViz
        self.marker_pub = self.create_publisher(MarkerArray, 'perception/yolo_frustums', 10)

        self.camera_info = None

    def info_callback(self, msg):
        self.camera_info = msg

    def unproject_pixel(self, u, v, intrinsics):
        """Convert pixel (u,v) to a 3D ray in camera optical frame."""
        # K matrix: [fx 0 cx; 0 fy cy; 0 0 1]
        fx = intrinsics[0]
        fy = intrinsics[4]
        cx = intrinsics[2]
        cy = intrinsics[5]

        x = (u - cx) / fx
        y = (v - cy) / fy
        return np.array([x, y, 1.0])

    def roi_callback(self, msg):
        if self.camera_info is None:
            self.get_logger().warn("Waiting for CameraInfo...")
            return

        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for i, obj in enumerate(obj_feature := msg.feature_objects):
            roi = obj.feature.roi
            label = "Unknown" # Map this to your class names if needed
            
            # 1. Get 4 corners of the ROI
            corners_2d = [
                (roi.x_offset, roi.y_offset),                             # Top-Left
                (roi.x_offset + roi.width, roi.y_offset),                 # Top-Right
                (roi.x_offset + roi.width, roi.y_offset + roi.height),    # Bottom-Right
                (roi.x_offset, roi.y_offset + roi.height)                 # Bottom-Left
            ]

            # 2. Generate rays and transform to target frame
            rays_in_target = []
            try:
                # Find transform from optical link to sensor kit
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    msg.header.stamp,
                    rclpy.duration.Duration(seconds=0.1)
                )

                for u, v in corners_2d:
                    # Point in optical frame (at depth frustum_length)
                    ray_cam = self.unproject_pixel(u, v, self.camera_info.k) * self.frustum_length
                    
                    ps = PointStamped()
                    ps.header = msg.header
                    ps.point.x = ray_cam[0]
                    ps.point.y = ray_cam[1]
                    ps.point.z = ray_cam[2]

                    # Transform to sensor_kit_base_link
                    ps_target = tf2_ros.TransformException
                    ps_target = self.tf_buffer.transform(ps, self.target_frame)
                    rays_in_target.append(ps_target.point)

                # 3. Create Marker (Line List for Frustum)
                frustum_marker = self.create_frustum_marker(i, rays_in_target, msg.header.stamp)
                marker_array.markers.append(frustum_marker)

                # 4. Create Label Marker
                label_marker = self.create_label_marker(i, rays_in_target[0], label, msg.header.stamp)
                marker_array.markers.append(label_marker)

            except Exception as e:
                self.get_logger().error(f"TF Error: {e}")

        self.marker_pub.publish(marker_array)

    def create_frustum_marker(self, id, corners, stamp):
        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = stamp
        m.ns = "frustums"
        m.id = id
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.05 # Line thickness
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8

        # Origin (Camera position in target frame)
        try:
            t = self.tf_buffer.lookup_transform(self.target_frame, self.camera_info.header.frame_id, stamp)
            origin = t.transform.translation
        except:
            return m

        # Lines from camera to corners
        for p in corners:
            m.points.append(origin)
            m.points.append(p)
        
        # Lines connecting the far corners (rectangle)
        for i in range(4):
            m.points.append(corners[i])
            m.points.append(corners[(i+1)%4])

        return m

    def create_label_marker(self, id, position, text, stamp):
        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = stamp
        m.ns = "labels"
        m.id = id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = position
        m.scale.z = 1.0  # Text height
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.text = text
        return m

def main():
    rclpy.init()
    node = YoloRoiTo3DFrustum()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()