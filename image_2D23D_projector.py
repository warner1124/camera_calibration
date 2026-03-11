import rclpy
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import CameraInfo
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Pose
from image_geometry import PinholeCameraModel
import numpy as np

class VisionTo3DFusion(Node):
    def __init__(self):
        super().__init__('vision_3d_projector')
        
        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        self.roi_sub = self.create_subscription(DetectedObjectsWithFeature, '/perception/object_recognition/detection/rois0', self.roi_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/sensing/camera/camera0/camera_info', self.info_callback, 10)
        
        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/vision_3d_markers', 10)
        
        self.cam_model = PinholeCameraModel()
        self.target_frame = 'seyond_left'
        self.ground_frame = 'base_link'

        # Label Sizes: [Length, Width, Height]
        self.label_configs = {
            0: [1.0, 1.0, 1.0], # UNKNOWN
            1: [4.5, 1.8, 1.5], # CAR
            2: [7.0, 2.5, 3.0], # TRUCK
            3: [10.0, 2.5, 3.5],# BUS
            4: [12.0, 2.5, 3.5],# TRAILER
            5: [2.0, 0.8, 1.5], # MOTORCYCLE
            6: [1.8, 0.6, 1.5], # BICYCLE
            7: [0.8, 0.8, 1.7], # PEDESTRIAN
            8: [1.0, 0.5, 0.8], # ANIMAL
        }

    def info_callback(self, msg):
        self.cam_model.fromCameraInfo(msg)

    def roi_callback(self, msg):
        if self.cam_model.projection_matrix is None:
            return

        try:
            # Get transform from camera to base_link to find ground intersection
            t_cam_to_base = self.tf_buffer.lookup_transform(self.ground_frame, msg.header.frame_id, rclpy.time.Time())
            # Get transform from base_link to target frame
            t_base_to_target = self.tf_buffer.lookup_transform(self.target_frame, self.ground_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")
            return

        marker_array = MarkerArray()
        for i, obj in enumerate(msg.feature_objects):
            roi = obj.feature.roi
            # 1. Project bottom-center of 2D box to ray in camera frame
            u = roi.x_offset + roi.width / 2
            v = roi.y_offset + roi.height
            ray = self.cam_model.projectPixelTo3dRay((u, v))
            
            # 2. Transform Ray to Ground Frame (base_link)
            # Origin of camera in base_link
            cam_origin_in_base = np.array([t_cam_to_base.transform.translation.x, 
                                           t_cam_to_base.transform.translation.y, 
                                           t_cam_to_base.transform.translation.z])
            
            # Note: We need to rotate the ray vector using the camera rotation
            # A simplified ground projection assuming Z_base = 0:
            # Intersection: cam_z + k * ray_z_in_base = 0  => k = -cam_z / ray_z_in_base
            
            # Transform ray vector into base_link (ignoring translation)
            ray_stamped = PointStamped()
            ray_stamped.point.x, ray_stamped.point.y, ray_stamped.point.z = ray
            ray_in_base = tf2_geometry_msgs.do_transform_point(ray_stamped, t_cam_to_base)
            
            # Direction vector in base_link
            dir_vec = np.array([ray_in_base.point.x - cam_origin_in_base[0],
                                ray_in_base.point.y - cam_origin_in_base[1],
                                ray_in_base.point.z - cam_origin_in_base[2]])
            
            if dir_vec[2] >= 0: continue # Ray points up
            
            k = -cam_origin_in_base[2] / dir_vec[2]
            ground_pt_in_base = PointStamped()
            ground_pt_in_base.header.frame_id = self.ground_frame
            ground_pt_in_base.point.x = cam_origin_in_base[0] + k * dir_vec[0]
            ground_pt_in_base.point.y = cam_origin_in_base[1] + k * dir_vec[1]
            ground_pt_in_base.point.z = 0.0
            
            # 3. Transform point to Target Frame (seyond_left)
            final_pt = tf2_geometry_msgs.do_transform_point(ground_pt_in_base, t_base_to_target)

            # 4. Create Marker
            label_id = obj.object.classification[0].label
            size = self.label_configs.get(label_id, [1.0, 1.0, 1.0])

            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "vision_projection"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = final_pt.point
            marker.pose.position.z += size[2] / 2 # Adjust so box sits on ground
            marker.scale.x, marker.scale.y, marker.scale.z = size
            marker.color.a = 0.6
            marker.color.r, marker.color.g, marker.color.b = (1.0, 0.0, 0.0) if label_id == 7 else (0.0, 1.0, 0.0)
            
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = VisionTo3DFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()