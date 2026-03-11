import rclpy
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import CameraInfo
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
import numpy as np

class VisionTo3DFusionLabels(Node):
    def __init__(self):
        super().__init__('vision_3d_projector_with_labels')
        
        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Topics - Update these if your bag uses different names
        self.roi_topic = '/perception/object_recognition/detection/rois0'
        self.info_topic = '/sensing/camera/camera0/camera_info'
        
        # Subscriptions
        self.roi_sub = self.create_subscription(DetectedObjectsWithFeature, self.roi_topic, self.roi_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_callback, 10)
        
        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/vision_3d_markers', 10)
        
        self.cam_model = PinholeCameraModel()
        self.target_frame = 'seyond_left'
        self.ground_frame = 'base_link' # We project onto Z=0 in this frame

        # Label Mapping: ID -> [Name, Length, Width, Height]
        self.label_map = {
            0: ["UNKNOWN",    1.0, 1.0, 1.0],
            1: ["CAR",        4.5, 1.8, 1.5],
            2: ["TRUCK",      7.0, 2.5, 3.0],
            3: ["BUS",       10.0, 2.5, 3.5],
            4: ["TRAILER",   12.0, 2.5, 3.5],
            5: ["MOTORCYCLE", 2.0, 0.8, 1.5],
            6: ["BICYCLE",    1.8, 0.6, 1.5],
            7: ["PEDESTRIAN", 0.8, 0.8, 1.7],
            8: ["ANIMAL",     1.0, 0.5, 0.8],
        }

    def info_callback(self, msg):
        self.cam_model.fromCameraInfo(msg)

    def roi_callback(self, msg):
        if self.cam_model.projection_matrix is None:
            return

        try:
            # Transform from camera to ground frame to find intersection
            t_cam_to_base = self.tf_buffer.lookup_transform(self.ground_frame, msg.header.frame_id, rclpy.time.Time())
            # Transform from ground to the target display frame (seyond_left)
            t_base_to_target = self.tf_buffer.lookup_transform(self.target_frame, self.ground_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")
            return

        marker_array = MarkerArray()
        
        for i, obj in enumerate(msg.feature_objects):
            # 1. Project 2D Bottom-Center to 3D Ray
            roi = obj.feature.roi
            u = roi.x_offset + roi.width / 2
            v = roi.y_offset + roi.height
            ray = self.cam_model.projectPixelTo3dRay((u, v))
            
            # 2. Ray-Ground Intersection Logic
            cam_origin_in_base = np.array([t_cam_to_base.transform.translation.x, 
                                           t_cam_to_base.transform.translation.y, 
                                           t_cam_to_base.transform.translation.z])
            
            # Use TF to rotate ray vector into base_link
            ray_pt = PointStamped()
            ray_pt.point.x, ray_pt.point.y, ray_pt.point.z = ray
            ray_in_base_pt = tf2_geometry_msgs.do_transform_point(ray_pt, t_cam_to_base)
            
            # Direction vector (Base Link)
            dir_vec = np.array([ray_in_base_pt.point.x - cam_origin_in_base[0],
                                ray_in_base_pt.point.y - cam_origin_in_base[1],
                                ray_in_base_pt.point.z - cam_origin_in_base[2]])
            
            if dir_vec[2] >= 0: continue # Ray doesn't hit ground
            
            k = -cam_origin_in_base[2] / dir_vec[2]
            ground_pt_in_base = PointStamped()
            ground_pt_in_base.header.frame_id = self.ground_frame
            ground_pt_in_base.point.x = cam_origin_in_base[0] + k * dir_vec[0]
            ground_pt_in_base.point.y = cam_origin_in_base[1] + k * dir_vec[1]
            ground_pt_in_base.point.z = 0.0
            
            # 3. Final transform to 'seyond_left'
            final_pt = tf2_geometry_msgs.do_transform_point(ground_pt_in_base, t_base_to_target)

            # 4. Get Label Info
            label_id = obj.object.classification[0].label
            label_name, l, w, h = self.label_map.get(label_id, ["UNKNOWN", 1.0, 1.0, 1.0])

            # --- BOX MARKER ---
            box_marker = Marker()
            box_marker.header.frame_id = self.target_frame
            box_marker.header.stamp = self.get_clock().now().to_msg()
            box_marker.ns = "boxes"
            box_marker.id = i
            box_marker.type = Marker.CUBE
            box_marker.pose.position = final_pt.point
            box_marker.pose.position.z += h / 2
            box_marker.scale.x, box_marker.scale.y, box_marker.scale.z = l, w, h
            box_marker.color.a = 0.5
            box_marker.color.r, box_marker.color.g, box_marker.color.b = (1.0, 1.0, 0.0) # Yellow Boxes
            marker_array.markers.append(box_marker)

            # --- TEXT MARKER ---
            text_marker = Marker()
            text_marker.header = box_marker.header
            text_marker.ns = "labels"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.text = f"{label_name} ({obj.object.classification[0].probability:.2f})"
            text_marker.pose.position = final_pt.point
            text_marker.pose.position.z += h + 0.5 # Float 0.5m above the box
            text_marker.scale.z = 0.5 # Text Height
            text_marker.color.a = 1.0
            text_marker.color.r, text_marker.color.g, text_marker.color.b = (1.0, 1.0, 1.0) # White Text
            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = VisionTo3DFusionLabels()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
