import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from visualization_msgs.msg import Marker, MarkerArray
from image_geometry import PinholeCameraModel
import numpy as np

class VisionTo3D(Node):
    def __init__(self):
        super().__init__('vision_to_3d_projector')
        
        # Subscriptions
        self.roi_sub = self.create_subscription(DetectedObjectsWithFeature, '/perception/object_recognition/detection/rois0', self.roi_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/sensing/camera/camera0/camera_info', self.info_callback, 10)
        
        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/vision_3d_markers', 10)
        
        self.cam_model = PinholeCameraModel()
        self.label_sizes = {
            0: [4.5, 1.8, 1.5], # CAR (L, W, H)
            1: [0.8, 0.8, 1.7], # PEDESTRIAN
            2: [10.0, 2.5, 3.5],# BUS
            3: [7.0, 2.3, 3.0]  # TRUCK
        }

    def info_callback(self, msg):
        self.cam_model.fromCameraInfo(msg)

    def roi_callback(self, msg):
        if not self.cam_model.projection_matrix is not None:
            return

        marker_array = MarkerArray()
        for i, obj in enumerate(msg.feature_objects):
            roi = obj.feature.roi
            # 1. Get bottom center of 2D box (best for ground projection)
            u = roi.x_offset + roi.width / 2
            v = roi.y_offset + roi.height
            
            # 2. Project ray into 3D
            ray = self.cam_model.projectPixelTo3dRay((u, v))
            
            # 3. Simple Ground Projection (Assuming camera is at height H and ground is Z=0)
            # You should ideally use TF to get camera height. Here we assume a fixed height for demo.
            cam_height = 1.5 
            scale = -cam_height / ray[2] if ray[2] != 0 else 0
            
            x_3d = ray[0] * scale
            y_3d = ray[1] * scale
            z_3d = 0.0 # Ground
            
            # 4. Create 3D Marker
            label = obj.object.classification[0].label
            size = self.label_sizes.get(label, [1.0, 1.0, 1.0])

            marker = Marker()
            marker.header = msg.header
            marker.ns = "vision_detection"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x_3d
            marker.pose.position.y = y_3d
            marker.pose.position.z = size[2] / 2 # Lift to sit on ground
            marker.scale.x, marker.scale.y, marker.scale.z = size
            marker.color.a = 0.5
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = VisionTo3D()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()