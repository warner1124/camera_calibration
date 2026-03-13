import os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R

# --- SETTINGS ---
EXTRINSIC_FILE = '/home/chen/Downloads/SensorsCalibration-master/lidar2camera/auto_calib_v2.0/data/camera6/2/extrinsic.txt'
BAG_PATH = '/home/chen/camera-calib-test-bags/20260312-13-calib-bags/rosbag2_2026_03_12-15_23_02'

LIDAR_FRAME = 'seyond_left'
OPTICAL_FRAME = 'camera6/camera_optical_link'
BASE_LINK_FRAME = 'sensor_kit_base_link' 
CAMERA_LINK = 'camera6/camera_link'

def get_4x4_matrix(t, q):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat

def parse_pjlab_extrinsic(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    start = content.find('[')
    end = content.rfind(']') + 1
    matrix_str = content[start:end].replace('\n', '').replace(' ', '')
    rows = matrix_str.strip('[]').split('],[')
    mat = []
    for row in rows:
        mat.append([float(val) for val in row.split(',')])
    return np.array(mat)

def get_static_tf_tree(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'), 
                rosbag2_py.ConverterOptions('', ''))
    tf_data = {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/tf_static':
            msg = deserialize_message(data, TFMessage)
            for transform in msg.transforms:
                tf_data[transform.child_frame_id] = (
                    transform.header.frame_id, 
                    get_4x4_matrix(transform.transform.translation, transform.transform.rotation)
                )
    return tf_data

def get_full_transform(tf_data, child_frame, target_parent):
    curr = child_frame
    mat = np.eye(4)
    while curr != target_parent:
        if curr not in tf_data: return None
        parent, t_mat = tf_data[curr]
        mat = t_mat @ mat
        curr = parent
    return mat

def main():
    # 1. Load Matrix
    T_calib = parse_pjlab_extrinsic(EXTRINSIC_FILE)
    
    # --- THE CRITICAL FIX ---
    # The tool said LiDAR is at +Y (Down). But LiDAR is Above.
    # We flip the Y-translation sign to correct for the software image flip.
    T_calib[1, 3] = -T_calib[1, 3] 
    
    # 2. Get TF Tree
    tf_tree = get_static_tf_tree(BAG_PATH)
    T_base_lidar = get_full_transform(tf_tree, LIDAR_FRAME, BASE_LINK_FRAME)
    T_link_opt = get_full_transform(tf_tree, OPTICAL_FRAME, CAMERA_LINK)
    
    # 3. Calculate Final Transform (Option B Logic with the Y-fix)
    T_sk_c_link = T_base_lidar @ np.linalg.inv(T_calib) @ np.linalg.inv(T_link_opt)

    # 4. Extract Autoware Values
    pos = T_sk_c_link[:3, 3]
    euler = R.from_matrix(T_sk_c_link[:3, :3]).as_euler('xyz', degrees=False)

    print(f"LiDAR height was: {T_base_lidar[2, 3]:.4f}")
    print("\n--- FINAL CORRECTED CALIBRATION FOR Autoware YAML ---")
    print(f"camera0/camera_link:")
    print(f"  x: {pos[0]:.6f}")
    print(f"  y: {pos[1]:.6f}")
    print(f"  z: {pos[2]:.6f}  (Should be ~3.05)")
    print(f"  roll: {euler[0]:.6f}")
    print(f"  pitch: {euler[1]:.6f}")
    print(f"  yaw: {euler[2]:.6f}")

if __name__ == "__main__":
    main()
