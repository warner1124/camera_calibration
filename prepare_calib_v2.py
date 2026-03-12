import rosbag2_py
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R

# --- CUSTOMIZED FOR YOUR TF_STATIC ---
BAG_PATH = 'rosbag2_2026_03_09-15_35_38' # Your bag folder
INFO_TOPIC = '/camera0/camera_info'
LIDAR_FRAME = 'seyond_left'
# In your tf_static, this is the child of camera0/camera_link
OPTICAL_FRAME = 'camera0/camera_optical_link' 

def get_4x4_matrix(t, q):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat

def generate():
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3'), 
                rosbag2_py.ConverterOptions('', ''))

    k_str, d_str = "", ""
    tf_data = {}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == INFO_TOPIC:
            msg = deserialize_message(data, CameraInfo)
            k_str = " ".join(map(str, msg.k))
            d_str = " ".join(map(str, msg.d[:4])) # Use first 4 distortion coeffs
        
        if topic == '/tf_static':
            msg = deserialize_message(data, TFMessage)
            for transform in msg.transforms:
                child = transform.child_frame_id
                parent = transform.header.frame_id
                tf_data[child] = (parent, get_4x4_matrix(transform.transform.translation, transform.transform.rotation))

    # CHAINING THE TRANSFORMS (Finding Optical <- Lidar)
    # Based on your file: 
    # camera_optical_link -> camera_link -> sensor_kit_base_link -> seyond_left_base_link -> seyond_left
    
    def get_full_transform(child_frame):
        current_child = child_frame
        full_mat = np.eye(4)
        while current_child in tf_data:
            parent, mat = tf_data[current_child]
            full_mat = mat @ full_mat
            current_child = parent
        return full_mat

    # T_base_to_optical
    T_b_o = get_full_transform(OPTICAL_FRAME)
    # T_base_to_lidar
    T_b_l = get_full_transform(LIDAR_FRAME)

    # We need T_optical_to_lidar (T_o_l = T_o_b * T_b_l)
    T_o_l = np.linalg.inv(T_b_o) @ T_b_l
    
    # Slice to 3x4 and flatten for PJLab
    t_3x4 = T_o_l[:3, :4].flatten()
    t_str = " ".join(map(str, t_3x4))

    result = f"K: {k_str}\nD: {d_str}\nT: {t_str}"
    with open("calib.txt", "w") as f:
        f.write(result)
    
    print("Successfully generated calib.txt for auto_calib_v2.0")
    print("Verification: Your 3.4m height difference should be reflected in the T matrix.")

generate()