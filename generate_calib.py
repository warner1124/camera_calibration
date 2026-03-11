import rosbag
import numpy as np
import tf_conversions # standard in ROS
from tf.transformations import quaternion_matrix, concatenate_matrices, inverse_matrix

# --- CONFIGURATION ---
BAG_FILE = 'your_data.bag'
CAMERA_INFO_TOPIC = '/camera0/camera_info'
LIDAR_FRAME = 'seyond_left'
CAMERA_LINK_FRAME = 'camera0/camera_link'
# Standard ROS Optical Frame naming convention
CAMERA_OPTICAL_FRAME = 'camera0/camera_optical_frame' 

def msg_to_matrix(transform_msg):
    """Converts geometry_msgs/Transform to 4x4 matrix"""
    t = transform_msg.translation
    q = transform_msg.rotation
    mat = quaternion_matrix([q.x, q.y, q.z, q.w])
    mat[0,3], mat[1,3], mat[2,3] = t.x, t.y, t.z
    return mat

def get_calib_data(bag_path):
    bag = rosbag.Bag(bag_path)
    
    # 1. Get Intrinsics (K and D)
    k_list, d_list = None, None
    for topic, msg, t in bag.read_messages(topics=[CAMERA_INFO_TOPIC]):
        k_list = list(msg.K)
        d_list = list(msg.D)[:4] # PJLab usually expects 4 coefficients
        break
    
    # 2. Get Extrinsics from /tf_static
    # We need to find: LiDAR -> Base -> Camera_Link -> Camera_Optical
    tf_map = {}
    for topic, msg, t in bag.read_messages(topics=['/tf_static', '/tf']):
        for transform in msg.transforms:
            tf_map[transform.child_frame_id] = {
                'parent': transform.header.frame_id,
                'matrix': msg_to_matrix(transform.transform)
            }

    bag.close()

    # Build the Transform Chain: Optical <- Link <- Base <- LiDAR
    # A) Base to LiDAR
    T_base_lidar = tf_map[LIDAR_FRAME]['matrix']
    # B) Link to Base (Inverse of Base to Link)
    T_base_link = tf_map[CAMERA_LINK_FRAME]['matrix']
    T_link_base = inverse_matrix(T_base_link)
    
    # C) Link to Optical (The standard ROS rotation)
    # If not in TF, we apply the standard: Z=Forward, X=Right, Y=Down
    if CAMERA_OPTICAL_FRAME in tf_map:
        T_optical_link = inverse_matrix(tf_map[CAMERA_OPTICAL_FRAME]['matrix'])
    else:
        # Static rotation: URDF standard
        # R_link_optical = R_z(-90) * R_x(-90)
        T_optical_link = np.array([
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1]
        ])

    # Final Extrinsic: T_optical_lidar = T_opt_link * T_link_base * T_base_lidar
    T_final = concatenate_matrices(T_optical_link, T_link_base, T_base_lidar)
    
    # Slice to 3x4 and flatten for PJLab format
    T_3x4 = T_final[:3, :4].flatten()

    # Format Output
    output = []
    output.append(f"K: {' '.join(map(str, k_list))}")
    output.append(f"D: {' '.join(map(str, d_list))}")
    output.append(f"T: {' '.join(map(str, T_3x4))}")
    
    return "\n".join(output)

if __name__ == "__main__":
    result = get_calib_data(BAG_FILE)
    with open("calib.txt", "w") as f:
        f.write(result)
    print("Successfully generated calib.txt")