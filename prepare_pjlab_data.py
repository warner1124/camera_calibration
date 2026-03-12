import os
import cv2
import rosbag2_py
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, PointCloud2, CameraInfo
from tf2_msgs.msg import TFMessage
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
BAG_PATH = 'rosbag2_2026_03_09-15_35_38' # Path to your .db3 folder
IMAGE_TOPIC = '/camera0/image_raw/compressed'
LIDAR_TOPIC = '/seyond_left_pointcloud'
INFO_TOPIC = '/camera0/camera_info'

LIDAR_FRAME = 'seyond_left'
OPTICAL_FRAME = 'camera0/camera_optical_link'

# --- DYNAMIC OUTPUT DIRECTORY ---
# This takes 'path/to/my_bag' and makes 'my_bag_calib_data'
bag_name = os.path.basename(os.path.normpath(BAG_PATH))
OUTPUT_DIR = f"{bag_name}_calib_data"

PCD_DIR = os.path.join(OUTPUT_DIR, 'pcd')
IMG_DIR = os.path.join(OUTPUT_DIR, 'image')
os.makedirs(PCD_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

SYNC_THRESHOLD_NS = 50000000 # 50ms synchronization window

def get_4x4_matrix(t, q):
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat

def save_pcd(points, filename):
    """Saves points to ASCII PCD format (Required for some OpenCalib tools)"""
    header = (
        "# .PCD v0.7 - Point Cloud Data\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\nDATA ascii\n"
    )
    with open(filename, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.2f}\n")

def process_bag():
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3'), 
                rosbag2_py.ConverterOptions('', ''))

    images = [] 
    lidars = [] 
    tf_data = {}
    k_str, d_str = "", ""

    print(f"Opening Bag: {BAG_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")

    while reader.has_next():
        topic, data, t_bag = reader.read_next()
        
        if topic == IMAGE_TOPIC:
            msg = deserialize_message(data, CompressedImage)
            images.append((msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec, msg))
            
        elif topic == LIDAR_TOPIC:
            msg = deserialize_message(data, PointCloud2)
            lidars.append((msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec, msg))
            
        elif topic == INFO_TOPIC and not k_str:
            msg = deserialize_message(data, CameraInfo)
            k_str = " ".join(map(str, msg.k))
            d_str = " ".join(map(str, msg.d[:4]))
            
        elif topic == '/tf_static':
            msg = deserialize_message(data, TFMessage)
            for transform in msg.transforms:
                tf_data[transform.child_frame_id] = (transform.header.frame_id, 
                    get_4x4_matrix(transform.transform.translation, transform.transform.rotation))

    # --- 1. GENERATE CALIB.TXT (Coordinate Tree Climbing) ---
    def get_full_transform(child_frame):
        curr = child_frame
        mat = np.eye(4)
        while curr in tf_data:
            parent, t_mat = tf_data[curr]
            mat = t_mat @ mat
            curr = parent
        return mat

    # Calculate Lidar to Camera Optical transform
    T_base_opt = get_full_transform(OPTICAL_FRAME)
    T_base_lid = get_full_transform(LIDAR_FRAME)
    
    # We need Lidar relative to Camera (T_optical_to_lidar)
    T_o_l = np.linalg.inv(T_base_opt) @ T_base_lid
    t_3x4 = T_o_l[:3, :4].flatten()
    
    with open(os.path.join(OUTPUT_DIR, "calib.txt"), "w") as f:
        f.write(f"K: {k_str}\nD: {d_str}\nT: {' '.join(map(str, t_3x4))}")
    print(f"Generated calib.txt using TF tree.")

    # --- 2. SYNCHRONIZE AND SAVE DATA ---
    print(f"Synchronizing {len(images)} images and {len(lidars)} LiDAR frames...")
    count = 0
    
    # We sort to ensure chronological sync
    images.sort(key=lambda x: x[0])
    lidars.sort(key=lambda x: x[0])

    for t_img, img_msg in images:
        # Find closest LiDAR message by timestamp
        # Optimization: binary search could be used, but simple min works for small/mid bags
        closest_lidar = min(lidars, key=lambda x: abs(x[0] - t_img))
        time_diff = abs(closest_lidar[0] - t_img)

        if time_diff < SYNC_THRESHOLD_NS:
            # Save Image (Decompressing on the fly)
            np_arr = np.frombuffer(img_msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Tier4 driver flipped the image? CV2 writes it as-is (upright)
            cv2.imwrite(os.path.join(IMG_DIR, f"{count}.png"), cv_img)

            # Save PCD (Extracting X,Y,Z and Intensity)
            pcd_msg = closest_lidar[1]
            # Seyond Robin W intensity is already 0-255 based on your check
            points = pc2.read_points(pcd_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
            save_pcd(list(points), os.path.join(PCD_DIR, f"{count}.pcd"))
            
            count += 1

    print(f"Success! {count} frame pairs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_bag()