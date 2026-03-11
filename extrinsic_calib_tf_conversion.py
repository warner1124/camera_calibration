import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. INPUT DATA (Edit these values)
# ==========================================

# Camera Intrinsics (from your ROS CameraInfo)
# K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
K_ros = [2152.8, 0, 971.3, 0, 2155.5, 605.9, 0, 0, 1]

# Distortion Coefficients
# D: [k1, k2, p1, p2]
D_ros = [-0.1192, 0.162, 0.00073985, 0.0014]

# Transformations (Format: [x, y, z, roll, pitch, yaw])
# Note: Angles are in RADIANS. 
# If you have degrees, use np.radians(value)

# Transform: base_link -> camera0
# (Where is the camera relative to the base?)
tf_base_to_cam0 = [0.2, 0.0, 1.5, 0.0, 0.0, 0.0] 

# Transform: base_link -> seyond_left 
# (Where is the lidar relative to the base?)
tf_base_to_seyond = [2.5, 0.5, 1.8, 0.0, 0.0, 0.0]

# ==========================================
# 2. CALCULATION LOGIC
# ==========================================

def rpy_xyz_to_matrix(tf):
    """Converts [x, y, z, r, p, y] to 4x4 Matrix"""
    x, y, z, roll, pitch, yaw = tf
    mat = np.eye(4)
    # ROS uses the 'xyz' convention for Euler angles (Extrinsic)
    # which is equivalent to 'XYZ' (Intrinsic)
    rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    mat[:3, :3] = rot
    mat[:3, 3] = [x, y, z]
    return mat

# Create 4x4 Matrices
T_base_cam = rpy_xyz_to_matrix(tf_base_to_cam0)
T_base_lidar = rpy_xyz_to_matrix(tf_base_to_seyond)

# We want the transform from Lidar TO Camera:
# T_cam_lidar = (T_base_cam)^-1 * (T_base_lidar)
T_cam_lidar = np.linalg.inv(T_base_cam) @ T_base_lidar

# Extract the 3x4 matrix (Top 3 rows)
T_3x4 = T_cam_lidar[:3, :].flatten()

# ==========================================
# 3. OUTPUT FOR calib.txt
# ==========================================

print("--- COPY AND SAVE AS calib.txt ---")
print(f"K: {' '.join(map(str, K_ros))}")
print(f"D: {' '.join(map(str, D_ros))}")
print(f"T: {' '.join(map(str, T_3x4))}")
print("----------------------------------")
