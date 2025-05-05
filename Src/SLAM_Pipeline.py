import cv2
import numpy as np
from include.data_loader import get_image_paths, load_intrinsic_matrix
from include.feature_extractor import extract_features
from include.pose_estimator import estimate_pose
from include.trajectory_plotter import plot_trajectory

def run_slam(image_folder, calib_file_path):
    image_paths = get_image_paths(image_folder)
    K = load_intrinsic_matrix(calib_file_path)
    
    trajectory = []
    R_f = np.eye(3)
    t_f = np.zeros((3, 1))

    prev_img = cv2.imread(image_paths[0], 0)
    kp1, des1 = extract_features(prev_img)

    for img_path in image_paths[1:]:
        curr_img = cv2.imread(img_path, 0)
        kp2, des2 = extract_features(curr_img)

        R, t = estimate_pose(kp1, kp2, des1, des2, K)

        t_f += R_f @ t
        R_f = R @ R_f

        trajectory.append((t_f[0, 0], t_f[1, 0], t_f[2, 0]))
        
        kp1, des1 = kp2, des2

    plot_trajectory(trajectory)
