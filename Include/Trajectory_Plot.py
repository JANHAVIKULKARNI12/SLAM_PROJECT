import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_intrinsic_matrix(calib_file_path):
    with open(calib_file_path, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02:'):
                parts = line.strip().split()[1:]
                P = np.array([float(p) for p in parts]).reshape(3, 4)
                K = P[:, :3]
                return K
    raise ValueError("P_rect_02 not found in calibration file.")

def get_image_paths(folder):
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.endswith('.png') or f.endswith('.jpg')
    ])

def extract_keypoints_and_matches(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    return pts1, pts2

def compute_pose(K, pts1, pts2):
    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def compute_trajectory(image_paths, K, step):
    pose = np.eye(4)  # Start at identity matrix (0, 0, 0)
    trajectory = [pose[:3, 3:]]  # Initial position

    for i in range(0, len(image_paths) - step, step):
        print(f"Processing frame pair: {i} -> {i+step}")
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[i + step], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Skipping frames {i}-{i+step}")
            continue

        pts1, pts2 = extract_keypoints_and_matches(img1, img2)
        if len(pts1) < 8 or len(pts2) < 8:
            print(f"Not enough keypoints between frames {i}-{i+step}")
            continue

        R, t = compute_pose(K, pts1, pts2)

        # Build homogeneous transformation matrix (do not invert)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        # Accumulate motion
        pose = pose @ T
        trajectory.append(pose[:3, 3:])

    return np.hstack(trajectory)

def plot_trajectory(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points_3d[0], points_3d[1], points_3d[2], marker='o', markersize= 2, color ='darkblue', linewidth=1, label="Trajectory")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Driving Trajectory')
    ax.legend()
    plt.show()

def run():
    print("\nRunning Lab 6 - Trajectory Plot")

    image_folder = "Data/2011_09_26-2/2011_09_26_drive_0001_sync/image_02/data"
    calib_file_path = "Data/Calibration_Files/calib_cam_to_cam.txt"
    output_file = "OUTPUT/lab6_trajectory_results.txt"
    step = 1  # You can change this to 3 or 5

    K = load_intrinsic_matrix(calib_file_path)
    image_paths = get_image_paths(image_folder)

    trajectory_points = compute_trajectory(image_paths, K, step)
    plot_trajectory(trajectory_points)

    # Save to file
    with open(output_file, 'w') as f:
        f.write("Lab 6 - 3D Trajectory Results\n")
        f.write(f"Trajectory points:\n{trajectory_points}\n")

    print(f"\nTrajectory_results saved to: {output_file}")
