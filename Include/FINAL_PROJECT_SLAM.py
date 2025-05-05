import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_intrinsic_matrix(calib_file_path):
    """Loads the intrinsic camera matrix from the calibration file."""
    with open(calib_file_path, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02:'):
                parts = line.strip().split()[1:]
                P = np.array([float(p) for p in parts]).reshape(3, 4)
                K = P[:, :3]
                return K
    raise ValueError("P_rect_02 not found in calibration file.")

def get_image_paths(folder):
    """Returns sorted image paths from the specified folder."""
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.endswith('.png') or f.endswith('.jpg')
    ])

def extract_keypoints_and_matches(img1, img2):
    """Extracts keypoints and matches between two images using ORB."""
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
    """Computes rotation and translation matrices using essential matrix."""
    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def compute_trajectory(image_paths, K, step):
    """Computes the vehicle trajectory using successive frames."""
    trajectory = [np.array([[0], [0], [0]])]
    pose = np.eye(4)

    for i in range(0, len(image_paths) - step, step):
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[i + step], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Skipping frames {i}-{i+step}")
            continue

        pts1, pts2 = extract_keypoints_and_matches(img1, img2)
        if len(pts1) < 8:
            print(f"Not enough keypoints between frames {i}-{i+step}")
            continue

        R, t = compute_pose(K, pts1, pts2)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t

        pose = pose @ np.linalg.inv(T)
        trajectory.append(pose[:3, 3:])

    return np.hstack(trajectory)

def plot_trajectory(points_3d):
    """Plots the 3D driving trajectory."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points_3d[0], points_3d[1], points_3d[2], marker='o', markersize = 2, color = 'darkblue',  linewidth=1, label="Trajectory")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Driving Trajectory')
    ax.legend()
    plt.show()

def save_trajectory_to_file(points_3d, filename):
    """Saves the computed 3D trajectory to a file."""
    with open(filename, 'w') as f:
        f.write("Trajectory points:\n")
        f.write(str(points_3d))

def run():
    image_folder = "Data/2011_09_26-2/2011_09_26_drive_0001_sync/image_02/data"
    calib_file_path = "Data/Calibration_Files/calib_cam_to_cam.txt"
    output_file = "OUTPUT/lab6_trajectory_results.txt"
    step = 1   # or 3 or 5 for keyframe skipping

    print("Running FINAL_PROJECT_SLAM...")

    K = load_intrinsic_matrix(calib_file_path)
    image_paths = get_image_paths(image_folder)

    trajectory_points = compute_trajectory(image_paths, K, step)

    print("Trajectory points:")
    print(trajectory_points)

    save_trajectory_to_file(trajectory_points, output_file)
    plot_trajectory(trajectory_points)

if __name__ == "__main__":
    run()
