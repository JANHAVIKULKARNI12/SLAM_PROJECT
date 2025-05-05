import cv2
import numpy as np
import os

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
    orb = cv2.ORB_create(nfeatures=500)
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

def compute_essential_matrix_and_decompose(K, pts1, pts2):
    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def process_frame_pairs(image_paths, K, step):
    print(f"\n--- Frame Matching with Step {step} ---")
    for i in range(0, len(image_paths) - step, step):
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[i + step], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"Failed to load image pair: {i}, {i + step}")
            continue

        pts1, pts2 = extract_keypoints_and_matches(img1, img2)
        if len(pts1) < 8 or len(pts2) < 8:
            print(f"Not enough keypoints in pair: {i}, {i + step}")
            continue

        R, t = compute_essential_matrix_and_decompose(K, pts1, pts2)
        print(f"\nFrame {i+1} to Frame {i+step+1}:")
        print("Rotation Matrix R:")
        print(np.array2string(R, precision=4, suppress_small=True))
        print("Translation Vector t:")
        print(np.array2string(t, precision=4, suppress_small=True))

def run():
    print("[Lab 4] Computing Essential Matrix and Pose Estimation...")

    image_folder = "./Data/2011_09_26-2/2011_09_26_drive_0001_sync/image_02/data"
    calib_file_path = "./Data/Calibration_Files/calib_cam_to_cam.txt"

    K = load_intrinsic_matrix(calib_file_path)
    image_paths = get_image_paths(image_folder)

    for step in [1, 2, 4]:
        process_frame_pairs(image_paths, K, step=step)

    print("Essential Matrix estimation complete.")

if __name__ == "__main__":
    run()