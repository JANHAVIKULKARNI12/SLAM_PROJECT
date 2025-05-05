import os
import cv2
import numpy as np

def load_images(image_folder):
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ])
    images = []
    for f in image_files:
        path = os.path.join(image_folder, f)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append((f, image))
    return images

def extract_fast_features(image, threshold=25):
    fast = cv2.FastFeatureDetector_create(threshold)
    keypoints = fast.detect(image, None)
    return keypoints

def extract_orb_features(image, max_keypoints=500):
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def draw_and_save_matches(image1, kp1, image2, kp2, matches, out_path):
    match_img = cv2.drawMatches(image1, kp1, image2, kp2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, match_img)

def run():
    print("[Lab 3] Starting feature extraction and matching...")

    image_folder = "Data/2011_09_26-2/2011_09_26_drive_0001_sync/image_02/data"
    output_folder = "OUTPUT/Lab3_Output"
    os.makedirs(output_folder, exist_ok=True)

    images = load_images(image_folder)

    fast_threshold = 30
    orb_nfeatures = 300

    all_fast_kps = []
    all_orb_kps = []
    all_orb_desc = []

    for idx, (fname, img) in enumerate(images):
        fast_kps = extract_fast_features(img, threshold=fast_threshold)
        orb_kps, orb_desc = extract_orb_features(img, max_keypoints=orb_nfeatures)

        print(f"[{idx+1}] {fname} -> FAST: {len(fast_kps)}, ORB: {len(orb_kps)} keypoints")

        all_fast_kps.append(fast_kps)
        all_orb_kps.append(orb_kps)
        all_orb_desc.append(orb_desc)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    total_frames = len(images)

    # Match with steps: 1, 2, 4
    step_patterns = [1, 2, 4]

    for step in step_patterns:
        print(f"\n--- Matching with step: {step} ---")
        for i in range(0, total_frames - step):
            j = i + step
            if all_orb_desc[i] is not None and all_orb_desc[j] is not None:
                matches = bf.match(all_orb_desc[i], all_orb_desc[j])
                matches = sorted(matches, key=lambda x: x.distance)

                output_path = os.path.join(output_folder, f"match_{i+1}_{j+1}_step{step}.png")
                draw_and_save_matches(images[i][1], all_orb_kps[i], images[j][1], all_orb_kps[j], matches[:50], output_path)

                print(f"Saved match {i+1} - {j+1} (step {step}) -> {output_path}")
            else:
                print(f"Skipping match {i+1} - {j+1}: descriptors missing.")

if __name__ == "__main__":
    run()
