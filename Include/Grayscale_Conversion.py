import os
import cv2
import numpy as np

def run():
    print("[Lab 2] Converting images to grayscale...")

    # Relative paths (adjust based on your project structure)
    image_folder = "Data/2011_09_26-2/2011_09_26_drive_0001_sync/image_02/data"
    output_folder = "OUTPUT/Grayscale_Output"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files (assumed .png or .jpg), sorted by filename
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ])

    if not image_files:
        print("No image files found in:", image_folder)
        return

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_file}")
            continue

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Optional: print matrix values
        print(f"[Frame {idx + 1}: {image_file}] Grayscale image processed.")

        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, grayscale)

    print(f"✅ All grayscale images saved to: {output_folder}")

if __name__ == "__main__":
    run()
