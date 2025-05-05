# main.py

from Include import Kalman_Filter
from Include import Grayscale_Conversion
from Include import Feature_Extraction_and_Matching
from Include import Essential_Matrix_Decomposition
from Include import Trajectory_Plot
from Include import FINAL_PROJECT_SLAM as FINAL_PROJECT_SLAM
def main():
    print("Running - Kalman Filter")
    Kalman_Filter.run()

    print("Running - Grayscale Conversion")
    Grayscale_Conversion.run()

    print("Running  - Feature Extraction and Matching")
    Feature_Extraction_and_Matching.run()

    print("Running - Essential Matrix Decomposition")
    Essential_Matrix_Decomposition.run()

    print("Running - Trajectory Plot")
    Trajectory_Plot.run()

    print("Running Lab 6 Trajectory Plotting...")
    FINAL_PROJECT_SLAM.run()
if __name__ == "__main__":
    main()
