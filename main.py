import sys
from descriptor import generate_covariance_matrix, generate_color_histogram
from tracking import mean_shift_tracking
import cv2 as cv
from skimage import io
import matplotlib.pyplot as plt

def main(target_img: str, source_file: str, type: str, debug: bool):
    # go across different version types and run corresponding one
    if type == "V1":
        cov_matrix, dimensions = generate_covariance_matrix(target_img, debug)
        # track_results = covariance_tracking(source_file, cov_matrix, dimensions, debug)
    if type == "V2":
        color_histogram, dimensions = generate_color_histogram(target_img, debug)
        track_results = mean_shift_tracking(color_histogram, dimensions, source_file, debug)
        
if __name__ == "__main__":
    args = sys.argv

    if "-target" not in args or "-source" not in args or "-type" not in args:
        print("Please run program like python main.py -target <target_img_path> -source <source_video_path> -type<algorithm version> --debug?")
    else:
        main(args[2], args[4], args[6], '--debug' in args)