import sys
from descriptor import generate_covariance_matrix, generate_color_histogram, bn_generate_color_histogram
from tracking import color_based_tracking, covariance_tracking,template_matching,  bn_color_based_tracking
import cv2 as cv
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def main(target_img: str, source_file: str, type: str, debug: bool):
    # go across different version types and run corresponding one
    if type == "V0":
        cov_matrix, dimensions = generate_covariance_matrix(target_img, debug)
        covariance_tracking(source_file, cov_matrix, dimensions)
    if type == "V1":
        template_matching(target_img,source_file)
    if type == "V2":
        color_histogram, cov_matrix, dimensions = generate_color_histogram(target_img, debug)
        track_results = color_based_tracking(color_histogram, cov_matrix, dimensions, source_file, debug)

        cap = cv.VideoCapture(source_file)
        if not cap.isOpened():
            exit()
            return
    
        ret, frame = cap.read()
        frame_num = 0
        while ret:  
            cv.rectangle(frame, (int(track_results[frame_num][0][1]), int(track_results[frame_num][0][0])), (int(track_results[frame_num][3][1]), int(track_results[frame_num][3][0])), (0,255,0), 2)

            cv.imshow("Frame", frame)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break
            frame_num += 1
            ret, frame = cap.read()
    if type == "V3":
        target_img, dimensions = bn_generate_color_histogram(target_img, debug)
        track_results = bn_color_based_tracking(target_img, dimensions, source_file, debug)

        cap = cv.VideoCapture(source_file)
        if not cap.isOpened():
            exit()
            return
    
        ret, frame = cap.read()
        frame_num = 0
        while ret:  
            cv.rectangle(frame, (int(track_results[frame_num][0][1]), int(track_results[frame_num][0][0])), (int(track_results[frame_num][3][1]), int(track_results[frame_num][3][0])), (0,255,0), 2)

            cv.imshow("Frame", frame)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break
            frame_num += 1
            ret, frame = cap.read()
    elif type == "D1":
        cap = cv.VideoCapture(source_file)
        if not cap.isOpened():
            exit()
            return
        
        ret, frame = cap.read()
        cv.imwrite(target_img, frame)
        cap.release()
        cv.destroyAllWindows() 
    elif type == "D2":
        img = cv.imread(target_img)
        img = ndimage.zoom(img, (0.8, 0.8, 1))
        cv.imwrite("data/targets/book_small.png", img)      

if __name__ == "__main__":
    args = sys.argv

    if "-target" not in args or "-source" not in args or "-type" not in args:
        print("Please run program like python main.py -target <target_img_path> -source <source_path> -type<algorithm version> --debug?")
    else:
        main(args[2], args[4], args[6], '--debug' in args)