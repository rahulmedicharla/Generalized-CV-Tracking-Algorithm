import cv2
import numpy as np

roi_template = cv2.imread('ritzn1.png')  
roi_template_gray = cv2.cvtColor(roi_template, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture('ritzvideo1.mp4')  
roi_top_left = None
roi_bottom_right = None

cv2.namedWindow('Video with ROI', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video with ROI', 640, 480)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(frame_gray, roi_template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + roi_template_gray.shape[1], top_left[1] + roi_template_gray.shape[0])

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('Video with ROI', frame)

    key = cv2.waitKey(45) & 0xFF
    if key == 27:  
        break

cap.release()
cv2.destroyAllWindows()
