import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

SIGMA = 1

def smooth(img: np.array) -> np.array:
    return cv2.GaussianBlur(img, (2 * math.ceil(3 * SIGMA) + 1, 2 * math.ceil(3 * SIGMA) + 1), SIGMA)