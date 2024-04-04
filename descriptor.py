import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def generate_covariance_matrix(target_img_path: str, debug:bool) -> np.array:
    img = io.imread(target_img_path).astype("float64")

    featureVector = []            
    for a in range(img.shape[0]):
        for b in range(img.shape[1]):
            featureVector.append([b,a, img[a][b][0], img[a][b][1], img[a][b][2]])

    featureVector = np.array(featureVector)
    cov_matrix = np.cov(np.transpose(featureVector), bias=True)

    if debug:
        io.imshow(cov_matrix, cmap='grey')
        plt.title("Covariance Matrix")
        plt.show()

    return cov_matrix
            