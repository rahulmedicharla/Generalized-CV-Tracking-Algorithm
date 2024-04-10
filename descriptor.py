import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage

from misc import smooth

def generate_covariance_matrix(target_img_path: str, debug:bool) -> np.array:
    img = io.imread(target_img_path).astype("float64")

    img = smooth(img)

    featureVector = []            
    for a in range(img.shape[0]):
        for b in range(img.shape[1]):
            featureVector.append([img[a][b][0], img[a][b][1], img[a][b][2]])

    featureVector = np.array(featureVector)
    cov_matrix = np.cov(np.transpose(featureVector), bias=True)

    if debug:
        io.imshow(cov_matrix, cmap='grey')
        plt.title("Covariance Matrix")
        plt.show()

    return cov_matrix, img.shape

def generate_color_histogram(target_img_path:str, debug:bool) -> tuple:
    img = io.imread(target_img_path)
    orig_shape = img.shape

    img = ndimage.zoom(img, (0.1, 0.1, 1))
    
    hist = np.zeros(shape=(16, 16, 16))
    featureVector = []

    h = np.sqrt(img.shape[0]**2 + img.shape[1]**2)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            r_index = int(img[x,y,0] // (256 // 16))
            g_index = int(img[x,y,1] // (256 // 16))
            b_index = int(img[x,y,2] // (256 // 16))

            r = 1 - (np.sqrt((x - img.shape[0]//2)**2 + (y - img.shape[1]//2)**2) / h)
            
            hist[r_index][g_index][b_index] += r

            featureVector.append([img[x,y,0], img[x,y,1], img[x,y,2]])

    hist /= np.sum(hist)

    featureVector = np.array(featureVector)
    cov_matrix = np.cov(np.transpose(featureVector), bias=True)


    return hist, cov_matrix, orig_shape, img.shape