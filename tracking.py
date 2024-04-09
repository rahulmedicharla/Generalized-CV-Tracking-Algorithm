import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
import cv2 as cv
import numpy as np
import math
from os import listdir
from os.path import join, isfile
from skimage import morphology
from skimage import measure,color
from skimage import io, data
from numpy.linalg import eig
# from scipy import ndimage, misc, resize
from scipy.ndimage import median_filter
import matplotlib.patches as patches
from scipy.linalg import eigh

from scipy import ndimage

def scale_down_img(img: np.array):
    return ndimage.zoom(img, (0.1, 0.1, 1))

def circularNeighbors(img, x, y, dimensions):
    neighbor_regions = []

    for i in range(x-5, x+5):
        for j in range(y-5,y+5):
            region = img[i:i+dimensions[0], j:j+dimensions[1]]
            neighbor_regions.append((region, (i,j)))    
    return neighbor_regions

def covariance_tracking(source_file,covariance_matrix,dimension):
    inputImage = io.imread(source_file)
    plt.subplot(1,1,1)
    plt.imshow(inputImage)
    plt.show()
    resizedImage = ndimage.zoom(inputImage, (0.2, 0.2, 1))  # Resize to 20% of the original size
    plt.imshow(resizedImage)
    plt.show()
    # covariance_matrix = np.array([[13804., 0., -229.34716145, 19.77503884, -113.31250348],
    #                             [0., 13134., 274.36863082, 221.80274664, -172.53018028],
    #                             [-229.34716145, 274.36863082, 564.51330601, 551.61563186, 201.42609058],
    #                             [19.77503884, 221.80274664, 551.61563186, 1290.74264532, 723.45067926],
    #                             [-113.31250348, -172.53018028, 201.42609058, 723.45067926, 574.08273869]])
    # print(model_covariance.shape)
    print(covariance_matrix.shape)
    a,b,c = resizedImage.shape
    print(a)
    print(b)
    print(c)
    featureList = []
    print(dimension)
    h= dimension[0]//5
    w = dimension[1]//5
    for i in range(a-h):
        if (i) % 50 == 0:
            print(i)
        for j in range(b-w):
            
            window = np.zeros((h,w,5))
            for k in range(h):
                for l in range(w):
                    xCoordinate = j + l
                    yCoordinate = i + k
                    R = inputImage[yCoordinate][xCoordinate][0]
                    G = inputImage[yCoordinate][xCoordinate][1]
                    B = inputImage[yCoordinate][xCoordinate][2]
                    window[k][l] = xCoordinate, yCoordinate, R, G, B
            featureList.append(window)        
    featureListReshaped = []
    for matrix in featureList:
        reshapedMatrix = matrix.reshape(matrix.shape[0]*matrix.shape[1],(matrix.shape[2]))
        featureListReshaped.append(reshapedMatrix)
    print(len(featureListReshaped))
    candidateCovMatrix = []
    for i in range(len(featureListReshaped)):
        if i%4000 ==0:
            print(i)
        covMatrix = np.cov(featureListReshaped[i].transpose(),bias=True)
        candidateCovMatrix.append(covMatrix)
        
    distanceMetric = []
    alpha = 0
    for matrix in candidateCovMatrix:
        eigvals = eigh(covariance_matrix, matrix, eigvals_only=True)
        for values in eigvals:
            if (values != 0):
                alpha += (math.log(values))**2
        beta = math.sqrt(alpha)
        distanceMetric.append(beta)
        alpha=0
    valueOfMaximumSimilarity = min(distanceMetric)
    indexOfMaximumSimilarity = distanceMetric.index(valueOfMaximumSimilarity)
    coordinatesOfMaximumSimilarity = featureListReshaped[indexOfMaximumSimilarity][0][0:2]
    print(valueOfMaximumSimilarity)
    print(coordinatesOfMaximumSimilarity)

    fig,ax = plt.subplots()
    ax.imshow(inputImage)
    rect = patches.Rectangle((coordinatesOfMaximumSimilarity[1]*5,coordinatesOfMaximumSimilarity[0]*5),225,225,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()



def mean_shift_tracking(target_histogram: np.array, dimensions: tuple, source_file:str, debug:bool):
    cap = cv.VideoCapture(source_file)

    if not cap.isOpened():
        exit()
        return
    
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    scaled_frame = scale_down_img(frame)
    initial_location = ((0, 0), 0, 0)
    region_num = 0

    for x in range(0, scaled_frame.shape[0] - dimensions[0]):
        for y in range(0, scaled_frame.shape[1] - dimensions[1]):
            if debug:
                print("region num " + str(region_num))
                region_num += 1

            region = scaled_frame[x:x+dimensions[0], y:y+dimensions[1]]

            hist = np.zeros(shape=(16, 16, 16))
            for a in range(region.shape[0]):
                for b in range(region.shape[1]):
                    r_index = int(region[a,b,0] // (256 // 16))
                    g_index = int(region[a,b,1] // (256 // 16))
                    b_index = int(region[a,b,2] // (256 // 16))
                    
                    hist[r_index][g_index][b_index] += 1
            hist /= np.sum(hist)

            similarity = np.sum(np.sqrt(target_histogram * hist))
            if similarity > initial_location[1]:
                initial_location = ((x, y), similarity, hist)
    
    if debug:
        print("initial location: ", initial_location)
        _, ax = plt.subplots()
        io.imshow(scaled_frame)
        plt.plot(initial_location[0][1] + dimensions[1]//2, initial_location[0][0] + dimensions[0]//2, 'ro')
        plt.show()

    final_track_results = [initial_location]
    
    while ret:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        scaled_frame = scale_down_img(frame)

        last_match = final_track_results[-1]
        current_match = ((last_match[0][0], last_match[0][1]), 0, 0)
        
        neighbor_regions = circularNeighbors(scaled_frame, last_match[0][0], last_match[0][1], dimensions)
        for region, loc in neighbor_regions:
            hist = np.zeros(shape=(16, 16, 16))
            for a in range(region.shape[0]):
                for b in range(region.shape[1]):
                    r_index = int(region[a,b,0] // (256 // 16))
                    g_index = int(region[a,b,1] // (256 // 16))
                    b_index = int(region[a,b,2] // (256 // 16))
                    
                    hist[r_index][g_index][b_index] += 1
            hist /= np.sum(hist)

            similarity = np.sum(np.sqrt(target_histogram * hist))

            if similarity > current_match[1]:
                current_match = (loc, similarity, hist)
            
        if debug:
            print("best match in frame: ", current_match)
            _, ax = plt.subplots()
            io.imshow(scaled_frame)
            plt.plot(current_match[0][1] + dimensions[1]//2, current_match[0][0] + dimensions[0]//2, 'ro')
            plt.show()
        
        final_track_results.append((current_match))
    
    cap.release()
    cv.destroyAllWindows()
    
    return final_track_results