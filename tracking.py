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
from scipy.ndimage import median_filter, rotate
import matplotlib.patches as patches
from scipy.linalg import eigh

import math
import scipy
from scipy import ndimage, linalg
from misc import smooth

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


def scale_down_img(img: np.array):
    return ndimage.zoom(img, (0.1, 0.1, 1))

def circularNeighbors(img, x, y, dimensions):
    neighbor_regions = []

    for i in range(max(x-7, dimensions[0]//2), min(x+7, img.shape[0] - dimensions[0]//2)):
        for j in range(max(y-7, dimensions[1]//2),min(y+7, img.shape[1] - dimensions[1]//2)):
            region = img[i-dimensions[0]//2:i+dimensions[0]//2, j-dimensions[1]//2:j+dimensions[1]//2]
            neighbor_regions.append((region, (i,j)))    
    return neighbor_regions

def create_cov_matrix(region):
    feature_vector = []
    for a in range(region.shape[0]):
        for b in range(region.shape[1]):
            feature_vector.append([a,b,region[a,b,0], region[a,b,1], region[a,b,2]])
    
    feature_vector = np.array(feature_vector)
    cov_matrix = np.cov(np.transpose(feature_vector), bias=True)
    return cov_matrix

def create_color_hist(region):
    hist = np.zeros(shape=(16, 16, 16))
    h = np.sqrt(region.shape[0]**2 + region.shape[1]**2)
    for a in range(region.shape[0]):
        for b in range(region.shape[1]):
            r_index = int(region[a,b,0] // (256 // 16))
            g_index = int(region[a,b,1] // (256 // 16))
            b_index = int(region[a,b,2] // (256 // 16))

            r = 1 - (np.sqrt((a - region.shape[0]//2)**2 + (b - region.shape[1]//2)**2) / h)
            
            hist[r_index][g_index][b_index] += r
    
    if np.sum(hist) > 0:
        hist /= np.sum(hist)
    return hist

def color_based_tracking(target_histogram: np.array, target_cov_matrix:np.array, dimensions: tuple, source_file:str, debug:bool):
    cap = cv.VideoCapture(source_file)

    if not cap.isOpened():
        exit()
        return
    
    ret, frame = cap.read()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    scaled_frame = scale_down_img(frame)
    # bbox locations, similarity value, hist
    initial_location = ([(0, 0), (0,0), (0,0), (0,0)], 0, 0)
    xy_centers = (0,0)
    region_num = 0
    initial_cov_match = 1000000000000

    for x in range(0, scaled_frame.shape[0] - dimensions[0]):
        for y in range(0, scaled_frame.shape[1] - dimensions[1]):
            if debug:
                print("region num " + str(region_num))
                region_num += 1

            region = scaled_frame[x:x+dimensions[0], y:y+dimensions[1]]

            hist = create_color_hist(region)
            cov_matrix = create_cov_matrix(region)

            gen_eigen_vals, _ = linalg.eigh(target_cov_matrix, cov_matrix)

            cov_similarity = 0
            for eig in gen_eigen_vals:
                if eig != 0:
                    cov_similarity += np.log(eig) ** 2
            
            cov_similarity = np.sqrt(cov_similarity)

            similarity = np.sum(np.sqrt(target_histogram * hist))

            if similarity > initial_location[1] and cov_similarity < initial_cov_match:
                xy_centers = (x + dimensions[0]//2,y + dimensions[1]//2)
                initial_location = ([(xy_centers[0] - max(dimensions)//2, xy_centers[1] - max(dimensions)//2), (xy_centers[0] - max(dimensions)//2, xy_centers[1] + max(dimensions)//2), (xy_centers[0] + max(dimensions)//2, xy_centers[1] - max(dimensions)//2), (xy_centers[0] + max(dimensions)//2, xy_centers[1] + max(dimensions)//2)], similarity, hist)
                initial_cov_match = cov_similarity
                
    if debug:
        print("initial location: ", initial_location[0], initial_location[1])
        _, ax = plt.subplots()
        io.imshow(scaled_frame)
        bbox = np.array(initial_location[0]).reshape(4,2)
        plt.plot(bbox[:,1], bbox[:,0], 'ro')
        plt.plot(xy_centers[1], xy_centers[0], 'bo')
        plt.show()
    
    final_track_results = [(xy_centers, np.array(initial_location[0]).reshape(4,2))]

    while ret:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        scaled_frame = scale_down_img(frame)

        last_xy, last_bbox = final_track_results[-1]
        current_match = (last_bbox, 0, 0)
        current_cov_match = 1000000000000
        
        neighbor_regions = circularNeighbors(scaled_frame, last_xy[0], last_xy[1], dimensions)
        for region, loc in neighbor_regions:
            hist = create_color_hist(region)
            cov_matrix = create_cov_matrix(region)

            similarity = np.sum(np.sqrt(target_histogram * hist))

            gen_eigen_vals, _ = linalg.eigh(target_cov_matrix, cov_matrix)

            cov_similarity = 0
            for eig in gen_eigen_vals:
                if eig != 0:
                    cov_similarity += np.log(eig) ** 2
            
            cov_similarity = np.sqrt(cov_similarity)

            if similarity > current_match[1] and cov_similarity < current_cov_match:
                similarity = similarity * (1 - cov_similarity)
                current_xy = (int(np.mean([last_xy[0], loc[0]])), int(np.mean([last_xy[1], loc[1]])))
                current_match = ([(current_xy[0] - max(dimensions)//2,current_xy[1] - max(dimensions)//2),(current_xy[0] - max(dimensions)//2, current_xy[1] + max(dimensions)//2), (current_xy[0] + max(dimensions)//2, current_xy[1] - max(dimensions)//2), (current_xy[0] + max(dimensions)//2, current_xy[1] + max(dimensions)//2)], similarity, hist)
                current_cov_match = cov_similarity
        
        # feature_points = get_corners(np.copy(scaled_frame), current_xy, dimensions)

        # rotated_regions = get_rotated_regions(scaled_frame, current_xy[0], current_xy[1], dimensions, get_cov_matrix=True)
        # for region in rotated_regions:
        #     similarity = np.sum(np.sqrt(target_histogram * region[0]))
        #     gen_eigen_vals, _ = linalg.eigh(target_cov_matrix, region[3])

        #     cov_similarity = 0
        #     for eig in gen_eigen_vals:
        #         if eig != 0:
        #             cov_similarity += np.log(eig) ** 2
            
        #     cov_similarity = np.sqrt(cov_similarity)

        #     if similarity > current_match[1] and cov_similarity < current_cov_match:
        #         current_match = (region[1], similarity, region[0])

                                   
        if debug:
            print("best match in frame: ", current_match[0], current_match[1])
            _, ax = plt.subplots()
            io.imshow(scaled_frame)
            plt.plot(current_xy[1], current_xy[0], 'bo')
            plt.plot(np.array(current_match[0]).reshape(4,2)[:,1], np.array(current_match[0]).reshape(4,2)[:,0], 'ro')
            plt.show()

        final_track_results.append((current_xy, np.array(current_match[0]).reshape(4,2)))
    
    for i in range(len(final_track_results)):
        final_track_results[i] = final_track_results[i][1] * 10
        
    cap.release()
    cv.destroyAllWindows()
    
    return final_track_results