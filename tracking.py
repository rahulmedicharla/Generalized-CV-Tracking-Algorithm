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
from scipy import ndimage, linalg

def covariance_tracking(source_file,covariance_matrix,dimension):
    inputImage = io.imread(source_file)
    plt.subplot(1,1,1)
    plt.imshow(inputImage)
    plt.show()
    resizedImage = ndimage.zoom(inputImage, (0.2, 0.2, 1))  # Resize to 20% of the original size
    plt.imshow(resizedImage)
    plt.show()
    a,b,c = resizedImage.shape

    featureList = []
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
    # print(featureList.shape) 
    featureListReshaped = []
    for matrix in featureList:
        reshapedMatrix = matrix.reshape(matrix.shape[0]*matrix.shape[1],(matrix.shape[2]))
        featureListReshaped.append(reshapedMatrix)
    # print(len(featureListReshaped))
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

def brightness_normalization(target: np.array, search: np.array):
    for i in range(0, 3):
        target[:,:,i] = target[:,:,i] - np.mean(target[:,:,i])
        search[:,:,i] = search[:,:,i] - np.mean(search[:,:,i])
    
    return target, search

def bn_color_based_tracking(target_img: np.array, dimensions: tuple, source_file:str, debug:bool):
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
    new_dimensions = dimensions
    scales = [.8,.9,1.0, 1.1,1.2]

    for scale in scales:
        for x in range(0, scaled_frame.shape[0] - dimensions[0]):
            for y in range(0, scaled_frame.shape[1] - dimensions[1]):
                if debug:
                    print("region num " + str(region_num))
                    region_num += 1

                region = scaled_frame[x:x+dimensions[0], y:y+dimensions[1]]
                region = ndimage.zoom(region, (scale, scale, 1))
                new_dimensions = (int(dimensions[0] * 1/scale), int(dimensions[1] * 1/scale))

                normalized_target, region = brightness_normalization(np.copy(target_img), np.copy(region))

                hist = create_color_hist(region)
                cov_matrix = create_cov_matrix(region)

                target_cov_matrix = create_cov_matrix(normalized_target)
                target_histogram = create_color_hist(normalized_target)

                gen_eigen_vals, _ = linalg.eigh(target_cov_matrix, cov_matrix)

                cov_similarity = 0
                for eig in gen_eigen_vals:
                    if eig != 0:
                        cov_similarity += np.log(eig) ** 2
                
                cov_similarity = np.sqrt(cov_similarity)

                similarity = np.sum(np.sqrt(target_histogram * hist))

                if similarity > initial_location[1] and cov_similarity < initial_cov_match:
                    xy_centers = (x + new_dimensions[0]//2,y + new_dimensions[1]//2)
                    initial_location = ([(xy_centers[0] - max(new_dimensions)//2, xy_centers[1] - max(new_dimensions)//2), (xy_centers[0] - max(new_dimensions)//2, xy_centers[1] + max(new_dimensions)//2), (xy_centers[0] + max(new_dimensions)//2, xy_centers[1] - max(new_dimensions)//2), (xy_centers[0] + max(new_dimensions)//2, xy_centers[1] + max(new_dimensions)//2)], similarity, hist)
                    initial_cov_match = cov_similarity
                    
    if debug:
        print("initial location: ", initial_location[0], initial_location[1])
        _, ax = plt.subplots()
        # io.imshow(frame)
        bbox = np.array(initial_location[0]).reshape(4,2)
        # plt.plot(bbox[:,1], bbox[:,0], 'ro')
        # plt.plot(xy_centers[1], xy_centers[0], 'bo')
        # plt.show()
        cv.rectangle(frame, (int(bbox[0][1]* 10 ), int(bbox[0][0] * 10)), (int(bbox[3][1] * 10), int(bbox[3][0]* 10)), (0,255,0), 2)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("Frame", frame)

        cv.waitKey(0)
        cv.destroyAllWindows()
            
    
    final_track_results = [(xy_centers, np.array(initial_location[0]).reshape(4,2))]
    dimensions = new_dimensions

    while ret:
        ret, frame = cap.read()

        if not ret:
            break
    
        for scale in scales:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            scaled_frame = scale_down_img(frame)

            last_xy, last_bbox = final_track_results[-1]
            current_match = (last_bbox, 0, 0)
            current_cov_match = 1000000000000
            
            neighbor_regions = circularNeighbors(scaled_frame, last_xy[0], last_xy[1], dimensions)
            for region, loc in neighbor_regions:
                region = ndimage.zoom(region, (scale, scale, 1))
                new_dimensions = (int(dimensions[0] * 1/scale), int(dimensions[1] * 1/scale))
                normalized_target, region = brightness_normalization(np.copy(target_img), np.copy(region))

                hist = create_color_hist(region)
                cov_matrix = create_cov_matrix(region)

                target_cov_matrix = create_cov_matrix(normalized_target)
                target_histogram = create_color_hist(normalized_target)

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
                    current_match = ([(current_xy[0] - max(new_dimensions)//2,current_xy[1] - max(new_dimensions)//2),(current_xy[0] - max(new_dimensions)//2, current_xy[1] + max(new_dimensions)//2), (current_xy[0] + max(new_dimensions)//2, current_xy[1] - max(new_dimensions)//2), (current_xy[0] + max(new_dimensions)//2, current_xy[1] + max(new_dimensions)//2)], similarity, hist)
                    current_cov_match = cov_similarity
                                    
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
    
def template_matching(img_file,video_file):

    def normalize(img):
        img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        return img_normalized.astype(np.uint8)

    def ncc(template, patch):
        template_mean = np.mean(template)
        patch_mean = np.mean(patch)
        template_sub_mean = template - template_mean
        patch_sub_mean = patch - patch_mean
        numerator = np.sum(template_sub_mean * patch_sub_mean)
        denominator = np.sqrt(np.sum(template_sub_mean ** 2) * np.sum(patch_sub_mean ** 2))
        if denominator == 0:
            return 0  
        ncc = numerator / denominator
        return ncc
    
    def template_matching_algorithm(frame, template_gray):
        h, w, ch = template_gray.shape
        best_score = -np.inf
        best_loc = (0, 0)
        threshold = 0.8
        for y in range(frame.shape[0] - h + 1):
            for x in range(frame.shape[1] - w + 1):
                patch = frame[y:y+h, x:x+w, :]
                score = ncc(template_gray, patch)
                if score > threshold and score > best_score:
                    best_score = score
                    best_loc = (x,y)

        return best_score, best_loc

    roi_template_rgba = cv.imread(img_file)
    # Convert BGRA template to RGB and rescale
    roi_template_rgb = cv.cvtColor(roi_template_rgba, cv.COLOR_BGR2RGB)
    # print(roi_template_rgb.shape)
    roi_template_rgb_rescaled = normalize(roi_template_rgb)
    # print(roi_template_rgb_rescaled.shape)
    roi_template_rgb_rescaled = ndimage.zoom(roi_template_rgb_rescaled, (0.1, 0.1, 1))
    # Load the video file
    cap = cv.VideoCapture(video_file)

    # Create a window to display the video with ROI
    cv.namedWindow('Video with ROI')
    cv.resizeWindow('Video with ROI', 640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB and rescale
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # print(frame_rgb.shape)
        frame_rgb_rescaled = normalize(frame_rgb)
        frame_rgb_rescaled = ndimage.zoom(frame_rgb_rescaled, (0.1, 0.1, 1))
        # Perform template matching
        best_score, best_loc = template_matching_algorithm(frame_rgb_rescaled, roi_template_rgb_rescaled)
        # print(best_score)
        locb = best_loc
        # print(locb)
        scale_factor = 10
        top_left = (locb[0]*scale_factor, locb[1] * scale_factor)  # (180, 540
        # Calculate the bottom-right corner
        bottom_right = (top_left[0]+roi_template_rgb.shape[1], top_left[1]+roi_template_rgb.shape[0])
        # bottom_right = (locb[0], locb[1])
        cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        # Display the frame with the ROI rectangle
        cv.imshow("Video with ROI", frame)
        # Check for the 'Esc' key to exit the loop
        key = cv.waitKey(15) & 0xFF
        if key == 27:  
            break

    # Release the video capture and close all windows
    cap.release()
    cv.destroyAllWindows()
