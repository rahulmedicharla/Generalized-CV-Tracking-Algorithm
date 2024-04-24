# Generalized-CV-Tracking-Algorithm

## Vishnu Dasaka, Rahul Medicharla

This program is a generalized CV object tracking algorithm that attempts to be rotation, scale, brightness, and background invariant using non ML based computer vision techniques.

The program takes as input a target picture of the object to be tracked in a different setting and environment as the source video. The algorithm will then be able to track the object within the video disregarding the differences between the environments of the object.

## V0. 

This is a barebones version of the generalized algorithm. It is NOT rotation, scale, brightness, and background invariant and should be used with proper data to be successful.

V1 uses the covariance matrix of the target image to create a descriptor for the object and uses covariance tracking to match that matrix within an image and find the best resulting target.

V1 also uses a image pyramid sturcture to reduce the image size to decrease computation time while minimizing prediction error.

The covariance matrix uses the x & y locations, and the RGB colors of the target image as its feature vector.

## V1.

This is the improved version that is brightness invariant by using the NCC normalized-cross-correlation algorithm. It is improved to work across a source video and template image. 

The program works by normalizing the template image and the patches that are searched per frame to remove brightness differences before calculating the similarity between patches. 

The algorithm runs across every patch of every frame to find the the best match, and uses the image pyramid ideas to reduce computation times. 

## V2.

For V2, we wanted to make the algorithm rotation invariant. 

We do so by taking concepts from the mean shift tracking algorithm and using a 3D color histogram in combination with a covariance matrix to find the best match. Mean shift tracking has good rotation invariant properties, so it was a good starting point. We utilized the RGB channels for the color histogram, and the x,y & RGB channels for the covariance matrix tracking. 

The first thing we did was for the first frame, iterate over every patch, and find the best template match as the one that increases the Bhattacharyya coefficiant (similarity metric) and reduces the covariance distance (distance metric). By doing so, we were able to find the best position for the first frame.

Then for each successive frame, we used the mean-shift ideas to search around a neighborhood of pixels, and update the next best location using the same approach. 

As a result we were able to accomplish a rotation invariant tracking algorithm.

## V3.

For our last version, we combined all the previous versions to create a scale, brightness, and rotation invariant tracking algorithm.

To begin, we used V2 as a starting point. To make V2 brightness invariant, we used ideas from NCC, specifically its normalization properties, and made sure to normalize each patch before we calculated the similarity measurements. 

To make V2 scale invariant, when we ran our algorithm, we created a scale pyramid of different sizes and chose the one that best matched, effectively becoming scale invariant.

As a result, we were able to build a scale, rotation, and brightness invariant tracking algorithm.


