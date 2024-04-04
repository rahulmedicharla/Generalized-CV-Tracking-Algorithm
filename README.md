# Generalized-CV-Tracking-Algorithm

## Vishnu Dasaka, Rahul Medicharla

This program is a generalized CV object tracking algorithm that attempts to be rotation, scale, brightness, and background invariant using non ML based computer vision techniques.

The program takes as input a target picture of the object to be tracked in a different setting and environment as the source video. The algorithm will then be able to track the object within the video disregarding the differences between the environments of the object.

## V1. 

This is a barebones version of the generalized algorithm. It is NOT rotation, scale, brightness, and background invariant and should be used with proper data to be successful.

V1 uses the covariance matrix of the target image to create a descriptor for the object and uses covariance tracking to match that matrix within the video and find the best resulting target.

The covariance matrix uses the x & y locations, and the RGB colors of the target image as its feature vector.

