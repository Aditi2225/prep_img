import cv2
import numpy as np

def buildGaussianPyramid(img, levels):
    if levels < 1:
        print("Level should be greater than 1")
        return False

    currImg = img
    pyramid = []

    for l in range(levels):
        down = cv2.pyrDown(currImg)
        pyramid.append(down)
        currImg = down

    return np.array(pyramid)

def buildLaplacianPyramid(img, levels):
    print(img.shape)

    if levels < 1:
        print("Levels should be greater than 1")

    currImg = img
    pyramid = []

    for l in range(levels):
        down = cv2.pyrDown(currImg)
        h, w, c = currImg.shape
        up = cv2.pyrUp(down, dstsize=(w, h))
        lap = currImg - up
        pyramid.append(lap)
        currImg = down

    pyramid.append(currImg)
    return np.array(pyramid)
