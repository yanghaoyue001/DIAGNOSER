# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 96, 40)
# greenLower = (50, 100, 100)
greenUpper = (70, 255, 255)

# Read image
img = cv2.imread('sample0.png')
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask = cv2.inRange(hsv, greenLower, greenUpper)
# mask = cv2.erode(mask, None, iterations=2)
# mask = cv2.dilate(mask, None, iterations=2)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(hsv, hsv, mask=mask)

# Show green masked result
cv2.imshow("orig", hsv)
cv2.imshow("green mask", res)
cv2.waitKey(0)