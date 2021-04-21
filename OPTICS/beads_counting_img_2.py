# Standard imports
import cv2
import numpy as np

# Read image45
im_color = cv2.imread("sample1.png")
im = cv2.imread("sample1.png", cv2.IMREAD_GRAYSCALE)

img = im

# Enhance image
img = cv2.equalizeHist(im)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 50
params.maxThreshold = 255

params.blobColor = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 1


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("original",im_color)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
