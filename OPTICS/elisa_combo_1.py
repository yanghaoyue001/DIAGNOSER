# Standard imports
import cv2
from PIL import Image
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen

# Read image
img = cv2.imread("./ELISA-04192021/control/photos/3.png")
img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

################################ small blob detection  -- step 3 ----------------   in use

# Enhance image
# gray_img = cv2.equalizeHist(gray_img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Keypoints1", gray_img )
yen_threshold = threshold_yen(gray_img)
gray_img = rescale_intensity(gray_img, (0, yen_threshold), (0, 255)).astype(np.uint8)

cv2.imshow("orig", img )
cv2.imshow("increase brightness", gray_img )

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 120
params.maxThreshold = 255
params.blobColor = 0
# Filter by Area.
params.filterByArea = True
params.maxArea = 200
params.minArea = 0
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
#
# Detect blobs.
keypoints = detector.detect(gray_img)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints0 = cv2.drawKeypoints(img, keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints0 = img_np

# draw circle on each keypoint
for curkey in keypoints:
    x = int(curkey.pt[0])
    y = int(curkey.pt[1])
    size = int(curkey.size)
    cv2.circle(im_with_keypoints0,(x,y),size,(255,255,255),-1)

# find color range of dots
img_dot_count = hsv
H = []
S = []
V = []
for curkey in keypoints:
    x = int(curkey.pt[0])
    y = int(curkey.pt[1])
    H.append(img_dot_count[y, x, 0])
    S.append(img_dot_count[y, x, 1])
    V.append(img_dot_count[y, x, 2])

mask_dot_low = np.array([min(H),min(S),min(V)])
mask_dot_high = np.array([max(H),max(S),max(V)])

mask_dot = cv2.inRange(img_dot_count, mask_dot_low, mask_dot_high)

# Bitwise-AND mask and original image
res_dot = cv2.bitwise_and(img, img , mask=mask_dot)

cv2.imshow("orig", img)
cv2.imshow("mask", mask_dot)
cv2.imshow("res",res_dot)


# im_with_keypoints0 = cv2.cvtColor(im_with_keypoints0, cv2.COLOR_RGB2BGR)
# gray_im_with_keypoints0= cv2.cvtColor(im_with_keypoints0, cv2.COLOR_BGR2GRAY)
# #
# # im_with_keypoints = cv2.drawKeypoints(np.zeros(np.shape(gray_img)).astype(np.uint8), keypoints, np.array([]), (0,0,255),flags=0)
# # im_with_keypoints1 = cv2.drawKeypoints(np.zeros(np.shape(gray_img)).astype(np.uint8), keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # gray_im_with_keypoints0= cv2.cvtColor(im_with_keypoints0, cv2.COLOR_BGR2GRAY)
# ret,thresh_gray_im_with_keypoints0 = cv2.threshold(gray_im_with_keypoints0, 150, 255, cv2.THRESH_BINARY)
# # _, contours_keypoint0, _ = cv2.findContours(thresh_gray_im_with_keypoints0 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(im_with_keypoints1, contours_keypoint0, -1, (255,255,255), thickness=-1)
#
# # Show keypoints
# cv2.imshow("original",img)
# cv2.imshow("Keypoints0", im_with_keypoints0)
# cv2.imshow("Keypoints1", thresh_gray_im_with_keypoints0)

#
# _, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(thresh_gray_im_with_keypoints0 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# n_small_dot_1 = 0
# n_big_dot_1 = 0
#
# for contour in contours_thresh_gray_im_with_keypoints0:
#     if np.size(contour) < 9:
#         n_small_dot_1 += 1
#     else:
#         n_big_dot_1 += 1
# print("small dot number 1:", n_small_dot_1)
# print("big dot number 1:", n_big_dot_1)
#
# dot_area1 = np.sum(thresh_gray_im_with_keypoints0)/255
# print("dot area 1:",dot_area1)

cv2.waitKey(0)

