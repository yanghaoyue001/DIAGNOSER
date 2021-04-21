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

############################ red mask ####################### -- step 1 ----------  not in use
# greenLower = (29, 96, 50)
greenLower = (1, 90, 50)
greenUpper = (255, 255, 255)
# red mask
# redLower = (0, 0, 130)
# redUpper = (255, 255, 255)
redLower = (0, 0, 120)
redUpper = (255, 255, 255)
# re-red mask
re_redLower = (0, 0, 0)
re_redUpper = (255, 255, 120)

# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask_g0 = cv2.inRange(hsv, greenLower, greenUpper)
mask_r0 = cv2.inRange(hsv, redLower, redUpper)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(hsv, hsv, mask=mask_r0)

cv2.imshow("orig", img)
cv2.imshow("red mask", mask_r0)



#################################### threshold minus green  -- step 2 ----------  not in use
ret,th2 = cv2.threshold(img,60,255,cv2.THRESH_BINARY) #70 threshold ###################################
# cv2.imshow("th2",th2)
hsv_th2 = cv2.cvtColor(th2, cv2.COLOR_BGR2HSV)
mask_g1 = cv2.inRange(hsv_th2, greenLower, greenUpper)
mask_g2 = np.bitwise_not(mask_g1)
res_minus_green = cv2.bitwise_and(hsv_th2, hsv_th2, mask=mask_g2)
gray = cv2.cvtColor(res_minus_green, cv2.COLOR_BGR2GRAY)
# cv2.imshow("th2-green",res_minus_green)
# cv2.imshow("gray",gray)

mask_3 = np.bitwise_or(gray, mask_r0)
ret,mask_3 = cv2.threshold(mask_3,50,255,cv2.THRESH_BINARY) #10 threshold
# cv2.imshow("colormask+thresholdcolor",mask_3)

##### RED + GREEN
# _, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(mask_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##### ONLY RED
_, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(mask_r0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

n_small_dot_0 = 0
n_big_dot_0 = 0

for contour in contours_thresh_gray_im_with_keypoints0:
    if np.size(contour) < 9:
        n_small_dot_0 += 1
    else:
        n_big_dot_0 += 1
print("small dot number 0", n_small_dot_0)
print("big dot number 0", n_big_dot_0)

dot_area_0 = np.sum(mask_3)/255
print("dot area 0",dot_area_0)


################################ small blob detection  -- step 3 ----------------   in use

# Enhance image
# gray_img = cv2.equalizeHist(gray_img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Keypoints1", gray_img )
yen_threshold = threshold_yen(gray_img)
gray_img = rescale_intensity(gray_img, (0, yen_threshold), (0, 255)).astype(np.uint8)

# cv2.imshow("increase brightness", gray_img )

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 180
params.maxThreshold = 255
params.blobColor = 255
# Filter by Area.
params.filterByArea = True
params.maxArea = 5
params.minArea = 0
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(gray_img)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints0 = cv2.drawKeypoints(img, keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints0 = img_np
for curkey in keypoints:
    x = int(curkey.pt[0])
    y = int(curkey.pt[1])
    size = int(curkey.size)
    cv2.circle(im_with_keypoints0,(x,y),size,(255,255,255),-1)
im_with_keypoints0 = cv2.cvtColor(im_with_keypoints0, cv2.COLOR_RGB2BGR)
gray_im_with_keypoints0= cv2.cvtColor(im_with_keypoints0, cv2.COLOR_BGR2GRAY)

# im_with_keypoints = cv2.drawKeypoints(np.zeros(np.shape(gray_img)).astype(np.uint8), keypoints, np.array([]), (0,0,255),flags=0)
# im_with_keypoints1 = cv2.drawKeypoints(np.zeros(np.shape(gray_img)).astype(np.uint8), keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# gray_im_with_keypoints0= cv2.cvtColor(im_with_keypoints0, cv2.COLOR_BGR2GRAY)
ret,thresh_gray_im_with_keypoints0 = cv2.threshold(gray_im_with_keypoints0, 150, 255, cv2.THRESH_BINARY)
# _, contours_keypoint0, _ = cv2.findContours(thresh_gray_im_with_keypoints0 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im_with_keypoints1, contours_keypoint0, -1, (255,255,255), thickness=-1)

# Show keypoints
cv2.imshow("original",img)
cv2.imshow("Keypoints0", im_with_keypoints0)
cv2.imshow("Keypoints1", thresh_gray_im_with_keypoints0)


_, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(thresh_gray_im_with_keypoints0 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

n_small_dot_1 = 0
n_big_dot_1 = 0

for contour in contours_thresh_gray_im_with_keypoints0:
    if np.size(contour) < 9:
        n_small_dot_1 += 1
    else:
        n_big_dot_1 += 1
print("small dot number 1:", n_small_dot_1)
print("big dot number 1:", n_big_dot_1)

dot_area1 = np.sum(thresh_gray_im_with_keypoints0)/255
print("dot area 1:",dot_area1)

cv2.waitKey(0)

