# Standard imports
import cv2
from PIL import Image
import numpy as np
from imutils import paths
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen



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



imagePathString = "./20210310_BSA&SERUM/1PM/H/screenshots/"
imagePaths=sorted(list(paths.list_images(imagePathString)))

print("small dot number: #######################")
for imagePath in imagePaths:
    # imageName = imagePath[-23:-4]
    # Read image
    # img = cv2.imread("./20210310_BSA&SERUM/1PM/C/screenshots/18.png")
    img = cv2.imread(imagePath)
    img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ############################ red mask ####################### -- step 1 ----------  not in use
    mask_r0 = cv2.inRange(hsv, redLower, redUpper)

    #################################### threshold minus green  -- step 2 ----------  not in use
    ret, th2 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)  # 70 threshold ###################################
    # cv2.imshow("th2",th2)
    hsv_th2 = cv2.cvtColor(th2, cv2.COLOR_BGR2HSV)
    mask_g1 = cv2.inRange(hsv_th2, greenLower, greenUpper)
    mask_g2 = np.bitwise_not(mask_g1)
    res_minus_green = cv2.bitwise_and(hsv_th2, hsv_th2, mask=mask_g2)
    gray = cv2.cvtColor(res_minus_green, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("th2-green",res_minus_green)
    # cv2.imshow("gray",gray)

    mask_3 = np.bitwise_or(gray, mask_r0)
    ret, mask_3 = cv2.threshold(mask_3, 50, 255, cv2.THRESH_BINARY)  # 10 threshold
    # cv2.imshow("colormask+thresholdcolor",mask_3)

    ##### RED + GREEN
    _, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(mask_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_small_dot_0 = 0
    n_big_dot_0 = 0

    for contour in contours_thresh_gray_im_with_keypoints0:
        if np.size(contour) < 9:
            n_small_dot_0 += 1
        else:
            n_big_dot_0 += 1
    print(n_small_dot_0)


########################################################################################################3

print("big dot number: #######################")
for imagePath in imagePaths:
    # imageName = imagePath[-23:-4]
    # Read image
    # img = cv2.imread("./20210310_BSA&SERUM/1PM/C/screenshots/18.png")
    img = cv2.imread(imagePath)
    img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ############################ red mask ####################### -- step 1 ----------  not in use
    mask_r0 = cv2.inRange(hsv, redLower, redUpper)

    #################################### threshold minus green  -- step 2 ----------  not in use
    ret, th2 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)  # 70 threshold ###################################
    # cv2.imshow("th2",th2)
    hsv_th2 = cv2.cvtColor(th2, cv2.COLOR_BGR2HSV)
    mask_g1 = cv2.inRange(hsv_th2, greenLower, greenUpper)
    mask_g2 = np.bitwise_not(mask_g1)
    res_minus_green = cv2.bitwise_and(hsv_th2, hsv_th2, mask=mask_g2)
    gray = cv2.cvtColor(res_minus_green, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("th2-green",res_minus_green)
    # cv2.imshow("gray",gray)

    mask_3 = np.bitwise_or(gray, mask_r0)
    ret, mask_3 = cv2.threshold(mask_3, 50, 255, cv2.THRESH_BINARY)  # 10 threshold
    # cv2.imshow("colormask+thresholdcolor",mask_3)

    ##### RED + GREEN
    _, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(mask_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_small_dot_0 = 0
    n_big_dot_0 = 0

    for contour in contours_thresh_gray_im_with_keypoints0:
        if np.size(contour) < 9:
            n_small_dot_0 += 1
        else:
            n_big_dot_0 += 1
    print(n_big_dot_0)



########################################################################################################3

print("dot area: #######################")
for imagePath in imagePaths:
    # imageName = imagePath[-23:-4]
    # Read image
    # img = cv2.imread("./20210310_BSA&SERUM/1PM/C/screenshots/18.png")
    img = cv2.imread(imagePath)
    img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ############################ red mask ####################### -- step 1 ----------  not in use
    mask_r0 = cv2.inRange(hsv, redLower, redUpper)

    #################################### threshold minus green  -- step 2 ----------  not in use
    ret, th2 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)  # 70 threshold ###################################
    # cv2.imshow("th2",th2)
    hsv_th2 = cv2.cvtColor(th2, cv2.COLOR_BGR2HSV)
    mask_g1 = cv2.inRange(hsv_th2, greenLower, greenUpper)
    mask_g2 = np.bitwise_not(mask_g1)
    res_minus_green = cv2.bitwise_and(hsv_th2, hsv_th2, mask=mask_g2)
    gray = cv2.cvtColor(res_minus_green, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("th2-green",res_minus_green)
    # cv2.imshow("gray",gray)

    mask_3 = np.bitwise_or(gray, mask_r0)
    ret, mask_3 = cv2.threshold(mask_3, 50, 255, cv2.THRESH_BINARY)  # 10 threshold
    # cv2.imshow("colormask+thresholdcolor",mask_3)

    ##### RED + GREEN
    _, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(mask_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_small_dot_0 = 0
    n_big_dot_0 = 0

    for contour in contours_thresh_gray_im_with_keypoints0:
        if np.size(contour) < 9:
            n_small_dot_0 += 1
        else:
            n_big_dot_0 += 1

    dot_area_0 = np.sum(mask_3) / 255
    print(dot_area_0)




