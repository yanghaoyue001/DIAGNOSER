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



imagePathString = "./ELISA-04192021/control/photos/"
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

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask_g0 = cv2.inRange(hsv, greenLower, greenUpper)
    mask_r0 = cv2.inRange(hsv, redLower, redUpper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(hsv, hsv, mask=mask_r0)

    # cv2.imshow("orig", img)
    # cv2.imshow("red mask", mask_r0)

    ##### ONLY RED
    _, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(mask_r0, cv2.RETR_EXTERNAL,
                                                                     cv2.CHAIN_APPROX_SIMPLE)

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

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask_g0 = cv2.inRange(hsv, greenLower, greenUpper)
    mask_r0 = cv2.inRange(hsv, redLower, redUpper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(hsv, hsv, mask=mask_r0)

    # cv2.imshow("orig", img)
    # cv2.imshow("red mask", mask_r0)

    ##### ONLY RED
    _, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(mask_r0, cv2.RETR_EXTERNAL,
                                                                     cv2.CHAIN_APPROX_SIMPLE)

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

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask_g0 = cv2.inRange(hsv, greenLower, greenUpper)
    mask_r0 = cv2.inRange(hsv, redLower, redUpper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(hsv, hsv, mask=mask_r0)

    # cv2.imshow("orig", img)
    # cv2.imshow("red mask", mask_r0)

    ##### ONLY RED
    _, contours_thresh_gray_im_with_keypoints0, _ = cv2.findContours(mask_r0, cv2.RETR_EXTERNAL,
                                                                     cv2.CHAIN_APPROX_SIMPLE)

    n_small_dot_0 = 0
    n_big_dot_0 = 0

    for contour in contours_thresh_gray_im_with_keypoints0:
        if np.size(contour) < 9:
            n_small_dot_0 += 1
        else:
            n_big_dot_0 += 1

    dot_area_0 = np.sum(mask_r0) / 255
    print(dot_area_0)




