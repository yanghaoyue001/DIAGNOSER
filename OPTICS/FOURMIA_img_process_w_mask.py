# Standard imports
import os
import cv2
from PIL import Image
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
import imutils


# # # # Read image ----------------------------------------------------------------------------------------------------
img_file_path = "./MAG COVID EVAL/mag-05312021/plate3/A-1-G/green/38.png"
img_raw = cv2.imread(img_file_path)
img_0 = img_raw[120:1030,620:1230,:]
img_np_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
cv2.imshow("orig", img_0)
cv2.imshow("gray orig", img_gray_0)

# # # # use gray ------------------------------------------------------------------------------------------------------
# gray_img_1 = gray_img_0.copy()
# hist_full_black = cv2.calcHist([gray_img_0],[0],None,[256],[0,256])
# peak_black = 0
# threshold_black = 0
# for i in range(2,50):
#     if hist_full_black[i,0] > 1000 and peak_black == 0:
#         peak_black = 1
#     elif hist_full_black[i,0] < 900 and peak_black == 1:
#         # print(i)
#         for j in range(i,50):
#             if (hist_full_black[j,0] >100):
#                 break
#             threshold_black = j
#         break
# print(threshold_black)
#
# gray_img_0 = np.where(gray_img_0 < threshold_black, 0, gray_img_0)

# plt.hist(gray_img_0.ravel(),256,[0,256])
# plt.show()
x=1

# # # # Analyze RGB ----------------------------------------------------------------------------------------------------
# cv2.imshow("orig BLUE", img_0[:,:,0])
# cv2.imshow("orig GREEN", img_0[:,:,1])
# cv2.imshow("orig RED", img_0[:,:,2])
# cv2.imshow("orig H", hsv_0[:,:,0])
# cv2.imshow("orig S", hsv_0[:,:,1])
# cv2.imshow("orig V", hsv_0[:,:,2])
# img_1 = img_0.copy()
# img_1[:,:,2] = np.where(img_1[:,:,2] < 20, 0, img_1[:,:,2]) #red mask
# img_1[:,:,0] = np.where(img_1[:,:,0] < 20, 0, img_1[:,:,0]) #blue mask
# img_1[:,:,1] = np.where(img_1[:,:,1] < 40, 0, img_1[:,:,1]) #green mask
# cv2.imshow("mask", img_1)
# img_2 = img_0.copy()
# yen_threshold_0 = threshold_yen(img_2[:,:,0])
# img_2[:,:,0] = rescale_intensity(img_2[:,:,0], (0, yen_threshold_0), (0, 255)).astype(np.uint8)
# yen_threshold_1 = threshold_yen(img_2[:,:,1])
# img_2[:,:,1] = rescale_intensity(img_2[:,:,1], (0, yen_threshold_1), (0, 255)).astype(np.uint8)
# yen_threshold_2 = threshold_yen(img_2[:,:,2])
# img_2[:,:,2] = rescale_intensity(img_2[:,:,2], (0, yen_threshold_2), (0, 255)).astype(np.uint8)
# hist_red = cv2.calcHist([img_2[:,:,2]],[0],None,[256],[0,256])
# plt.hist(img_2.ravel(),256,[0,256])
# plt.show()
# cv2.imshow("rescale BGR img_2", img_2[:,:,2])

###### frequency analysis of colors, find the cluster of bright area, mag beads area, greeen area and dark area)

# iris = load_iris()
# X, y = iris.data, iris.target
img_2 = img_0.copy()
# yen_threshold_0 = threshold_yen(img_2[:,:,0])
# img_2[:,:,0] = rescale_intensity(img_2[:,:,0], (0, yen_threshold_0), (0, 255)).astype(np.uint8)
# yen_threshold_1 = threshold_yen(img_2[:,:,1])
# img_2[:,:,1] = rescale_intensity(img_2[:,:,1], (0, yen_threshold_1), (0, 255)).astype(np.uint8)
# yen_threshold_2 = threshold_yen(img_2[:,:,2])
# img_2[:,:,2] = rescale_intensity(img_2[:,:,2], (0, yen_threshold_2), (0, 255)).astype(np.uint8)
#
# img_gray_2 = img_gray_0.copy()
# yen_threshold_3 = threshold_yen(img_gray_2)
# img_gray_2 = rescale_intensity(img_gray_2, (0, yen_threshold_3), (0, 255)).astype(np.uint8)

# hist_gray = cv2.calcHist([gray_img_0],[0],None,[256],[0,256])
# plt.hist(gray_img_0.ravel(),256,[0,256])
# plt.show()

img_3 = img_0.copy()
s = (img_3.shape[0],img_3.shape[1],img_3.shape[2])
img_4 = np.zeros(s).astype(np.uint8)
img_4_2 = img_4.copy()
# for i in range(0,img_0.shape[0]):
#     for j in range(0,img_0.shape[1]):
#         if img_gray_0[i,j] > 15:
#             if img_2[i,j,2] > 20 and img_2[i,j,0] >15:
#                 if img_2[i,j,2]/img_2[i,j,1] >0.5:
#                     img_3[i,j,:]= img_2[i,j,:].copy()
#                 else:
#                     img_3[i, j, :] =[0,0,0]
#             else:
#                 img_3[i, j, :] = [0, 0, 0]
#         else:
#             img_3[i, j, :] = [0, 0, 0]
#
#         if img_gray_0[i,j] > 50:
#             if img_2[i, j, 2] / img_2[i, j, 1] > 0.5:
#                 img_4[i, j, :] = img_2[i, j, :].copy()

for i in range(0, img_0.shape[0]):
    for j in range(0, img_0.shape[1]):
        if img_gray_0[i, j] > 15:  # parameter 1
            if img_2[i, j, 2] > 10 and img_2[i, j, 0] > 8:  # parameter 2 and 3
                if img_2[i, j, 2] / img_2[i, j, 1] > 0.55:  # parameter 4
                    img_3[i, j, :] = img_2[i, j, :].copy()
                else:
                    img_3[i, j, :] = [0, 0, 0]
            else:
                img_3[i, j, :] = [0, 0, 0]
        else:
            img_3[i, j, :] = [0, 0, 0]


        if img_gray_0[i, j] > 50 and img_2[i, j, 2] / img_2[i, j, 1] > 0.6 and img_2[i, j, 2] / img_2[i, j, 1] < 0.8:  # parameter 5 6 green
            img_4[i, j, :] = img_2[i, j, :].copy()
        elif img_gray_0[i, j] > 40 and img_2[i, j, 2] / img_2[i, j, 1] > 0.7 and img_2[i, j, 2] / img_2[i, j, 1] < 0.8:  # parameter 7 8 green
            img_4[i, j, :] = img_2[i, j, :].copy()

        if img_gray_0[i, j] > 50 and img_gray_0[i, j] < 150 and img_2[i, j, 2] / img_2[i, j, 1] > 0.8:  # parameter 5 6 red
            img_4_2[i, j, :] = img_2[i, j, :].copy()
        elif img_gray_0[i, j] > 40 and img_gray_0[i,j] < 150 and img_2[i, j, 2] / img_2[i, j, 1] > 0.85:  # parameter 7 8 red
            img_4_2[i, j, :] = img_2[i, j, :].copy()

        # if img_gray_0[i, j] > 50 and img_2[i, j, 2] / img_2[i, j, 1] > 0.65 and img_2[i, j, 0] / img_2[i, j, 2] < .75:  # parameter 5 6 7; <0.75 red
        #     img_4[i, j, :] = img_2[i, j, :].copy()
        # elif img_gray_0[i, j] > 40 and img_2[i, j, 2] / img_2[i, j, 1] > 0.75 and img_2[i, j, 0] / img_2[i, j, 2] < .75:  # parameter 8 9 10; <0.75 red
        #     img_4[i, j, :] = img_2[i, j, :].copy()

        # if img_gray_0[i, j] > 50 and img_2[i, j, 2] / img_2[i, j, 1] > 0.4 and img_2[i, j, 0] / img_2[i, j, 2] > .75:  # parameter 5 6 7; >0.75 green
        #     img_4[i, j, :] = img_2[i, j, :].copy()
        # elif img_gray_0[i, j] > 40 and img_2[i, j, 2] / img_2[i, j, 1] > 0.5 and img_2[i, j, 0] / img_2[i, j, 2] > .75:  # parameter 8 9 10; >0.75 green
        #     img_4[i, j, :] = img_2[i, j, :].copy()



cv2.imshow("bright pixels", img_3) # mag beads + signals + bright noises
cv2.imshow("only sharp", img_4) # signals + bright noises

img_5 = img_3 - img_4 # only mag beads
img_gray_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2GRAY)

# # # remove large area sharp area by opening --------------------------------------------------------------------------
img_gray_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img_gray_4,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 2)
img_gray_4_denoise = img_gray_4 - erosion
# cv2.imshow("opening", erosion)
cv2.imshow("denoise green", img_gray_4_denoise) # only signals

img_gray_4_2 = cv2.cvtColor(img_4_2, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img_gray_4_2,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 2)
img_gray_4_2_denoise = img_gray_4_2 - erosion
# cv2.imshow("opening", erosion)
cv2.imshow("denoise red", img_gray_4_2_denoise) # only signals


### red mask
redLower = (0, 0, 80)
redUpper = (255, 255, 255)
hsv = cv2.cvtColor(img_4_2, cv2.COLOR_BGR2HSV)
mask_r0 = cv2.inRange(hsv, redLower, redUpper)
img_6_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_r0)
img_6 = cv2.cvtColor(img_6_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("red", img_6) # only signals



# # # calculation ------------------------------------------------------------------------------------------------------
num_mag_beads = np.count_nonzero(img_gray_5)
num_signals_green = np.count_nonzero(img_gray_4_denoise)
brightness_signals_green = np.sum(img_gray_4_denoise)
num_signals_red = np.count_nonzero(img_gray_4_2_denoise)
brightness_signals_red = np.sum(img_gray_4_2_denoise)

head, tail = os.path.split(img_file_path)
magbeads_filename = tail.split(".",1)[0] + "-mag.jpg"
cv2.imwrite(magbeads_filename,img_gray_5)
signal_filename_green = tail.split(".",1)[0] + "-signal-g.jpg"
cv2.imwrite(signal_filename_green,img_gray_4_denoise)
signal_filename_red = tail.split(".",1)[0] + "-signal-r.jpg"
cv2.imwrite(signal_filename_red,img_gray_4_denoise)


print(num_mag_beads, num_signals_green, brightness_signals_green, num_signals_red, brightness_signals_red)
cv2.waitKey(0)
x=1
