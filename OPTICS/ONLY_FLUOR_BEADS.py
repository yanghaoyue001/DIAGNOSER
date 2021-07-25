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
img_file_path = "./MAG COVID EVAL/mag-05302021/fluor-bg/5-1/1.png"
img_raw = cv2.imread(img_file_path)
img_0 = img_raw[120:1030,620:1230,:]
img_np_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
cv2.imshow("orig", img_0)
cv2.imshow("gray orig", img_gray_0)

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
# img_gray_2 = img_gray_0.copy()
# yen_threshold_0 = threshold_yen(img_2[:,:,0])
# img_2[:,:,0] = rescale_intensity(img_2[:,:,0], (0, yen_threshold_0), (0, 255)).astype(np.uint8)
# yen_threshold_1 = threshold_yen(img_2[:,:,1])
# img_2[:,:,1] = rescale_intensity(img_2[:,:,1], (0, yen_threshold_1), (0, 255)).astype(np.uint8)
# yen_threshold_2 = threshold_yen(img_gray_2[:,:,2])
# img_gray_2[:,:,2] = rescale_intensity(img_2[:,:,2], (0, yen_threshold_2), (0, 255)).astype(np.uint8)
#
img_gray_2 = img_gray_0.copy()
img_gray_2 = np.where(img_gray_2 < 15, 15, img_gray_2)
yen_threshold_3 = threshold_yen(img_gray_2)
img_gray_2 = rescale_intensity(img_gray_2, (0, yen_threshold_3), (0, 255)).astype(np.uint8)
print("yen threshold =",yen_threshold_3)
cv2.imshow("yen pixels", img_gray_2)
# hist_gray = cv2.calcHist([gray_img_0],[0],None,[256],[0,256])
# plt.hist(gray_img_0.ravel(),256,[0,256])
# plt.show()

img_3 = img_0.copy()
s = (img_3.shape[0],img_3.shape[1],img_3.shape[2])
img_4 = np.zeros(s).astype(np.uint8)

# for i in range(0, img_0.shape[0]):
#     for j in range(0, img_0.shape[1]):
#         if img_gray_0[i, j] > 15:  # parameter 1
#             if img_2[i, j, 2] > 10 and img_2[i, j, 0] > 8:  # parameter 2 and 3
#                 if img_2[i, j, 2] / img_2[i, j, 1] > 0.55:  # parameter 4
#                     img_3[i, j, :] = img_2[i, j, :].copy()
#                 else:
#                     img_3[i, j, :] = [0, 0, 0]
#             else:
#                 img_3[i, j, :] = [0, 0, 0]
#         else:
#             img_3[i, j, :] = [0, 0, 0]

for i in range(0, img_0.shape[0]):
    for j in range(0, img_0.shape[1]):
        if img_gray_2[i, j] > 200 and img_gray_0[i,j]>20:  # parameter 1
            img_3[i, j, :] = img_2[i, j, :].copy()
        else:
            img_3[i, j, :] = [0, 0, 0]

cv2.imshow("bright pixels", img_3) # mag beads + signals + bright noises


##############################################- only fluor beads condition #############################################

# # # calculation ------------------------------------------------------------------------------------------------------
img_gray_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)

num_signals = np.count_nonzero(img_gray_3)
brightness_signals = np.sum(img_gray_3)

head, tail = os.path.split(img_file_path)
signal_filename = tail.split(".",1)[0] + "-signal.jpg"
cv2.imwrite(signal_filename,img_3)

print(num_signals, brightness_signals)
cv2.waitKey(0)
