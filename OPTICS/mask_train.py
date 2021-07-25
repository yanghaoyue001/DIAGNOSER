# Standard imports
import cv2
from PIL import Image
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from matplotlib import pyplot as plt

# Read image
# img 0PM
img = cv2.imread("./MAG-04262021/SETTING 1/0PM.png")
img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# define range of blue color in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("gray_img", gray_img )
cv2.imshow('mask',mask)

blue_low   = np.uint8([[[0,0,0 ]]])
blue_high  = np.uint8([[[130,255,255 ]]])
hsv_blue_high = cv2.cvtColor(blue_high,cv2.COLOR_BGR2HSV)[0,0,:]
hsv_blue_low = cv2.cvtColor(blue_low,cv2.COLOR_BGR2HSV)[0,0,:]
mask_blue = cv2.inRange(hsv, hsv_blue_low , hsv_blue_high )
res_all = cv2.bitwise_and(img,img, mask= mask_blue)
cv2.imshow('mask blue',mask_blue)

cv2.waitKey(0)