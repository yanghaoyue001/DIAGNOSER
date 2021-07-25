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
img_file_path = "./MAG COVID EVAL/mag-05312021/plate3/A-1-G/2/20.png"
img_raw = cv2.imread(img_file_path)
img_0 = img_raw[120:1030,620:1230,:]
img_np_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
cv2.imshow("orig", img_0)
cv2.imshow("gray orig", img_gray_0)
cv2.waitKey(0)