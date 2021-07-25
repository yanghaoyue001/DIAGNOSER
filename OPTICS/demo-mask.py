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


FolderPathString = "./MAG COVID EVAL/mag-05312021/plate3/D-1-G/2 (copy)/"
filename = "16-demo.png"
img_file_path = FolderPathString + filename
img_raw = cv2.imread(img_file_path)
img_0 = img_raw.copy()
img_np_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
cv2.imshow("orig", img_0)
cv2.imshow("gray orig", img_gray_0)

# Lower = (0, 0, 80)
# Upper = (255, 255, 255)
# bg
Lower = (0, 0, 0)
Upper = (255, 255, 50)

img_1 = img_0.copy()
hsv = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
mask_r0 = cv2.inRange(hsv, Lower, Upper)
img_1_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_r0)
img_2 = cv2.cvtColor(img_1_hsv, cv2.COLOR_HSV2BGR)

# FolderPathString = "./MAG COVID EVAL/mag-05312021/plate3/C-3-G/2 (copy)/"
# filename = "16-demo-bg.jpg"
# img_file_path1 = FolderPathString + filename
# img_raw1 = cv2.imread(img_file_path1)
# img_3 = img_raw1.copy()
# img_2 = img_2 + img_3

head, tail = os.path.split(img_file_path)
result_filename = tail.split(".",1)[0] + "-1.jpg"
cv2.imwrite(os.path.join(FolderPathString,result_filename),img_2)
cv2.imshow("after mask bgr", img_2)

cv2.waitKey(0)