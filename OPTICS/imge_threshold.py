import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample1.png',0)
# img = cv2.equalizeHist(img)

ret,th1 = cv2.threshold(img,40,255,cv2.THRESH_TRUNC) #40 threshold
ret,th2 = cv2.threshold(img,120,122,cv2.THRESH_BINARY) #70 threshold
# th3 = cv2.adaptiveThreshold(img,125,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
th3 = cv2.adaptiveThreshold(img,122,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,6)

thf = np.multiply(th1,(th2+th3))

# Show keypoints
cv2.imshow("gray",img)
cv2.imshow("th1", th1)
cv2.imshow("th2",th2)
cv2.imshow("th3",th3)
cv2.imshow("thfinal",thf)
cv2.waitKey(0)
