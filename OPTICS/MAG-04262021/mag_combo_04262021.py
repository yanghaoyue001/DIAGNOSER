# Standard imports
import cv2
from PIL import Image
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from matplotlib import pyplot as plt

# Read image
# img 0PM
img = cv2.imread("./SETTING 1/0PM.png")
img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# hist_full_black = cv2.calcHist([gray_img],[0],None,[256],[0,256])

#################################  use gray range to make all dark area to real black ########################
gray_img_0 = gray_img
# find threshold for black background by histogram
hist_full_black = cv2.calcHist([gray_img],[0],None,[256],[0,256])
peak_black = 0
threshold_black = 0
for i in range(2,50):
    if hist_full_black[i,0] > 1000 and peak_black == 0:
        peak_black = 1
    elif hist_full_black[i,0] < 100 and peak_black == 1:
        # print(i)
        for j in range(i,50):
            if (hist_full_black[j,0] >100):
                break
            threshold_black = j
        break
print(threshold_black)

gray_img_0 = np.where(gray_img_0 < threshold_black, 0, gray_img_0)
cv2.imshow("gray_img", gray_img )
cv2.imshow("gray_img with black to 0", gray_img_0)
mask_no_black_corners = np.where(gray_img_0 > threshold_black, 255, gray_img_0)
cv2.imshow("mask with black to 0", mask_no_black_corners)

hsv_with_no_black = cv2.bitwise_and(hsv, hsv, mask=mask_no_black_corners)
cv2.imshow("hsv with no black", hsv_with_no_black)

bgr_with_no_black = cv2.cvtColor(hsv_with_no_black, cv2.COLOR_HSV2BGR)
#################################  remove well empty area in well ########################
# color threshold in BGR space
blue_low   = 0
blue_high  = 255
green_low  = 0
green_high = 255
red_low    = 0
red_high   = 255

for i in range(0,np.shape(bgr_with_no_black)[0]):
    for j in range(0,np.shape(bgr_with_no_black)[1]):





# plt.subplot(4,1,1)
plt.hist(gray_img.ravel(),256,[0,256])
# print(np.sum(hist_full_black[125:256]))
# cv2.imshow("GRAY 0PM", gray_img)
# cv2.imshow("HSV 0PM", hsv)







# # img 1PM
# img = cv2.imread("./SETTING 1/1PM.jpg")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hist_full_black = cv2.calcHist([gray_img],[0],None,[256],[0,256])
# # plt.subplot(4,1,2)
# # plt.hist(gray_img.ravel(),256,[50,256])
# print(np.sum(hist_full_black[125:256]))
# cv2.imshow("GRAY 1PM", gray_img)
#
# # img 10PM
# img = cv2.imread("./SETTING 1/10PM.jpg")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hist_full_black = cv2.calcHist([gray_img],[0],None,[256],[0,256])
# # plt.subplot(4,1,3)
# # plt.hist(gray_img.ravel(),256,[50,256])
# print(np.sum(hist_full_black[125:256]))
# cv2.imshow("GRAY 10PM", gray_img)
#
# # img 100PM
# img = cv2.imread("./SETTING 1/100PM.jpg")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hist_full_black = cv2.calcHist([gray_img],[0],None,[256],[0,256])
# # plt.subplot(4,1,4)
# # plt.hist(gray_img.ravel(),256,[50,256])
# print(np.sum(hist_full_black[125:256]))
# cv2.imshow("GRAY 100PM", gray_img)

plt.show()
cv2.waitKey(0)