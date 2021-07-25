# Standard imports
import cv2
from PIL import Image
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from matplotlib import pyplot as plt

# Read image
img = cv2.imread("./ELISA-04192021/5nm/photos/3.png")
img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
# plt.hist(gray_img.ravel(),256,[0,256])
# plt.show()
cv2.imshow("img", img)
cv2.imshow("gray_img", gray_img )
cv2.imshow("gray_img_with_black_edge", gray_img_0  )

#################################  use histogram to remove area outside well  ##################################
hist_full = cv2.calcHist([gray_img_0],[0],None,[256],[0,256])
peak_0 = 0
peak_0_climbed = 0
peak_1 = 0
valley = 0
valley_left = 0
valley_right = 255

for i in range(2,256):
    if hist_full[i,0] > 1000 and peak_0 == 0 and valley == 0 and peak_1 == 0 and peak_0_climbed == 0:
        peak_0 = 1
    elif hist_full[i,0] > 3000 and peak_0 == 1 and valley == 0 and peak_1 == 0 and peak_0_climbed == 0:
        peak_0_climbed = 1
    elif hist_full[i,0] < 1000 and peak_0 == 1 and valley == 0 and peak_1 == 0 and peak_0_climbed ==1:
        peak_0 = 0
        valley = 1
        valley_left = i

    elif hist_full[i,0] >1000 and valley == 1:
        valley = 0
        peak_1 = 1
        valley_right = i

print(valley_left)
print(valley_right)

plt.hist(gray_img_0.ravel(),256,[0,256])
plt.show()

# gray_img_0 = np.where(gray_img_0 > int(valley_left), 255, gray_img_0)
ret0,gray_img_1 = cv2.threshold(gray_img_0,int(valley_left),255,cv2.THRESH_BINARY)
cv2.imshow("gray_img_raw_bright_area", gray_img_1)

# morphological transformations : opening
kernel_o = np.ones((5,5),np.uint8)
# erosion = cv2.erode(thresh_0,kernel_o,iterations = 1)
erosion = cv2.erode(gray_img_1,kernel_o,iterations = 2)
opening = cv2.dilate(erosion,kernel_o,iterations = 3)
# cv2.imshow("opening", opening )

# gray_img_2 = gray_img_0.astype(np.float) - opening.astype(np.float)
# gray_img_2 = np.where(gray_img_1 < 2, 0, gray_img_1).astype(np.uint8)
gray_img_1 = cv2.bitwise_and(gray_img_0,gray_img_0, mask= cv2.bitwise_not(opening))
cv2.imshow("only well area", gray_img_1 )


################################# remove dusts in well #######################################
# gray_img_0 = np.where(gray_img_0 > int(valley_left), 255, gray_img_0)
ret0,gray_img_2 = cv2.threshold(gray_img_1,int(valley_left),255,cv2.THRESH_BINARY)
cv2.imshow("gray_img_all_bright_area", gray_img_1)

# morphological transformations : opening
kernel_o = np.ones((5,5),np.uint8)
# erosion = cv2.erode(thresh_0,kernel_o,iterations = 1)
erosion_1 = cv2.erode(gray_img_2,kernel_o,iterations = 1)
opening_1 = cv2.dilate(erosion_1,kernel_o,iterations = 1)
# cv2.imshow("opening", opening )

# gray_img_2 = gray_img_0.astype(np.float) - opening.astype(np.float)
# gray_img_2 = np.where(gray_img_1 < 2, 0, gray_img_1).astype(np.uint8)
gray_img_2 = cv2.bitwise_and(gray_img_1,gray_img_1, mask= cv2.bitwise_not(opening_1))
cv2.imshow("only well area without dust", gray_img_2 )



cv2.waitKey(0)