
import os
import cv2



FolderPathString = "./MAG COVID EVAL/mag-05312021/demo-images/B-2-G/"
filename = "30-demo-bg.jpg"
img_file_path = FolderPathString + filename
img_raw = cv2.imread(img_file_path)
img_0 = img_raw.copy()
img_np_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
cv2.imshow("orig", img_0)
cv2.imshow("gray orig", img_gray_0)

greenLower = (1, 90, 0)
greenUpper = (255, 255, 50)

# img_1 = img_0.copy()
# hsv = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
# mask_r0 = cv2.inRange(hsv, greenLower, greenUpper)
# img_1_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_r0)
# img_2 = cv2.cvtColor(img_1_hsv, cv2.COLOR_HSV2BGR)
#
# cv2.imshow("after mask bgr", img_2)

img_2 = img_0.copy()
for i in range(0, img_0.shape[0]):
    for j in range(0, img_0.shape[1]):
        if img_gray_0[i,j] >= 25:
            if img_2[i, j,2] < 20 or img_2[i, j,2]/img_2[i, j,1] < 0.68 or img_2[i, j,0] < 22:  # parameter 1
                img_2[i, j]=[0,0,0]
        if img_gray_0[i,j] < 25:
            if img_2[i, j,2] < 17 or img_2[i, j,2]/img_2[i, j,1] < 0.8 or img_2[i, j,0] < 16:  # parameter 1
                img_2[i, j]=[0,0,0]


# th3 = cv2.adaptiveThreshold(img_2,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)


head, tail = os.path.split(img_file_path)
result_filename = tail.split(".",1)[0] + "-1.jpg"
cv2.imwrite(os.path.join(FolderPathString,result_filename),img_2)
cv2.imshow("after mask + intensity bgr", img_2)

cv2.waitKey(0)