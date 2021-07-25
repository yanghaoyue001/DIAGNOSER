
import os
import cv2


FolderPathString = "./MAG COVID EVAL/mag-05312021/demo-images/A-3-G/"
filename = "16-demo.png"
img_file_path = FolderPathString + filename
img_raw = cv2.imread(img_file_path)
img_0 = img_raw.copy()
img_np_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
cv2.imshow("orig", img_0)
cv2.imshow("gray orig", img_gray_0)