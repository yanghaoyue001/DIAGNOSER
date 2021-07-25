
import os
import cv2
import numpy as np

FolderPathString = "./MAG COVID EVAL/mag-05312021/demo-images/A-3-G/"
filename = "37"
img_file_path_green = FolderPathString + filename + "-demo-only green.jpg"
img_file_path_red = FolderPathString + filename + "-demo-only red.jpg"
img_file_path_mag = FolderPathString + filename + "-demo-only mag.jpg"
img_green = cv2.imread(img_file_path_green)
img_red = cv2.imread(img_file_path_red)
img_mag = cv2.imread(img_file_path_mag)

img_gray_green = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)
num_green_beads = np.count_nonzero(img_gray_green)
sum_green_beads = np.sum(img_gray_green)
img_gray_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
num_red_beads = np.count_nonzero(img_gray_red)
sum_red_beads = np.sum(img_gray_red)
img_gray_mag = cv2.cvtColor(img_mag, cv2.COLOR_BGR2GRAY)
num_mag_beads = np.count_nonzero(img_gray_mag)

print("green intensity ratio", sum_green_beads/num_mag_beads/2 )
print("green pixel ratio", num_green_beads/num_mag_beads/2 )
print("red intensity ratio", sum_red_beads/num_mag_beads/2 )
print("red pixel ratio", num_red_beads/num_mag_beads/2 )
