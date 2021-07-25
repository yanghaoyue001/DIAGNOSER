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
from imutils import paths

# folder_names = ["1pm-2","1pm-3","100fm-1","100fm-2","100fm-3","10fm-1","10fm-2","10fm-3","1fm-1","1fm-2","1fm-3",
#                 "1fm-4","0fm-1","0fm-2","0fm-3","0fm-4",]
# folder_names = ["0fm-4","1fm-4"]
# folder_names = ["A-1-G","A-2-G","A-3-G","A-4-G",
#                 "B-1-G","B-2-G","B-3-G","B-4-G",
#                 "C-1-G","C-2-G","C-3-G","C-4-G",
#                 "D-1-G","D-2-G","D-3-G","D-4-G"]
folder_names = ["4-1","4-2","5-1","5-2",
                "6-1","6-2","7-1","7-2",
                "8-1","8-2","9-1","9-2",
                "10-1","10-2"]
file1 = open("results.txt","w")

for folder_name in folder_names:
    imagePathString = "./MAG COVID EVAL/mag-05302021/fluor-bg/" + folder_name
    imagePaths = sorted(list(paths.list_images(imagePathString)))

    for imagePath in imagePaths:
        img_raw = cv2.imread(imagePath)
        img_0 = img_raw[120:1030, 620:1230, :]
        img_np_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
        hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
        img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)

        img_2 = img_0.copy()

        img_gray_2 = img_gray_0.copy()
        img_gray_2 = np.where(img_gray_2 < 15, 15, img_gray_2)
        yen_threshold_3 = threshold_yen(img_gray_2)
        img_gray_2 = rescale_intensity(img_gray_2, (0, yen_threshold_3), (0, 255)).astype(np.uint8)
        # print("yen threshold =", yen_threshold_3)
        # cv2.imshow("yen pixels", img_gray_2)


        img_3 = img_0.copy()
        s = (img_3.shape[0], img_3.shape[1], img_3.shape[2])
        img_4 = np.zeros(s).astype(np.uint8)

        for i in range(0, img_0.shape[0]):
            for j in range(0, img_0.shape[1]):
                if img_gray_2[i, j] > 200 and img_gray_0[i, j] > 20:  # parameter 1
                    img_3[i, j, :] = img_2[i, j, :].copy()
                else:
                    img_3[i, j, :] = [0, 0, 0]

        # cv2.imshow("bright pixels", img_3)  # mag beads + signals + bright noises

        ##############################################- only fluor beads condition #############################################

        # # # calculation ------------------------------------------------------------------------------------------------------
        img_gray_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)

        num_signals = np.count_nonzero(img_gray_3)
        brightness_signals = np.sum(img_gray_3)

        head, tail = os.path.split(imagePath)
        signal_filename = tail.split(".", 1)[0] + "-signal.jpg"
        cv2.imwrite(os.path.join(imagePathString, signal_filename), img_3)

        print(folder_name, tail.split(".", 1)[0], num_signals, brightness_signals)
        write_line = folder_name + ","+ tail.split(".", 1)[0] +","+  str(num_signals) +","+  str(brightness_signals) +" \n"
        file1.write(write_line)
        # cv2.waitKey(0)

file1.close()
