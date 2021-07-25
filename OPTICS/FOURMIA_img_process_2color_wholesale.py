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
folder_names = ["A-1-G","A-2-G","A-3-G","A-4-G",
                "B-1-G","B-2-G","B-3-G","B-4-G",
                "C-1-G","C-2-G","C-3-G","C-4-G",
                "D-1-G","D-2-G","D-3-G","D-4-G"]
# folder_name = "10pm-2"
file1 = open("results.txt","w")

for folder_name in folder_names:
    imagePathString = "./MAG COVID EVAL/mag-05312021/plate3/" + folder_name + "/2/"
    imagePaths = sorted(list(paths.list_images(imagePathString)))

    for imagePath in imagePaths:
        img_raw = cv2.imread(imagePath)
        img_0 = img_raw[120:1030, 620:1230, :]
        img_np_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
        hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
        img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("orig", img_0)
        # cv2.imshow("gray orig", img_gray_0)

        img_2 = img_0.copy()
        img_3 = img_0.copy()
        s = (img_3.shape[0], img_3.shape[1], img_3.shape[2])
        img_4 = np.zeros(s).astype(np.uint8)
        img_4_2 = img_4.copy()
        for i in range(0, img_0.shape[0]):
            for j in range(0, img_0.shape[1]):
                if img_gray_0[i, j] > 15:  # parameter 1
                    if img_2[i, j, 2] > 10 and img_2[i, j, 0] > 8:  # parameter 2 and 3
                        if img_2[i, j, 2] / img_2[i, j, 1] > 0.55:  # parameter 4
                            img_3[i, j, :] = img_2[i, j, :].copy()
                        else:
                            img_3[i, j, :] = [0, 0, 0]
                    else:
                        img_3[i, j, :] = [0, 0, 0]
                else:
                    img_3[i, j, :] = [0, 0, 0]

                # #### for only green beads
                # if img_gray_0[i, j] > 50 and img_2[i, j, 2] / img_2[i, j, 1] > 0.6:  # parameter 5 6
                #     img_4[i, j, :] = img_2[i, j, :].copy()
                # elif img_gray_0[i, j] > 40 and img_2[i, j, 2] / img_2[i, j, 1] > 0.7:  # parameter 7 8
                #     img_4[i, j, :] = img_2[i, j, :].copy()

                ##### for red beads in green-red condition
                # if img_gray_0[i, j] > 50 and img_2[i, j, 2] / img_2[i, j, 1] > 0.65 and img_2[i, j, 0] / img_2[
                #     i, j, 2] < .75:  # parameter 5 6 7; <0.75 red
                #     img_4[i, j, :] = img_2[i, j, :].copy()
                # elif img_gray_0[i, j] > 40 and img_2[i, j, 2] / img_2[i, j, 1] > 0.75 and img_2[i, j, 0] / img_2[
                #     i, j, 2] < .75:  # parameter 8 9 10; <0.75 red
                #     img_4[i, j, :] = img_2[i, j, :].copy()

                # if img_gray_0[i, j] > 50 and img_2[i, j, 2] / img_2[i, j, 1] > 0.4 and img_2[i, j, 0] / img_2[
                #     i, j, 2] > .75:  # parameter 5 6 7; >0.75 green
                #     img_4[i, j, :] = img_2[i, j, :].copy()
                # elif img_gray_0[i, j] > 40 and img_2[i, j, 2] / img_2[i, j, 1] > 0.5 and img_2[i, j, 0] / img_2[
                #     i, j, 2] > .75:  # parameter 8 9 10; >0.75 green
                #     img_4[i, j, :] = img_2[i, j, :].copy()

                if img_gray_0[i, j] > 50 and img_2[i, j, 2] / img_2[i, j, 1] > 0.6 and img_2[i, j, 2] / img_2[
                    i, j, 1] < 0.8:  # parameter 5 6 green
                    img_4[i, j, :] = img_2[i, j, :].copy()
                elif img_gray_0[i, j] > 40 and img_2[i, j, 2] / img_2[i, j, 1] > 0.7 and img_2[i, j, 2] / img_2[
                    i, j, 1] < 0.8:  # parameter 7 8 green
                    img_4[i, j, :] = img_2[i, j, :].copy()

                if img_gray_0[i, j] > 50 and img_gray_0[i, j] < 150 and img_2[i, j, 2] / img_2[
                    i, j, 1] > 0.8:  # parameter 5 6 red
                    img_4_2[i, j, :] = img_2[i, j, :].copy()
                elif img_gray_0[i, j] > 40 and img_gray_0[i, j] < 150 and img_2[i, j, 2] / img_2[
                    i, j, 1] > 0.85:  # parameter 7 8 red
                    img_4_2[i, j, :] = img_2[i, j, :].copy()


        # cv2.imshow("bright pixels", img_3)  # mag beads + signals + bright noises
        # cv2.imshow("only sharp", img_4)  # signals + bright noises

        img_5 = img_3 - img_4  # only mag beads
        img_gray_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2GRAY)

        # # # remove large area sharp area by opening --------------------------------------------------------------------------
        img_gray_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(img_gray_4, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=2)
        img_gray_4_denoise = img_gray_4 - erosion
        # cv2.imshow("opening", erosion)
        # cv2.imshow("denoise", img_gray_4_denoise)  # only signals

        img_gray_4_2 = cv2.cvtColor(img_4_2, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(img_gray_4_2, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=2)
        img_gray_4_2_denoise = img_gray_4_2 - erosion
        # cv2.imshow("opening", erosion)
        # cv2.imshow("denoise red", img_gray_4_2_denoise)  # only signals

        ### red mask
        redLower = (0, 0, 80)
        redUpper = (255, 255, 255)
        hsv = cv2.cvtColor(img_4_2, cv2.COLOR_BGR2HSV)
        mask_r0 = cv2.inRange(hsv, redLower, redUpper)
        img_6_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_r0)
        img_6 = cv2.cvtColor(img_6_hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow("red", img_6)  # only signals

        # # # calculation ------------------------------------------------------------------------------------------------------
        num_mag_beads = np.count_nonzero(img_gray_5)
        num_signals_green = np.count_nonzero(img_gray_4_denoise)
        brightness_signals_green = np.sum(img_gray_4_denoise)
        num_signals_red = np.count_nonzero(img_gray_4_2_denoise)
        brightness_signals_red = np.sum(img_gray_4_2_denoise)

        head, tail = os.path.split(imagePath)
        magbeads_filename = tail.split(".", 1)[0] + "-mag.jpg"
        cv2.imwrite(os.path.join(imagePathString, magbeads_filename), img_gray_5)
        signal_filename_green = tail.split(".", 1)[0] + "-signal-g.jpg"
        cv2.imwrite(os.path.join(imagePathString, signal_filename_green), img_gray_4_denoise)
        signal_filename_red = tail.split(".", 1)[0] + "-signal-r.jpg"
        cv2.imwrite(os.path.join(imagePathString, signal_filename_red), img_gray_4_2_denoise)

        print(folder_name, tail.split(".", 1)[0], num_mag_beads, num_signals_green, brightness_signals_green, num_signals_red, brightness_signals_red)
        write_line = folder_name + ","+ tail.split(".", 1)[0] +","+  str(num_mag_beads) +","+  str(num_signals_green) +","+  str(brightness_signals_green) + \
                     "," + str(num_signals_red) + "," + str(brightness_signals_red) +" \n"
        file1.write(write_line)
        # cv2.waitKey(0)

file1.close()
