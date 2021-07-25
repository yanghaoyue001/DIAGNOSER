# Standard imports
import cv2
from PIL import Image
import numpy as np
from imutils import paths
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen


imagePathString = "./ELISA-04192021/5nm/photos/"
imagePaths=sorted(list(paths.list_images(imagePathString)))

print("small dot number: #######################")
for imagePath in imagePaths:
    # imageName = imagePath[-23:-4]
    # Read image
    # img = cv2.imread("./20210310_BSA&SERUM/1PM/C/screenshots/18.png")
    img = cv2.imread(imagePath)
    img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape_0 = int(gray_img.shape[0] / 4) * 4
    shape_1 = int(gray_img.shape[1] / 4) * 4

    gray_empty0 = np.zeros([shape_0, shape_1])

    k_steps = 4  # kernel numbers for x and y
    k_size_0 = int(shape_0 / k_steps)
    k_size_1 = int(shape_1 / k_steps)
    binary_threshold = 250  # threshold of binary threshold

    for i in range(0, shape_0 - k_size_0 + 1, k_size_0):
        for j in range(0, shape_1 - k_size_1 + 1, k_size_1):
            kernel = gray_img[i:(i + k_size_0), j:(j + k_size_1)]
            yen_threshold = threshold_yen(kernel)
            kernel_rescale = rescale_intensity(kernel, (0, yen_threshold), (0, 255)).astype(np.uint8)
            ret0, thresh_kernel = cv2.threshold(kernel_rescale, binary_threshold, 255, cv2.THRESH_BINARY)
            gray_empty0[i:(i + k_size_0), j:(j + k_size_1)] = gray_empty0[i:(i + k_size_0),
                                                              j:(j + k_size_1)] + thresh_kernel

    gray_processed = np.uint8(gray_empty0)
    # cv2.imshow("kernel binary 0", gray_processed)

    #### apply filter over the masked photo to pick up big bright areas

    # 1 morphological transformations : opening
    kernel_o = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(gray_processed, cv2.MORPH_OPEN, kernel_o)
    erosion = cv2.erode(gray_processed, kernel_o, iterations=1)
    opening = cv2.dilate(erosion, kernel_o, iterations=5)
    # cv2.imshow("opening", opening)

    #### masked photo - big bright areas to leave only actual signals

    signals = gray_processed - opening  # leave only signals
    # cv2.imshow("signals", signals)

    ### overlay original photo and signals
    signals_red = np.zeros([signals.shape[0], signals.shape[1], 3]).astype(np.uint8)
    signals_red[:, :, 2] = signals
    img_reshape = img[0:shape_0, 0:shape_1, :]
    add_0 = cv2.addWeighted(img_reshape, 1, signals_red, 1, 0)
    # cv2.imshow("signals vs original", add_0)
    save_path = imagePath[:-4] +"-signals" + ".png"
    cv2.imwrite(save_path,add_0)
    print(int(np.sum(signals) / 255))
