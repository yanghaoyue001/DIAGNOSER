import numpy as np
import cv2
from PIL import Image
import math
from imutils import paths
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns
sns.set()

# threshold for the distribution threshold
threshold= 60
imagePathString = "./20210117-BSA-T1/4/raw/"
imagePaths=sorted(list(paths.list_images(imagePathString)))

for imagePath in imagePaths:
    imageName = imagePath[-23:-4]
    # print(imageName)

    # Load an color image
    img = cv2.imread(imagePath)

    # Convert cv image to pil
    img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data_focus = img_pil[360 - 60 * 2:360 + 60 * 2, 480 - 60 * 2:480 + 60 * 2, :]
    img_focus = Image.fromarray(data_focus)

    # print(np.shape(data_focus[:, :, 0]))

    data_plot_x = []
    data_plot_y = []
    data_plot_z = []

    i = 0
    for x in range(360 - 60 * 2, 360 + 60 * 2):
        for y in range(480 - 60 * 2, 480 + 60 * 2):
            if ((x - 360) ** 2 + (y - 480) ** 2 < 25 ** 2):
                i += 1
                data_plot_x.append(x)
                data_plot_y.append(y)
                data_plot_z.append((0.2126 * img_pil[x, y, 0] + 0.7152 * img_pil[x, y, 1] + 0.00722 * img_pil[x, y, 2]))

    data_plot_x = np.array(data_plot_x)
    data_plot_y = np.array(data_plot_y)
    data_plot_z = np.array(data_plot_z)

    percentage = 0
    weighted_sum = 0
    for i in range(len(data_plot_z)):
        if data_plot_z[i] >threshold:
            percentage +=1
            weighted_sum += data_plot_z[i]
    percentage = percentage / len(data_plot_z)
    print(percentage, "", weighted_sum)

