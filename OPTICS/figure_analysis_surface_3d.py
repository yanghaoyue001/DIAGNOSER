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


# Load an color image
img = cv2.imread("./20201101/1/IMG_20201102-223538.jpg")

# Convert cv image to pil
img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
data_focus = img_pil[360 - 60 * 2:360 + 60 * 2, 480 - 60 * 2:480 + 60 * 2, :]
img_focus = Image.fromarray(data_focus)

print(np.shape(data_focus[:, :, 0]))

data_plot_x = []
data_plot_y = []
data_plot_z = []

i = 0
for x in range(360 - 60 * 2, 360 + 60 * 2):
    for y in range(480 - 60 * 2, 480 + 60 * 2):
        if ((x - 360) ** 2 + (y - 480) ** 2 < 55 ** 2):
            i += 1
            data_plot_x.append(x)
            data_plot_y.append(y)
            data_plot_z.append((0.2126 * img_pil[x, y, 0] + 0.7152 * img_pil[x, y, 1] + 0.00722 * img_pil[x, y, 2]))

data_plot_x = np.array(data_plot_x)
data_plot_y = np.array(data_plot_y)
data_plot_z = np.array(data_plot_z)

data_plot = np.zeros([max(data_plot_x) - min(data_plot_x) + 1, max(data_plot_y) - min(data_plot_y) + 1])
i = 0
for i in range(len(data_plot_x)):
    data_plot[(int)(data_plot_x[i] - min(data_plot_x)), (int)(data_plot_y[i] - min(data_plot_y))] = data_plot_z[i]


########################################### plot 3d surface
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.

X = np.arange(min(data_plot_x),max(data_plot_x)+1,1)
Y = np.arange(min(data_plot_y),max(data_plot_y)+1,1)
X, Y = np.meshgrid(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, data_plot, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
fig.savefig("test.pdf")



