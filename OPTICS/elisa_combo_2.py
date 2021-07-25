# Standard imports
import cv2
from PIL import Image
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen

# Read image
img = cv2.imread("./ELISA-04192021/5nm/photos/40.png")
img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

################################ enhance contrastness and mask threshold for the whole picture
# Enhance image
# gray_img = cv2.equalizeHist(gray_img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Keypoints1", gray_img )
yen_threshold = threshold_yen(gray_img)
gray_img = rescale_intensity(gray_img, (0, yen_threshold), (0, 255)).astype(np.uint8)

ret0,thresh0 = cv2.threshold(gray_img,248,255,cv2.THRESH_BINARY)
# Bitwise-AND mask and original image
# img_copy_0 = img
# res_dot = cv2.bitwise_and(img_copy_0, img_copy_0 , mask=thresh0)

cv2.imshow("orig", img )
# cv2.imshow("increase brightness", gray_img )
# cv2.imshow("whole area binary", thresh0 )

################################ enhance contrastness and mask for each kernel

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray orig", gray_img )
shape_0 = int(gray_img.shape[0]/10)*10
shape_1 = int(gray_img.shape[1]/10)*10

gray_enhance_0 = np.zeros([shape_0,shape_1])
gray_empty_0 = np.zeros([shape_0,shape_1])

gray_enhance_1 = np.zeros([shape_0,shape_1])
gray_empty_1 = np.zeros([shape_0,shape_1])
gray_kernel_sum_1 = np.zeros([shape_0,shape_1])

k_steps = 10 # kernel numbers for x and y
k_size_0 = int(shape_0/k_steps)
k_size_1 = int(shape_1/k_steps)
###### fixed binary threshold, and apply threshold after rescale
# binary_threshold = 250 # threshold of binary threshold

for i in range(0, shape_0 - k_size_0+1, k_size_0):
    for j in range(0, shape_1 - k_size_1+1, k_size_1):
        kernel = gray_img[i:(i+k_size_0),j:(j+k_size_1)]
        # #### use yen_threshold
        # yen_threshold = threshold_yen(kernel)
        # kernel_rescale = rescale_intensity(kernel, (0, yen_threshold), (0, 255)).astype(np.uint8)
        # ### adaptive binary threshold
        # binary_threshold = np.min(kernel_rescale)+(np.max(kernel_rescale)-np.min(kernel_rescale))*0.5
        # # apply threshold
        # ret0, thresh_kernel_0 = cv2.threshold(kernel_rescale, binary_threshold, 255, cv2.THRESH_BINARY)
        # gray_enhance_0[i:(i + k_size_0), j:(j + k_size_1)] = gray_enhance_0[i:(i + k_size_0),j:(j + k_size_1)] + kernel_rescale
        # gray_empty_0[i:(i+k_size_0),j:(j+k_size_1)] = gray_empty_0[i:(i+k_size_0),j:(j+k_size_1)] + thresh_kernel_0

        ### or choose yen_threshold without black corners
        kernel_1 = kernel
        # x = np.max(kernel_1[kernel_1 > 30])
        # valid_lowest_brightness = np.min(np.where(kernel_1<30, 255, kernel_1))
        # kernel_1 = np.where(kernel_1<30, valid_lowest_brightness, kernel_1)
        kernel_1 = np.where(kernel_1 < 45, 0, kernel_1)
        yen_threshold_1 = threshold_yen(kernel_1)
        kernel_rescale_1 = rescale_intensity(kernel_1, (0, yen_threshold_1), (0, 255)).astype(np.uint8)
        ### adaptive binary threshold
        binary_threshold = np.min(kernel_rescale_1)+(np.max(kernel_rescale_1)-np.min(kernel_rescale_1))*0.9
        # apply threshold
        ret_1, thresh_kernel_1 = cv2.threshold(kernel_rescale_1, binary_threshold, 255, cv2.THRESH_BINARY)
        gray_enhance_1[i:(i + k_size_0), j:(j + k_size_1)] = gray_enhance_0[i:(i + k_size_0),j:(j + k_size_1)] + kernel_rescale_1
        gray_empty_1[i:(i+k_size_0),j:(j+k_size_1)] = gray_empty_1[i:(i+k_size_0),j:(j+k_size_1)] + thresh_kernel_1
        gray_kernel_sum_1[i:(i + k_size_0), j:(j + k_size_1)] = gray_kernel_sum_1[i:(i + k_size_0),
                                                           j:(j + k_size_1)] + kernel_1

gray_enhanced_kernel_0 = np.uint8(gray_enhance_0)
gray_processed_0 = np.uint8(gray_empty_0)
# cv2.imshow("orig gray", gray_img )
# cv2.imshow("kernel enhance 0", gray_enhanced_kernel_0 )
# cv2.imshow("kernel binary 0", gray_processed_0 )

gray_enhanced_kernel_1 = np.uint8(gray_enhance_1)
gray_processed_1 = np.uint8(gray_empty_1)

cv2.imshow("gray kernel sum 1", gray_kernel_sum_1.astype(np.uint8) )
cv2.imshow("kernel enhance 1", gray_enhanced_kernel_1 )
cv2.imshow("kernel binary 1", gray_processed_1 )






#### apply filter over the masked photo to pick up big bright areas

# 1 morphological transformations : opening
kernel_o = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(gray_processed, cv2.MORPH_OPEN, kernel_o)
erosion = cv2.erode(gray_processed_1,kernel_o,iterations = 1)
opening = cv2.dilate(erosion,kernel_o,iterations = 5)
cv2.imshow("opening", opening )

#### masked photo - big bright areas to leave only actual signals

signals = gray_processed_1 - opening  # leave only signals
# cv2.imshow("signals", signals)

### overlay original photo and signals
signals_red = np.zeros([signals.shape[0],signals.shape[1],3]).astype(np.uint8)
signals_red[:,:,2]=signals
# img_reshape = img[0:shape_0,0:shape_1,:]
# add_0 = cv2.addWeighted(img_reshape, 1, signals_red, 1, 0)
img_reshape = gray_img[0:shape_0,0:shape_1]
add_0 = cv2.addWeighted(img_reshape, 1, signals, 1, 0)
cv2.imshow("signals vs original", add_0)
print(int(np.sum(signals)/255))
# ################################ echance contrastness for each kernel and mask for the whole picture
# not initiated











cv2.waitKey(0)

