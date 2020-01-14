import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def show_image(img_matrix):
    """
    show image
    :param img_matrix:
    :return:
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(img_matrix, cmap='gray'), plt.xticks([]), plt.yticks([])
    plt.show()


tic = time.time()
# read image
color = cv2.imread(r"D:\OneDrive\MyDrive\OneDrive\_Proj_code\hongpu_proj\kingsun\new.jpg")
gray = cv2.imread(r"D:\OneDrive\MyDrive\OneDrive\_Proj_code\hongpu_proj\kingsun\new.jpg", 0)

print(color.shape)
print(gray.shape)

# step 1
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
color[hsv[:, :, 0] < 60] = 255
# show_image(color)

# step 2
gray[(gray < 85) | (gray > 150)] = 255
blurred = cv2.dilate(gray, np.ones((3, 3), np.uint8), iterations=2)
blurred = cv2.erode(blurred, np.ones((3, 3), np.uint8), iterations=3)
color[blurred == 255] = 255
# show_image(color)
# cv2.imwrite("color.jpg", color)

print(time.time() - tic)
