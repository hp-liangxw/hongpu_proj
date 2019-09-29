import os
import cv2
import numpy as np

img_folder = r"D:\OneDrive\MyDrive\OneDrive\_Proj_code\hongpu_proj\leqi_battery\imgs"
bmp_name = r"xs_2019_09_18_131947_blockId#20528.bmp"
jpg_name = r"xs_2019_09_18_131947_blockId#20528.jpg"

bmp_data = cv2.imread(os.path.join(img_folder, bmp_name))

jpg_file = cv2.imwrite(os.path.join(img_folder, jpg_name), bmp_data)

jpg_data = cv2.imread(os.path.join(img_folder, jpg_name))

flag = np.sum(bmp_data - jpg_data)
print(flag)

# 字节转换后的数据
fd = open(os.path.join(img_folder, jpg_name), "rb")
image_byte = fd.read()
np_arr = np.fromstring(image_byte, np.uint8)
jpg_decode = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

flag = np.sum(jpg_decode - jpg_data)

print(flag)