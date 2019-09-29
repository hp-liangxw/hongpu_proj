"""
图片名称中有中文，先把中午去掉
"""
import os
import glob
import shutil
import cv2
from xpinyin import Pinyin

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
old_folder = r"D:\OneDrive\MyDrive\OneDrive\Projects\hongpu\Battery\images\NG\20190921_极耳"
pre_str = "20190921_"

# 将路径中的中文修改为英文
new_folder = Pinyin().get_pinyin(old_folder, splitter="")
# 若英文路径不存在则创建该路径
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 图片列表
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!
img_list = glob.glob(os.path.join(old_folder, "*.bmp"))
for i in img_list:
    # 1. 保存img
    old_img_path = i
    old_img_name = os.path.basename(old_img_path)

    # 将文件名中的中文修改为英文，并修改特殊字符
    new_img_name = pre_str + Pinyin().get_pinyin(old_img_name, splitter="")
    new_img_name = new_img_name.replace("-", "_").replace(" ", "").replace("(", "_").replace(")", "")

    if not i.endswith(".jpg"):
        temp_file = r"D:\OneDrive\MyDrive\OneDrive\Projects\hongpu\Battery\images\NG\temp.bmp"
        shutil.copyfile(old_img_path, temp_file)
        data = cv2.imread(temp_file)
        os.remove(temp_file)

        new_img_name = new_img_name.split(".")[0] + ".jpg"
        new_img_path = os.path.join(new_folder, new_img_name)
        cv2.imwrite(new_img_path, data)
    else:
        new_img_path = os.path.join(new_folder, new_img_name)
        shutil.copyfile(old_img_path, new_img_path)

    # 2. 保存xml
    old_xml_name = old_img_name.split(".")[0] + ".xml"
    old_xml_path = os.path.join(old_folder, old_xml_name)

    if os.path.exists(old_xml_path):
        new_xml_name = pre_str + Pinyin().get_pinyin(old_xml_name, splitter="")
        new_xml_name = new_xml_name.replace("-", "_").replace(" ", "").replace("(", "_").replace(")", "")
        new_xml_path = os.path.join(new_folder, new_xml_name)

        with open(old_xml_path, "r", encoding="utf8") as fr:
            strs = fr.read()
        strs = strs.replace(old_img_name, new_img_name)

        with open(new_xml_path, 'w', encoding='utf8') as fw:
            fw.write(strs)
