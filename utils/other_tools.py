import os
import re
import shutil
import glob
import cv2
import random
import numpy as np
from xpinyin import Pinyin


def split_train_val_test(folder, rate=None):
    """
    分割训练集、验证集、测试集
    :return:
    """

    if rate is None:
        rate = [0.7, 0.05, 0.25]
    file_list = os.listdir(folder)
    file_num = len(file_list)

    # 打乱顺序
    random.shuffle(file_list)

    # 设置比例
    # rate = [0.7, 0.05, 0.25]  # 训练，验证，测试的比例

    train_list = file_list[:int(file_num * rate[0])]
    val_list = file_list[int(file_num * rate[0]):int(file_num * (rate[0] + rate[1]))]
    test_list = file_list[int(file_num * (rate[0] + rate[1])):]

    # 写入train.txt
    with open(os.path.sep.join([folder, r'train.txt']), 'w') as train_f:
        for i in train_list:
            if i[-4:] == ".xml":
                train_f.write(i[:-4] + '\n')

    # 写入val.txt
    with open(os.path.sep.join([folder, r'val.txt']), 'w') as val_f:
        for i in val_list:
            val_f.write(i[:-4] + '\n')

    # 写入trainval.txt
    with open(os.path.sep.join([folder, r'trainval.txt']), 'w') as trainval_f:
        for i in train_list:
            trainval_f.write(i[:-4] + '\n')
        for j in val_list:
            trainval_f.write(j[:-4] + '\n')

    # 写入test.txt
    with open(os.path.sep.join([folder, r'test.txt']), 'w') as test_f:
        for i in test_list:
            if i[-4:] == ".xml":
                test_f.write(i[:-4] + '\n')


def folder_diff(in_path_1, in_path_2):
    """
    根据文件名，比较两个文件夹中文件的差异
    :param in_path_1: 文件较少的文件夹
    :param in_path_2: 文件较多的文件夹
    :return:
    """
    #
    in_path_1 = glob.glob(r'D:\HongPuCorp\上海洪朴信息科技有限公司\轮胎行业视觉检测 - 文档\General\X光自动判级困难图片\侧泡\侧泡\*.jpg')
    #
    in_path_2 = glob.glob(r'D:\HongPuCorp\上海洪朴信息科技有限公司\轮胎行业视觉检测 - 文档\General\X光自动判级困难图片\20171111-侧泡\侧泡\*.jpg')

    f_n_1 = []
    f_n_2 = []

    for f_p in in_path_1:
        f_n = os.path.basename(f_p)
        f_n_1.append(f_n)

    for f_p in in_path_2:
        f_n = os.path.basename(f_p)
        f_n_2.append(f_n)

    for i in f_n_1:
        if i not in f_n_2:
            print(i)


def convert_img_type(src_dir, dst_dir, src_type, dst_type):
    """
    将源文件夹(src_dir)下格式为src_type的图片，转换为dst_type格式，并存放在dst_dir文件夹下。
    :param src_dir:
    :param dst_dir:
    :param src_type:
    :param dst_type:
    :return:
    """
    img_list = glob.glob(os.path.join(src_dir, "*." + src_type))

    for i in img_list:
        src_name = os.path.basename(i)
        dst_name = ".".join([src_name.split(".")[0], dst_type])

        img_data = cv2.imread(i)
        cv2.imwrite(os.path.join(dst_dir, dst_name), img_data)


def modify_filename(in_dir, file_type, char_before, char_after):
    """
    重命名文件。如将文件名中的空格替换为_：remove_space("in_dir", "jpg", " ", "_")

    将in_dir文件夹下file_type类型的文件的文件名中的char_before字符，替换为char_after
    :param in_dir:
    :param file_type:
    :param char_before:
    :param char_after:
    :return:
    """

    f_list = glob.glob(os.path.join(in_dir, "*." + file_type))

    for i in f_list:
        name = os.path.basename(i)
        output_name = name.replace(char_before, char_after)

        os.rename(os.path.join(in_dir, name),
                  os.path.join(in_dir, output_name))


def chinese_2_english(img_path):
    """
    将文件名中的中文修改为英文
    :param img_path: 图片的地址及名字
    :return: 修改后的图片地址及名字
    """
    pic_old_path, old_file_name = os.path.split(img_path)
    pic_new_path = Pinyin().get_pinyin(pic_old_path, '')
    if os.path.exists(pic_new_path):
        new_path = pic_new_path
    else:
        os.makedirs(pic_new_path)
        new_path = pic_new_path
    new_file_name = Pinyin().get_pinyin(old_file_name, '')
    new_pic = new_path + '\\' + new_file_name
    if not os.path.exists(new_pic):
        shutil.copyfile(img_path, new_pic)
    else:
        print('had changed')
    return new_pic


def remove_chinese_letters(img_folder, save_path):
    """
    使用正则表达式去掉文件名中的中文
    :param img_folder: 
    :param save_path: 
    :return: 
    """
    img_list = os.listdir(img_folder)

    for i in img_list:
        # 保留数字
        nums = re.findall(r"\d+\:?\d*", i)
        out_name = '_'.join(nums)
        os.rename(os.path.sep.join([img_folder, i]), os.path.sep.join([save_path, out_name + '.bmp']))


def get_iou(box1, box2):
    """
    计算两个bbox的交并比。
    usage:
        g_bbox = [20, 20, 50, 50]
        p_bbox = [40, 30, 80, 70]
        IOU = get_iou(g_bbox, p_bbox)
    :param box1:
    :param box2:
    :return:
    """

    ixmin = np.maximum(box1[0], box2[0])
    iymin = np.maximum(box1[1], box2[1])
    ixmax = np.minimum(box1[2], box2[2])
    iymax = np.minimum(box1[3], box2[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # intersection
    inters = iw * ih
    # union
    uni = ((box1[2] - box1[0]) * (box1[3] - box1[1]) +
           (box2[2] - box2[0]) * (box2[3] - box2[1]) - inters)

    # 计算该p框与所有g框之间的IOU
    IOU = inters / uni

    return IOU
