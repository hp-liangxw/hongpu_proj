# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import cv2
import glob
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import yaml
import shutil
import logging
import pandas as pd
import numpy as np
from src.pre_process import grid_cut
from src.yinlie_post_process import yinlie_xizhi
from src.auto_labeling import csv2xml
from output_md import *

sys.path.append('/detectron/tools')
from infer_simple import infer_simple


class Config:
    def __init__(self, config_path):
        self.config_path = config_path

        self.yml_str = None

        self.types_and_ratios = None
        self.model_paths = None

        self.INFO = None
        self.PRE_PROCESS = None
        self.IMG_CUT_FOLDER = None
        self.OUTPUT_FOLDER = None
        self.MODEL = None
        self.AUTO_LABELING = None
        self.POST_PROCESS = None
        self.EVALUATION = None

        self.is_valid = self.check_if_valid()

    def check_if_valid(self):
        """
        确认配置文件的正确性
        :param yml_str:
        :return:
        """
        if not os.path.exists(self.config_path):
            logging.info("the configuration file: {} is not exists!".format(self.config_path))
            return 1

        # 读取配置文件
        self.yml_str = open(self.config_path).read()
        try:
            cfg_info = yaml.load(self.yml_str)

            self.INFO = cfg_info['info']

            self.PRE_PROCESS = cfg_info['pre_process']
            self.POST_PROCESS = cfg_info['post_process']

            self.IMG_CUT_FOLDER = cfg_info['img_folder']
            self.OUTPUT_FOLDER = cfg_info['output_folder']

            self.MODEL = cfg_info['model']

            self.AUTO_LABELING = cfg_info['auto_labeling']
            self.EVALUATION = cfg_info['evaluation']
        except:
            print("failed to read configuration file")
            return 1

        # 有前处理时，前处理文件夹是否存在
        if self.PRE_PROCESS['switch'] is True:
            if not os.path.exists(self.PRE_PROCESS['origin_folder']):
                print("path: {} not exists!".format(self.PRE_PROCESS['origin_folder']))
                return 1

        # 图片输入路径是否存在
        if not os.path.exists(self.IMG_CUT_FOLDER):
            print("path: {} not exists!".format(self.IMG_CUT_FOLDER))
            return 1

        # 模型路径是否存在
        self.types_and_ratios, self.model_paths = self.get_defect_info()
        for i in self.model_paths.values():
            if not os.path.exists(i):
                print("path: {} not exists!".format(i))
                return 1

        # 答案是否存在
        if self.EVALUATION['switch'] is True:
            if not os.path.exists(self.EVALUATION['csv_path']):
                print("path: {} not exists!".format(self.EVALUATION['csv_path']))
                return 1

        # 创建输出文件夹
        if os.path.exists(self.OUTPUT_FOLDER):
            shutil.rmtree(self.OUTPUT_FOLDER)
        os.mkdir(self.OUTPUT_FOLDER)
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'csv'))
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'img_with_box'))
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'xml'))
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'pics'))

        # 再次查看配置信息正确性
        print("\n\n>>>>>>>>----------------------------------------------------------------<<<<<<<<\n")
        print(self.yml_str)
        print("\n>>>>>>>>----------------------------------------------------------------<<<<<<<<\n\n")

        answer = raw_input("Is the configuration file correct? [yes/no]:")
        while answer.upper() not in ["YES", "Y", "NO", "N"]:
            print("Please input yes or no!")
            answer = raw_input()

        if answer.upper() in ["YES", "Y"]:
            return 0
        else:
            return 1

    def get_defect_info(self):
        """
        获取缺陷类型、缺陷置信度、模型路径
        :param MODEL:
        :return:
        """
        # 缺陷类型与置信度
        types_and_ratios = {}
        model_paths = {}
        for model_type, model_info in self.MODEL.items():
            model_paths[model_type] = model_info["model_dir"]
            for defect_type, defect_info in model_info["threshold"].items():
                types_and_ratios[defect_type] = defect_info

        return types_and_ratios, model_paths

    def get_pic_info(self):
        """
        获取图片名称和图片大小
        若需要前处理，则输出切图时的行列信息
        :return:
        """
        logging.info('!!!getting pic information')

        pic_info = {}
        line_dict = {}

        if self.PRE_PROCESS['switch']:
            logging.info('!!!start pre_process')

            # 如果存在小图文件夹，先删除再新建
            if os.path.exists(self.IMG_CUT_FOLDER):
                shutil.rmtree(self.IMG_CUT_FOLDER)
            os.mkdir(self.IMG_CUT_FOLDER)

            # cut zujian pic
            for img in glob.glob(os.path.join(self.PRE_PROCESS['origin_folder'], '*.jpg')):
                img_name = os.path.basename(img).split('.')[0]
                img_data = cv2.imread(img)
                try:
                    pic_info[img_name] = img_data.shape[:2]
                    cut_images, (row_lines, col_lines) = grid_cut(img_data, self.PRE_PROCESS['size']['rows'],
                                                                  self.PRE_PROCESS['size']['cols'], False)
                    line_dict[img_name] = (row_lines, col_lines)
                    for i in range(len(row_lines) - 1):
                        for j in range(len(col_lines) - 1):
                            cv2.imwrite(
                                os.path.join(self.IMG_CUT_FOLDER, "-".join([img_name, str(i), str(j)]) + '.jpg'),
                                cut_images[row_lines[i], col_lines[j]])
                except:
                    pass

            print('===========================================================')
            print('!!!finish pre_process')
        else:
            for img in glob.glob(os.path.join(self.IMG_CUT_FOLDER, '*.jpg')):
                img_name = os.path.basename(img).split('.')[0]
                img_data = cv2.imread(img)
                pic_info[img_name] = img_data.shape[:2]

        return pic_info, line_dict


def model_predict(config_info):
    """
    model predict
    :param config_info: 配置信息
    :return:
    """
    MODEL = config_info.MODEL
    INPUT_FOLDER = config_info.IMG_CUT_FOLDER
    OUTPUT_FOLDER = config_info.OUTPUT_FOLDER

    for model_key, model_values in MODEL.items():
        model_dir = model_values['model_dir']
        threshold = model_values['threshold']
        infer_simple(INPUT_FOLDER, os.path.join(OUTPUT_FOLDER, 'csv'), model_dir, threshold)
    logging.info('===========================================================')
    logging.info('!!!finish model predict')


def merge_csv_files(config_info, line_dict):
    """
    merge bbox output
    :param config_info:
    :param line_dict:
    :return:
    """
    INPUT_FOLDER = config_info.IMG_CUT_FOLDER
    OUTPUT_FOLDER = config_info.OUTPUT_FOLDER
    PRE_PROCESS = config_info.PRE_PROCESS
    POST_PROCESS = config_info.POST_PROCESS

    # each defect model corresponds to a csv
    csv_total = pd.DataFrame(columns=['pic_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'score'])
    for csv_file in glob.glob(os.path.join(OUTPUT_FOLDER, 'csv', '*.csv')):
        csv_temp = pd.read_csv(csv_file)
        csv_total = pd.concat([csv_total, csv_temp], ignore_index=True, sort=False)

    csv_total['xmin'] = csv_total['xmin'].astype('int')
    csv_total['ymin'] = csv_total['ymin'].astype('int')
    csv_total['xmax'] = csv_total['xmax'].astype('int')
    csv_total['ymax'] = csv_total['ymax'].astype('int')

    if PRE_PROCESS['switch']:
        csv_total['big_pic'] = [name.split('-')[0] for name in csv_total['pic_name']]
        csv_total['row'] = [int(name.split('-')[1]) for name in csv_total['pic_name']]
        csv_total['col'] = [int(name.split('-')[2]) for name in csv_total['pic_name']]
        for i in range(len(csv_total)):
            x_append = line_dict[csv_total.loc[i, 'big_pic']][1][csv_total.loc[i, 'col']]
            y_append = line_dict[csv_total.loc[i, 'big_pic']][0][csv_total.loc[i, 'row']]
            csv_total.loc[i, 'bigxmin'] = x_append + csv_total.loc[i, 'xmin']
            csv_total.loc[i, 'bigxmax'] = x_append + csv_total.loc[i, 'xmax']
            csv_total.loc[i, 'bigymin'] = y_append + csv_total.loc[i, 'ymin']
            csv_total.loc[i, 'bigymax'] = y_append + csv_total.loc[i, 'ymax']
        csv_total['bigxmin'] = csv_total['bigxmin'].astype('int')
        csv_total['bigymin'] = csv_total['bigymin'].astype('int')
        csv_total['bigxmax'] = csv_total['bigxmax'].astype('int')
        csv_total['bigymax'] = csv_total['bigymax'].astype('int')

    logging.info('===========================================================')
    logging.info('!!!merge bbox output')

    csv_total['post_status'] = True
    # post process
    if POST_PROCESS['switch']:
        for i in np.where(csv_total['class'] == 'yinlie')[0]:
            img_post = cv2.imread(os.path.join(INPUT_FOLDER, csv_total.loc[i, 'pic_name'] + '.jpg'), 0)
            bbox = [csv_total.loc[i, 'xmin'], csv_total.loc[i, 'ymin'], csv_total.loc[i, 'xmax'],
                    csv_total.loc[i, 'ymax']]
            post_status = yinlie_xizhi(img_post, PRE_PROCESS['size']['rows'], PRE_PROCESS['size']['cols'], bbox,
                                       yinlie_xizhi_thresh=2.5)
            csv_total.loc[i, 'post_status'] = post_status
        logging.info('===========================================================')
        logging.info('!!!finish post process')

    csv_total.to_csv(os.path.join(OUTPUT_FOLDER, 'csv', 'output_total.csv'), index=False)

    return csv_total


def auto_labeling(config_info, pic_shape, csv_total):
    """
    自动打标签
    :param config_info:
    :param pic_shape:
    :param csv_total:
    :return:
    """
    PRE_PROCESS = config_info.PRE_PROCESS
    POST_PROCESS = config_info.POST_PROCESS
    OUTPUT_FOLDER = config_info.OUTPUT_FOLDER

    if POST_PROCESS['switch']:
        filter_bbox = csv_total[csv_total['post_status'] == True].copy()
    else:
        filter_bbox = csv_total.copy()
    if PRE_PROCESS['switch']:
        xml_bbox = filter_bbox.loc[:, ['big_pic', 'bigxmin', 'bigymin', 'bigxmax', 'bigymax', 'class']]
    else:
        xml_bbox = filter_bbox.loc[:, ['pic_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class']]
    xml_bbox.columns = ['pic_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    csv2xml(xml_bbox, os.path.join(OUTPUT_FOLDER, 'xml'), pic_shape)
    logging.info('===========================================================')
    logging.info('!!!finish auto labeling')


def get_output_defect_info(result_df, defect_types, pre_process):
    """
    读取检测的输出结果
    :param result_df:
    :param defect_types:
    :param pre_process:
    :return:
    """
    res = []
    for i in defect_types:
        t = result_df[(result_df["class"] == i)]
        if len(t) > 0:
            res.append(t)
    defect_loc_df = pd.concat(res)

    # 有前处理
    if pre_process:
        defect_loc_df["loc"] = defect_loc_df["class"] + "_" \
                               + (defect_loc_df["bigxmin"]).astype(str) + "_" \
                               + (defect_loc_df["bigxmax"]).astype(str) + "_" \
                               + (defect_loc_df["bigymin"]).astype(str) + "_" \
                               + (defect_loc_df["bigymax"]).astype(str) + "_" \
                               + (defect_loc_df["score"]).astype(str) + "_" \
                               + (defect_loc_df["row"]).astype(str) + "_" \
                               + (defect_loc_df["col"]).astype(str)
        # 对loc按serial_number进行聚合
        defect_grouped = defect_loc_df['loc'].groupby(defect_loc_df['big_pic']).apply(lambda x: set(x))
    # 无前处理
    else:
        defect_loc_df["loc"] = defect_loc_df["class"] + "_" \
                               + (defect_loc_df["xmin"]).astype(str) + "_" \
                               + (defect_loc_df["xmax"]).astype(str) + "_" \
                               + (defect_loc_df["ymin"]).astype(str) + "_" \
                               + (defect_loc_df["ymax"]).astype(str) + "_" \
                               + (defect_loc_df["score"]).astype(str)
        defect_grouped = defect_loc_df['loc'].groupby(defect_loc_df['pic_name']).apply(lambda x: set(x))

    res = {}
    for s_n in defect_grouped.index:
        res[s_n] = list(defect_grouped[s_n])

    return res


def get_gold_defect_info(result_df, defect_types, pre_process):
    """
    读取答案csv
    :param result_df:
    :param defect_types:
    :param pre_process:
    :return:
    """
    res = []
    for i in defect_types:
        t = result_df[(result_df["class"] == i)]
        if len(t) > 0:
            res.append(t)
    defect_loc_df = pd.concat(res)

    # 有前处理
    if pre_process:
        defect_loc_df["loc"] = defect_loc_df["class"] + "_" \
                               + (defect_loc_df["row"]).astype(str) + "_" \
                               + (defect_loc_df["col"]).astype(str)
    # 无前处理
    else:
        defect_loc_df["loc"] = defect_loc_df["class"]

    # 对loc按serial_number进行聚合
    defect_grouped = defect_loc_df['loc'].groupby(defect_loc_df['pic_name']).apply(lambda x: set(x))
    res = {}
    for s_n in defect_grouped.index:
        res[s_n] = list(defect_grouped[s_n])

    return res


def draw_bbox_no_answer(img_path, save_path, result_loc):
    """
    无答案时绘制bbox
    :param img_path:
    :param save_path:
    :param result_loc:
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, type_loc in result_loc.items():
        # print(name, type_loc)
        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in type_loc:
                splited_str = i.split("_")
                defect_type, x1, x2, y1, y2, score = splited_str[:6]

                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (255, 0, 0), 3)
                cv2.putText(img_data, defect_type + ":" + score[:7], (x1 - 12, y1 - 30), font, 1.2, (255, 0, 0), 3)
            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)


def draw_bbox_with_answer_with_pre(img_path, save_path, result_loc, gold_loc, line_dict):
    """
    有答案，有前处理时，绘制bbox。
    :param img_path:
    :param save_path:
    :param result_loc: 检测结果（字典）
    :param gold_loc: 答案（字典）
    :param line_dict: 组件切分的行列号
    :return:
    """
    result_names = set(result_loc.keys())
    gold_names = set(gold_loc.keys())

    ok_names = result_names & gold_names
    miss_names = gold_names - result_names
    overkill_names = result_names - gold_names

    # 正确检出
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name in ok_names:
        r_info = result_loc[name]
        g_info = gold_loc[name]

        row_lines, col_lines = line_dict[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in r_info:
                defect_type, x1, x2, y1, y2, score, row, col = i.split("_")
                row, col, x1, x2, y1, y2 = int(row) + 1, int(col) + 1, int(x1), int(x2), int(y1), int(y2)

                if "_".join([defect_type, str(row), str(col)]) in g_info:
                    # 正常检出为绿色
                    cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 0), 3)
                    cv2.putText(img_data, score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 0), 3)
                else:
                    # 过检为黄色
                    cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 255), 3)
                    cv2.putText(img_data, score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 255), 3)

            for i in g_info:
                defect_type, row, col = i.split("_")
                row, col = int(row) - 1, int(col) - 1
                x1, y1, x2, y2 = col_lines[col], row_lines[row], col_lines[col + 1], row_lines[row + 1]

                temp = ["_".join(i.split("_")[:3]) for i in r_info]
                if "_".join([defect_type, str(row), str(col)]) not in temp:
                    # 漏检为红色框
                    cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 0, 255), 3)
                    cv2.putText(img_data, defect_type, (x1 - 12, y1 - 30), font, 1.2, (0, 0, 255), 3)

            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)

    # 过检
    for name in overkill_names:
        r_info = result_loc[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in r_info:
                defect_type, x1, x2, y1, y2, score, row, col = i.split("_")
                row, col, x1, x2, y1, y2 = int(row) + 1, int(col) + 1, int(x1), int(x2), int(y1), int(y2)

                # 过检为黄色
                cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 255), 3)
                cv2.putText(img_data, score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 255), 3)

            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)

    # 漏检
    for name in miss_names:
        row_lines, col_lines = line_dict[name]
        g_info = gold_loc[name]
        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in g_info:
                defect_type, row, col = i.split("_")
                row, col = int(row) - 1, int(col) - 1
                x1, y1, x2, y2 = col_lines[col], row_lines[row], col_lines[col + 1], row_lines[row + 1]
                # 漏检为红色
                cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 0, 255), 3)
                cv2.putText(img_data, defect_type, (x1 - 12, y1 - 30), font, 1.2, (0, 0, 255), 3)

            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)


def draw_bbox_with_answer_no_pre(img_path, save_path, result_loc, gold_loc):
    """
    有答案，无前处理时，绘制bbox
    :param img_path:
    :param save_path:
    :param result_loc: 检测结果（字典）
    :param gold_loc: 答案（字典）
    :return:
    """
    result_names = set(result_loc.keys())
    gold_names = set(gold_loc.keys())

    ok_names = result_names & gold_names
    miss_names = gold_names - result_names
    overkill_names = result_names - gold_names

    # 正确检出
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name in ok_names:
        r_info = result_loc[name]
        g_info = gold_loc[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in r_info:
                defect_type, x1, x2, y1, y2, score = i.split("_")
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                # 正确检出
                if defect_type in g_info:
                    cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 0), 3)
                    cv2.putText(img_data, defect_type + ":" + score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 0), 3)
                # 过检
                else:
                    cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 255), 3)
                    cv2.putText(img_data, defect_type + ":" + score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 255),
                                3)
            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)

            # 漏检
            for i in g_info:
                if i not in [j.split("_")[0] for j in r_info]:
                    shutil.copyfile(pic_file, os.path.join(save_path, name + "_miss_" + i + "_.jpg"))

    # 过检
    for name in overkill_names:
        r_info = result_loc[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in r_info:
                defect_type, x1, x2, y1, y2, score = i.split("_")
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 255), 3)
                cv2.putText(img_data, defect_type + ":" + score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 255), 3)

            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)

    # 漏检
    for name in miss_names:
        g_info = gold_loc[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            for i in g_info:
                shutil.copyfile(pic_file, os.path.join(save_path, name + "_miss_" + i + "_.jpg"))


def main(argv):
    """
    main function
    :param config_path: 配置文件路径
    :return:
    """
    # --------------------------------------------------------
    # 一、读取配置文件
    # 1.1 检查配置文件是否准确
    # 1.2 正确则读取配置文件，
    # 1.3 从配置文件中获取模型相关信息，主要是：缺陷类型，置信度，模型路径
    config_path = argv[1]
    cfgs = Config(config_path)
    if cfgs.is_valid:
        print("Please modify the configuration file!")
        exit(0)

    # --------------------------------------------------------
    # 1.4 读取测试图片的信息
    # 若需要预处理，则进行电池片切分
    pic_shape, line_dict = cfgs.get_pic_info()

    # --------------------------------------------------------
    # 二、测试
    # 2.1 将图片输入模型进行测试
    # 最终会生成csv文件，每个模型生成一个csv文件
    model_predict(cfgs)

    # 2.2 将多个模型的csv文件进行合并
    # 若存在前处理，则将大图信息也写入csv文件
    output_csv_total = merge_csv_files(cfgs, line_dict)

    # 三、自动打标签
    if cfgs.AUTO_LABELING['switch']:
        auto_labeling(cfgs, pic_shape, output_csv_total)

    # 四、画图
    # 缺陷类型
    defect_types = cfgs.types_and_ratios.keys()
    # 读取output_total.csv
    csv_path = os.path.join(cfgs.OUTPUT_FOLDER, 'csv', 'output_total.csv')
    result_df = pd.read_csv(csv_path)
    # 输出文件夹
    save_folder = os.path.join(cfgs.OUTPUT_FOLDER, "img_with_box")

    # 开始画图
    # 4.1 无答案
    if not cfgs.EVALUATION['switch']:
        # 无前处理
        if not cfgs.PRE_PROCESS['switch']:
            img_folder = cfgs.IMG_CUT_FOLDER
            result_loc = get_output_defect_info(result_df, defect_types, pre_process=False)
        # 有前处理
        else:
            img_folder = cfgs.PRE_PROCESS['origin_folder']
            result_loc = get_output_defect_info(result_df, defect_types, pre_process=True)
        # 绘图
        draw_bbox_no_answer(img_folder, save_folder, result_loc)
    # 4.2 有答案
    else:
        # 读取答案文件
        gold_df = pd.read_csv(cfgs.EVALUATION['csv_path'])

        # 无前处理
        if not cfgs.PRE_PROCESS['switch']:
            img_folder = cfgs.IMG_CUT_FOLDER
            result_loc = get_output_defect_info(result_df, defect_types, pre_process=False)
            gold_loc = get_gold_defect_info(gold_df, defect_types, pre_process=False)
            draw_bbox_with_answer_no_pre(img_folder, save_folder, result_loc, gold_loc)
        # 有前处理
        else:
            img_folder = cfgs.PRE_PROCESS['origin_folder']
            result_loc = get_output_defect_info(result_df, defect_types, pre_process=True)
            gold_loc = get_gold_defect_info(gold_df, defect_types, pre_process=True)
            draw_bbox_with_answer_with_pre(img_folder, save_folder, result_loc, gold_loc, line_dict)

    # 五、报告输出
    md_path = os.path.join(cfgs.OUTPUT_FOLDER, 'summary.md')
    with open(md_path, "w") as f_md:
        f_md.write(title_str())
        f_md.write(module_basic_info(cfgs))
        f_md.write(module_config_info(cfgs))
        f_md.write(module_results_info(cfgs, result_loc))
        # 若需要评估，则再增加最后部分
        if cfgs.EVALUATION['switch']:
            print(gold_loc)

    logging.info('===========================================================')
    logging.info('!!!finish write md file')


if __name__ == "__main__":
    # TODO
    # 1. 分文件夹
    #
    main(sys.argv)