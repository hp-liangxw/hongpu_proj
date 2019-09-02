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

sys.path.append('/detectron/tools')
from infer_simple import infer_simple


def get_pic_info(PRE_PROCESS, IMG_FOLDER):
    """
    get names and shapes of images
    :param PRE_PROCESS:
    :param IMG_FOLDER:
    :return:
    """
    logging.info('!!!getting pic information')

    # pre process
    pic_shape = {}

    # get pic names and shapes
    line_dict = {}
    if PRE_PROCESS['switch']:
        logging.info('!!!start pre_process')

        # 如果存在小图文件夹，先删除再新建
        if os.path.exists(IMG_FOLDER):
            shutil.rmtree(IMG_FOLDER)
        os.mkdir(IMG_FOLDER)

        # cut zujian pic
        for img in glob.glob(os.path.join(PRE_PROCESS['origin_folder'], '*.jpg')):
            img_name = os.path.basename(img).split('.')[0]
            img_data = cv2.imread(img)
            try:
                pic_shape[img_name] = img_data.shape[:2]
                cut_images, (row_lines, col_lines) = grid_cut(img_data, PRE_PROCESS['size']['rows'],
                                                              PRE_PROCESS['size']['cols'], False)
                line_dict[img_name] = (row_lines, col_lines)
                for i in range(len(row_lines) - 1):
                    for j in range(len(col_lines) - 1):
                        cv2.imwrite(os.path.join(IMG_FOLDER, "-".join([img_name, str(i), str(j)]) + '.jpg'),
                                    cut_images[row_lines[i], col_lines[j]])
            except:
                pass

        logging.info('===========================================================')
        logging.info('!!!finish pre_process')
    else:
        for img in glob.glob(os.path.join(IMG_FOLDER, '*.jpg')):
            img_name = os.path.basename(img).split('.')[0]
            img_data = cv2.imread(img)
            pic_shape[img_name] = img_data.shape[:2]

    return pic_shape, line_dict


def model_predict(MODEL, IMG_FOLDER, OUTPUT_FOLDER):
    """
    model predict
    :param MODEL:
    :param IMG_FOLDER:
    :param OUTPUT_FOLDER:
    :return:
    """
    for model_key, model_values in MODEL.items():
        model_dir = model_values['model_dir']
        threshold = model_values['threshold']
        infer_simple(IMG_FOLDER, os.path.join(OUTPUT_FOLDER, 'bbox'), model_dir, threshold)
    logging.info('===========================================================')
    logging.info('!!!finish model predict')


def output_csv(IMG_FOLDER, OUTPUT_FOLDER, PRE_PROCESS, POST_PROCESS, line_dict):
    """
    merge bbox output
    :param IMG_FOLDER:
    :param OUTPUT_FOLDER:
    :param PRE_PROCESS:
    :param POST_PROCESS:
    :param line_dict:
    :return:
    """
    # each defect class corresponds to a csv
    csv_total = pd.DataFrame(columns=['pic_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'score'])
    for csv_file in glob.glob(os.path.join(OUTPUT_FOLDER, 'bbox/*.csv')):
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
            img_post = cv2.imread(os.path.join(IMG_FOLDER, csv_total.loc[i, 'pic_name'] + '.jpg'), 0)
            bbox = [csv_total.loc[i, 'xmin'], csv_total.loc[i, 'ymin'], csv_total.loc[i, 'xmax'],
                    csv_total.loc[i, 'ymax']]
            post_status = yinlie_xizhi(img_post, PRE_PROCESS['size']['rows'], PRE_PROCESS['size']['cols'], bbox,
                                       yinlie_xizhi_thresh=2.5)
            csv_total.loc[i, 'post_status'] = post_status
        logging.info('===========================================================')
        logging.info('!!!finish post process')

    csv_total.to_csv(OUTPUT_FOLDER + '/bbox/output_total.csv', index=False)

    return csv_total


def auto_labeling(PRE_PROCESS, POST_PROCESS, OUTPUT_FOLDER, pic_shape, csv_total):
    os.mkdir(os.path.join(OUTPUT_FOLDER, 'xml'))
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


def get_result_defect_loc(result_df):
    """
    读取检测的输出结果
    :param result_df:
    :return:
    """
    defect_loc_df = result_df[
        (result_df["class"] == "yinlie") | (result_df["class"] == "shixiao") | (result_df["class"] == "xuhan")
        ]
    defect_loc_df = defect_loc_df.copy()

    if "big_pic" in defect_loc_df.columns:
        # 结果csv中存在big_pic字段，说明有post process
        # 将row和col组合成loc
        defect_loc_df["loc"] = defect_loc_df["class"] + "_" \
                               + (defect_loc_df["row"]).astype(str) + "_" \
                               + (defect_loc_df["col"]).astype(str) + "_" \
                               + (defect_loc_df["bigxmin"]).astype(str) + "_" \
                               + (defect_loc_df["bigxmax"]).astype(str) + "_" \
                               + (defect_loc_df["bigymin"]).astype(str) + "_" \
                               + (defect_loc_df["bigymax"]).astype(str) + "_" \
                               + (defect_loc_df["score"]).astype(str)
        # 对loc按serial_number进行聚合
        defect_grouped = defect_loc_df['loc'].groupby(defect_loc_df['big_pic']).apply(lambda x: set(x))
    else:
        # 不存在big_pic字段，说明不存在post process
        defect_loc_df["loc"] = defect_loc_df["class"] + "_" \
                               + (defect_loc_df["xmin"]).astype(str) + "_" \
                               + (defect_loc_df["xmax"]).astype(str) + "_" \
                               + (defect_loc_df["ymin"]).astype(str) + "_" \
                               + (defect_loc_df["ymax"]).astype(str) + "_" \
                               + (defect_loc_df["score"]).astype(str)
        defect_grouped = defect_loc_df['loc'].groupby(defect_loc_df['pic_name']).apply(lambda x: set(x))

    res = {}
    for s_n in defect_grouped.index:
        res[s_n] = defect_grouped[s_n]

    return res


def get_gold_defect_loc(result_df):
    """
    读取答案csv
    :param result_df:
    :return:
    """
    defect_loc_df = result_df[
        (result_df["class"] == "yinlie") | (result_df["class"] == "shixiao") | (result_df["class"] == "xuhan")
        ]
    defect_loc_df = defect_loc_df.copy()

    if "row" in defect_loc_df.columns:
        # 结果csv中存在big_pic字段，说明有post process
        # 将row和col组合成loc
        defect_loc_df["loc"] = defect_loc_df["class"] + "_" \
                               + (defect_loc_df["row"]).astype(str) + "_" \
                               + (defect_loc_df["col"]).astype(str)
    else:
        # 不存在big_pic字段，说明不存在post process
        defect_loc_df["loc"] = defect_loc_df["class"]

    # 对loc按serial_number进行聚合
    defect_grouped = defect_loc_df['loc'].groupby(defect_loc_df['pic_name']).apply(lambda x: set(x))
    res = {}
    for s_n in defect_grouped.index:
        res[s_n] = defect_grouped[s_n]

    return res


def draw_rectangle_0(img_path, save_path, result_loc):
    """
    无答案时绘制bbox
    :param img_path:
    :param save_path:
    :param result_loc:
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, type_loc in result_loc.items():
        # print(name, loc)
        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in type_loc:
                splited_str = i.split("_")
                if len(splited_str) == 8:
                    defect_type, row, col, x1, x2, y1, y2, score = splited_str
                else:
                    defect_type, x1, x2, y1, y2, score = splited_str

                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (255, 0, 0), 3)
                cv2.putText(img_data, score[:7], (x1 - 12, y1 - 30), font, 1.2, (255, 0, 0), 3)
            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)


def draw_rectangle_1(img_path, save_path, result_loc, gold_loc, line_dict):
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

    # draw pic
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name in ok_names:
        r_info = result_loc[name]
        g_info = gold_loc[name]

        row_lines, col_lines = line_dict[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in r_info:
                defect_type, row, col, x1, x2, y1, y2, score = i.split("_")
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

    for name in overkill_names:
        r_info = result_loc[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in r_info:
                defect_type, row, col, x1, x2, y1, y2, score = i.split("_")
                row, col, x1, x2, y1, y2 = int(row) + 1, int(col) + 1, int(x1), int(x2), int(y1), int(y2)

                # 过检为黄色
                cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 255), 3)
                cv2.putText(img_data, score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 255), 3)

            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)

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


def draw_rectangle_2(img_path, save_path, result_loc, gold_loc):
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

    # draw pic
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

                if defect_type in g_info:
                    cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 0), 3)
                    cv2.putText(img_data, score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 0), 3)
                else:
                    cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 255), 3)
                    cv2.putText(img_data, score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 255), 3)
            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)

            for i in g_info:
                if i not in [j.split("_")[0] for j in r_info]:
                    shutil.copyfile(pic_file, os.path.join(save_path, name + "_miss_" + i + "_.jpg"))

    for name in overkill_names:
        r_info = result_loc[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)

            for i in r_info:
                defect_type, x1, x2, y1, y2, score = i.split("_")
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                cv2.rectangle(img_data, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), (0, 255, 255), 3)
                cv2.putText(img_data, score[:7], (x1 - 12, y1 - 30), font, 1.2, (0, 255, 255), 3)

            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)

    for name in miss_names:
        g_info = gold_loc[name]

        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            for i in g_info:
                shutil.copyfile(pic_file, os.path.join(save_path, name + "_miss_" + i + "_.jpg"))


def check_config(yml_str):
    """
    确认配置文件正确性
    :param yml_str:
    :return:
    """
    print("\n>>>>>>>>----------------------------------------------------------------<<<<<<<<")
    print(yml_str)
    print(">>>>>>>>----------------------------------------------------------------<<<<<<<<\n")
    answer = raw_input("Is the configuration file correct? [yes/no]:")
    while answer.upper() not in ["YES", "Y", "NO", "N"]:
        logging.info("Please input yes or no!")
        answer = raw_input()

    if answer.upper() in ["YES", "Y"]:
        flag = 0
    else:
        flag = 1

    return flag


def main():
    """

    :return:
    """
    # 1 read and load test config file
    # 1.1 check config contents
    yml_str = open('./config/yjh.yaml').read()
    if check_config(yml_str):
        logging.info("Please modify the configuration file!")
        exit(0)
    # 1.2 load config
    test_cfg = yaml.load(yml_str)
    PRE_PROCESS = test_cfg['pre_process']
    IMG_CUT_FOLDER = test_cfg['img_cut_folder']
    OUTPUT_FOLDER = test_cfg['output_folder']
    MODEL = test_cfg['model']
    AUTO_LABELING = test_cfg['auto_labeling']
    POST_PROCESS = test_cfg['post_process']
    EVALUATION = test_cfg['evaluation']

    # create OUTPUT_FOLDER if not exists
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.mkdir(OUTPUT_FOLDER)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'bbox'))

    # get names and shapes of images
    pic_shape, line_dict = get_pic_info(PRE_PROCESS, IMG_CUT_FOLDER)

    # 2 make prediction
    model_predict(MODEL, IMG_CUT_FOLDER, OUTPUT_FOLDER)

    # get output csv file
    csv_total = output_csv(IMG_CUT_FOLDER, OUTPUT_FOLDER, PRE_PROCESS, POST_PROCESS, line_dict)

    # 3. auto labeling
    if AUTO_LABELING['switch']:
        auto_labeling(PRE_PROCESS, POST_PROCESS, OUTPUT_FOLDER, pic_shape, csv_total)

    # 4. draw box
    img_folder = PRE_PROCESS['origin_folder']
    save_folder = os.path.join(OUTPUT_FOLDER, "img_with_box")
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    csv_path = os.path.join(OUTPUT_FOLDER, 'bbox', 'output_total.csv')
    # 1. read output_total csv
    result_df = pd.read_csv(csv_path)
    result_loc = get_result_defect_loc(result_df)

    # 4.1 如果有答案，则读取答案
    gold_loc = {}
    if os.path.exists(EVALUATION['csv_path']):
        gold_df = pd.read_csv(EVALUATION['csv_path'])
        gold_loc = get_gold_defect_loc(gold_df)

    # 4.2 画图
    # 有前处理，无答案
    if PRE_PROCESS['switch'] and len(gold_loc) == 0:
        draw_rectangle_0(img_folder, save_folder, result_loc)
    # 有前处理，有答案
    elif PRE_PROCESS['switch'] and len(gold_loc) > 0:
        draw_rectangle_1(img_folder, save_folder, result_loc, gold_loc, line_dict)
    # 无前处理，无答案
    elif not PRE_PROCESS['switch'] and len(gold_loc) == 0:
        draw_rectangle_0(img_folder, save_folder, result_loc)
    # 无前处理，有答案
    else:
        draw_rectangle_2(img_folder, save_folder, result_loc, gold_loc)


if __name__ == "__main__":
    main()
