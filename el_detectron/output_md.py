# -*- coding: utf-8 -*-
import os
import datetime
import math
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def title_str():
    """
    大标题
    :return:
    """
    # markdown字符串
    title = """
# **测试总结报表**
"""
    return title


def module_basic_info(config_info):
    """
    第一部分：基本信息
    :param config_info:
    :return:
    """
    author = config_info.INFO["author"]
    report_version = config_info.INFO["report_version"]

    # markdown字符串
    module_str = """
|一、基本信息||
| ---- | ---- |
|测试人|{}|
|测试时间|{}|
|报表版本|{}|
<br>
    """.format(author, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), report_version)

    return module_str


def module_config_info(config_info):
    """
    第二部分：测试配置信息
    :param config_info:
    :return:
    """
    temp = []
    for t, r in config_info.types_and_ratios.items():
        temp.append(t + ": " + str(r))

    # markdown字符串
    module_str = """
|二、测试配置||
| ---- | ---- |
|是否前处理|{}|
|缺陷类型|{}|
|模型路径|{}|
|自动标签|{}|
|后处理|{}|
|评估|{}|
|答案|{}|
<br>
    """.format(
        config_info.PRE_PROCESS["switch"],
        "<br>".join(temp),
        "<br>".join(config_info.model_paths.values()),
        config_info.AUTO_LABELING["switch"],
        config_info.POST_PROCESS["switch"],
        config_info.EVALUATION["switch"],
        config_info.EVALUATION["csv_path"]
    )

    return module_str


def module_results_info(config_info, alg_results):
    """
    第三部分：
    1. 一共检测出多少缺陷，每种缺陷的占比，绘制饼图
    2. 每种缺陷的置信度分布直方图
    3. 缺陷分布热力图
    :param alg_results:
    :param config_info:
    :return:
    """
    # 缺陷类型
    defect_types = config_info.types_and_ratios.keys()

    d_info = {}
    # 绘制缺陷分布热力图
    if config_info.PRE_PROCESS['switch']:
        r = config_info.PRE_PROCESS['size']['rows']
        c = config_info.PRE_PROCESS['size']['cols']
        loc_matrix = np.zeros((r, c))  # 用于绘制热力图
        for _, defects in alg_results.items():
            for d in defects:
                d_type, _, _, _, _, score, row, col = d.split("_")
                loc_matrix[row, col] += 1
                if d_type in defect_types:
                    if d_type in d_info.keys():
                        d_info[d_type].append(score)
                    else:
                        d_info[d_type] = [score]
    else:
        loc_matrix = np.zeros((600, 600))  # 用于绘制热力图
        for _, defects in alg_results.items():
            for d in defects:
                d_type, x1, x2, y1, y2, score = d.split("_")
                loc_matrix[y1:y2, x1:x2] += 1
                if d_type in defect_types:
                    if d_type in d_info.keys():
                        d_info[d_type].append(score)
                    else:
                        d_info[d_type] = [score]

    # 绘制饼图
    plt.figure(figsize=(6, 6))
    values = []
    labels = []
    for i in defect_types:
        values.append(len(d_info[i]))
        labels.append(i)
    plt.pie(values, labels=labels, labeldistance=1.1, autopct='%2.0f%%', startangle=90, pctdistance=0.7)
    defects_ratio = os.path.join(config_info.OUTPUT_FOLDER, 'pics', "defects_ratio.jpg")
    plt.savefig(defects_ratio)

    # 绘制置信度直方图
    for i in defect_types:
        plt.figure(figsize=(6, 6))
        sns.distplot(d_info[i])
        plt.savefig(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "hist_{}.jpg".format(i)))

    # 绘制热力图
    plt.figure(figsize=(6, 6))
    sns.heatmap(loc_matrix, linewidths=0.05, vmax=math.ceil(np.max(loc_matrix) / 10) * 10, vmin=0, cmap='rainbow')
    heatmap_alg_results = os.path.join(config_info.OUTPUT_FOLDER, 'pics', "heatmap_alg_results.jpg")
    plt.savefig(heatmap_alg_results)

    # markdown字符串
    module_str = """
# 三、测试结果统计

1. 测试缺陷分布
    ![]({})

2. 置信度区间分布
    """.format(defects_ratio)

    for i in defect_types:
        module_str += """
    ![]({})
        """.format(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "hist_{}.jpg".format(i)))

    module_str += """<br>
3. 测试结果缺陷分布热力图
    ![]({})
    <br>
    """.format(heatmap_alg_results)

    return module_str


# ------------------------------------

def get_piclist_defect(datasets, defects, if_preprocess):
    """

    :param datasets:
    :param defects:
    :param if_preprocess:
    :return:
    """
    pic_list = []
    for _, rows in datasets.iterrows():
        if rows['class'] in defects:
            # 按照大图
            if not if_preprocess:
                pic_list.append(rows['pic_name'])
            # 按照小图
            else:
                pic_list.append(''.join([rows['pic_name'], '-', str(rows['row']), '-', str(rows['col'])]))
    return set(pic_list)


def select_defect_thresh(select_defect, defect_dict, standard_testsets, csv_bbox,
                         step, miss_num, overkill_num, if_preprocess):
    """
    根据测试集选择单个缺陷置信度
    param:
        select_defect: 待测缺陷, str
        defect_dict-->config_info.types_and_ratios
        standard_testsets
        csv_bbox
        step: 查找置信度的步长, float
        miss_rate: 限制最多漏检比例, float
        overkill_rate: 限制最多漏检比例, float
    """
    total_missover = []
    start_th = defect_dict[select_defect]
    answer_defect = get_piclist_defect(standard_testsets, [select_defect], if_preprocess)
    for tmp_th in np.linspace(start_th, 1, int(round((1 - start_th) / step)), endpoint=False):
        tmp_defect = get_piclist_defect(csv_bbox[csv_bbox['score'] >= tmp_th], [select_defect], if_preprocess)
        miss_defect = answer_defect - tmp_defect
        over_defect = tmp_defect - answer_defect
        total_missover.append([tmp_th, len(miss_defect), len(over_defect), len(miss_defect) + len(over_defect)])
    total_missover = pd.DataFrame(total_missover)
    total_missover.columns = ['defect_th', 'miss_num', 'overkill_num', 'ng_num']
    filter_missover = total_missover[
        (total_missover['miss_num'] < miss_num) & (total_missover['overkill_num'] < overkill_num)]
    if len(filter_missover) != 0:
        defect_th = filter_missover['defect_th'][np.argmin(filter_missover['ng_num'])]
    else:
        defect_th = start_th

    return defect_th, total_missover


def module_evaluation(config_info, ground_truth, csv_bbox):
    """
    第四部分，评估测试结果
    1. 答案基本情况。多少组件，多少缺陷。
    2. 各个置信度下过检漏检情况，折线图。
    3. 最终选择的置信度下，过检漏检情况，统计表。
    4.
    :return:
    """
    # 缺陷类型
    defect_types = config_info.types_and_ratios.keys()

    info = {}
    pic_list = []
    if config_info.PRE_PROCESS['switch']:
        r = config_info.PRE_PROCESS['size']['rows']
        c = config_info.PRE_PROCESS['size']['cols']
        loc_matrix = np.zeros((r, c))  # 用于绘制热力图
        for names, defects in ground_truth.items():
            for d in defects:
                d_type, row, col = d.split("_")
                loc_matrix[row, col] += 1
                pic_list.append("-".join([names, row, col]))
                if d_type in defect_types:
                    if d_type in info.keys():
                        info[d_type] += 1
                    else:
                        info[d_type] = 1

        # 绘制热力图
        plt.figure(figsize=(6, 6))
        sns.heatmap(loc_matrix, linewidths=0.05, vmax=math.ceil(np.max(loc_matrix) / 10) * 10, vmin=0, cmap='rainbow')
        heatmap_ground_truth = os.path.join(config_info.OUTPUT_FOLDER, 'pics', "heatmap_ground_truth.jpg")
        plt.savefig(heatmap_ground_truth)

        # 选择合适置信度
        step = 0.001
        miss_num = 10
        overkill_num = 1000000000
        if_preprocess = config_info.PRE_PROCESS['switch']
        standard_testsets = pd.read_csv(config_info.EVALUATION['csv_path'])
        dtype_ratio = {}
        total_missover = None
        for select_defect in defect_types:
            defect_th, total_missover = select_defect_thresh(select_defect, config_info.types_and_ratios,
                                                             standard_testsets, csv_bbox, step,
                                                             miss_num, overkill_num, if_preprocess)
            dtype_ratio[select_defect] = defect_th

        plt.figure(figsize=(6, 6))
        plt.plot(total_missover['defect_th'].values, total_missover['miss_num'].values, label='miss_num')
        plt.plot(total_missover['defect_th'].values, total_missover['overkill_num'].values, label='overkill_num')
        plt.plot(total_missover['defect_th'].values, total_missover['ng_num'].values, label='ng_num')
        line_threshold = os.path.join(config_info.OUTPUT_FOLDER, 'pics', "line_threshold.jpg")
        plt.savefig(line_threshold)

    else:
        for names, defects in ground_truth.items():
            pic_list.append(names)
            for d in defects:
                d_type = d.split("_")[0]
                if d_type in defect_types:
                    if d_type in info.keys():
                        info[d_type] += 1
                    else:
                        info[d_type] = 1

    # 答案中总组件个数
    print(len(info))
    # 答案中各个缺陷的个数
    for i in defect_types:
        print(info[i])

