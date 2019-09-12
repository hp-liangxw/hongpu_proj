# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import yaml


def title_str():
    """

    :return:
    """
    title = """
# **测试总结报表**
"""
    return title


def module_basic_info(config_info):
    """

    :param config_info:
    :return:
    """
    author = config_info.INFO["author"]
    report_version = config_info.INFO["report_version"]

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

    :param config_info:
    :return:
    """
    temp = []
    for t, r in config_info.types_and_ratios.items():
        temp.append(t + ": " + str(r))

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


# # csv_path = os.path.join(OUTPUT_FOLDER, 'bbox', 'output_total.csv')
# result_csv_path = r"D:\Desktop\gpu2_test\output_total.csv"
# result_df = pd.read_csv(result_csv_path)
# result_loc = get_result_defect_loc(result_df)

# n = {}
# for _, infos in result_loc.items():
#     for i in infos:
#         t = i.split("_")[0]  # 缺陷名称
#         if t in n.keys():
#             n[t] += 1
#         else:
#             n[t] = 1
#     x = []
#     y = []
#     for k, v in n.items():
#         x.append(k)
#         y.append(v)
#     plt.pie(y, labels=x, autopct='%3.1f%%')
#     plt.savefig("1.jpg")
#     print("检测出有缺陷的电池片的数量：{}".format(n))

# answer_path = os.path.exists(EVALUATION['csv_path'])
# if PRE_PROCESS["switch"]:  # 有前处理
#     print("一共{}个组件".format(len(os.listdir(ORIGINAL_FOLDER))))
#     print("一共检测出{}个有缺陷的组件".format(len(result_loc)))
# else:  # 无前处理
#     pass
#
#
# if test_cfg["evaluation"]["switch"]:  # 需要进行评估时才会继续进行
#
#
#     if not os.path.exists(answer_path):  # 无答案
#         pass
#     else:  # 有答案
#         pass
#
#
#
# else:  # 不需要评估
#     pass

# def select_defect_thresh(select_defect, defect_dict, standard_testsets, csv_bbox, step, miss_num, overkill_num,
#                          if_preprocess):
#     """
#     根据测试集选择单个缺陷置信度
#     param:
#         select_defect: 待测缺陷, str
#         defect_dict
#         standard_testsets
#         csv_bbox
#         step: 查找置信度的步长, float
#         miss_rate: 限制最多漏检比例, float
#         overkill_rate: 限制最多漏检比例, float
#     """
#     total_missover = []
#     start_th = defect_dict[select_defect]
#     answer_defect = get_piclist_defect(standard_testsets, [select_defect], if_preprocess)
#     for tmp_th in np.linspace(start_th, 1, int(round((1 - start_th) / step)), endpoint=False):
#         tmp_defect = get_piclist_defect(csv_bbox[csv_bbox['score'] >= tmp_th], [select_defect], if_preprocess)
#         miss_defect = answer_defect - tmp_defect
#         over_defect = tmp_defect - answer_defect
#         total_missover.append([tmp_th, len(miss_defect), len(over_defect), len(miss_defect) + len(over_defect)])
#     total_missover = pd.DataFrame(total_missover)
#     total_missover.columns = ['defect_th', 'miss_num', 'overkill_num', 'ng_num']
#     filter_missover = total_missover[
#         (total_missover['miss_num'] < miss_num) & (total_missover['overkill_num'] < overkill_num)]
#     if len(filter_missover) != 0:
#         defect_th = filter_missover['defect_th'][np.argmin(filter_missover['ng_num'])]
#     else:
#         defect_th = start_th
#
#     return defect_th, total_missover


# ----------------------------------------------------------------------------------------------------------------------
# print(module_1)
# with open("test.md", "w", encoding="utf-8") as f:
#     f.write(title)
#     f.write(module_1)
#     f.write(module_2)
