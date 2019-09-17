# -*- coding: utf-8 -*-
import os
import datetime
import seaborn as sns
from matplotlib import pyplot as plt


def title_str():
    """
    大标题
    :return:
    """
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


def module_results_info(config_info, results):
    """
    第三部分：
    1. 一共检测出多少缺陷，每种缺陷的占比，绘制饼图
    2. 每种缺陷的置信度分布直方图
    3. 热力图(TO DO)
    :param config_info:
    :return:
    """
    # 缺陷类型
    defect_types = config_info.types_and_ratios.keys()
    # loc_matrix = np.zeros((6, 12))  # 用于绘制热力图

    info = {}
    for _, defects in results.items():
        for d in defects:
            d_type, _, _, _, _, score = d.split("_")[:6]
            if d_type in defect_types:
                if d_type in info.keys():
                    info[d_type].append(score)
                else:
                    info[d_type] = [score]

    # 绘制饼图
    plt.figure(figsize=(6, 6))
    values = []
    labels = []
    for i in defect_types:
        values.append(len(info[i]))
        labels.append(i)
    plt.pie(values, labels=labels, labeldistance=1.1, autopct='%2.0f%%', startangle=90, pctdistance=0.7)
    plt.savefig(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "defects_ratio.jpg"))

    # 绘制置信度直方图
    for i in defect_types:
        plt.figure(figsize=(6, 6))
        sns.distplot(info[i])
        plt.savefig(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "hist_{}.jpg".format(i)))

    module_str = """
三、测试结果统计

1. 测试缺陷分布
    ![]({})

2. 置信度区间分布
    """.format(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "defects_ratio.jpg"))

    for i in defect_types:
        module_str += """
    ![]({})
        """.format(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "hist_{}.jpg".format(i)))

    module_str += """<br>"""

    return module_str

