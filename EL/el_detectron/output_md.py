# -*- coding: utf-8 -*-
import os
import datetime
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def report_title():
    """
    大标题
    :return:
    """
    # markdown字符串
    title = """
# **测试总结报表**
"""
    return title


def basic_info(config_info):
    """
    第一部分：基本信息
    包括测试版本以及测试配置
    :param config_info:
    :return:
    """
    author = config_info.INFO["author"]
    report_version = config_info.INFO["report_version"]

    # (1) 版本信息
    module_str = """
## 一、基本信息

|版本信息||
| ---- | ---- |
|测试人|{}|
|测试时间|{}|
|报表版本|{}|
<br>
    """.format(author, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), report_version)

    # (2) 测试配置信息
    # # 模型路径与缺陷类型
    model_strs = ""
    for i in config_info.model_infos.values():
        model_strs += i["model_dir"]
        for d_name, thr in i['threshold'].items():
            model_strs += "<br>" + d_name + ": " + str(thr)
        model_strs += "<br>"
    # # 预处理信息
    pre_strs = ""
    if config_info.PRE_PROCESS["switch"]:
        for i, j in config_info.PRE_PROCESS["size"].items():
            pre_strs += i + ": " + str(j) + " "
    else:
        pre_strs = "/"

    module_str += """
|测试配置||
| ---- | ---- |
|是否预处理|{}|
|预处理参数|{}|
|模型与缺陷|{}|
|自动打标签|{}|
|是否后处理|{}|
|是否评估|{}|
|是否有答案|{}|
<br>
    """.format(
        config_info.PRE_PROCESS["switch"],
        pre_strs,
        model_strs,
        config_info.AUTO_LABELING["switch"],
        config_info.POST_PROCESS["switch"],
        config_info.EVALUATION["switch"],
        config_info.EVALUATION["csv_path"]
    )

    return module_str


def module_results_info(config_info, inner, outer, scores, loc_matrix):
    """
    第二部分 测试结果信息：
    1. 一共检测出多少缺陷，每种缺陷的占比，绘制饼图
    2. 每种缺陷的置信度分布直方图
    3. 缺陷分布热力图
    :param config_info:
    :param inner:
    :param outer:
    :param scores:
    :param loc_matrix:
    :return:
    """
    # 所有缺陷类型
    all_defect_types = []
    for i in config_info.model_infos.values():
        all_defect_types.extend(i['threshold'].keys())

    # 检测到的缺陷类型
    detected_types = []
    each_defect_num = {}
    for i in outer:
        if outer[i] > 0:
            rei = i.replace("multi_", "")
            detected_types.append(rei)
            if rei in each_defect_num.keys():
                each_defect_num[rei] += outer[i]
            else:
                each_defect_num[rei] = outer[i]
    detected_types = set(detected_types)

    # 1. 饼图
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(1, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[0, -1])
    cmap1 = plt.get_cmap("Pastel1")
    cmap2 = plt.get_cmap("tab20c")

    # 绘制饼图的val，label，color
    outer_val = []
    outer_label = []
    outer_color = []
    inner_val = []
    inner_label = []
    inner_color = []

    if inner['none'] > 0:
        inner_val.append(inner['none'])
        inner_label.append(inner['none'])
        inner_color.append(11)
        outer_val.append(inner['none'])
        outer_label.append('')
        outer_color.append(19)

    # 各个缺陷的颜色映射表
    color_map = {}
    for ind, i in enumerate(detected_types):
        if ind >= 2:  # 2为内图的颜色
            ind = ind + 1
        color_map[i] = ind

    for i in detected_types:
        if inner[i] > 0:
            inner_val.append(inner[i])
            inner_label.append(inner[i])
            inner_color.append(11)
            outer_val.append(inner[i])
            outer_label.append(i + ":" + str(outer[i]))
            outer_color.append(color_map[i])

    if inner["multi"] > 0:
        n = 0
        for i in detected_types:
            if outer["multi_" + i] > 0:
                outer_val.append(outer["multi_" + i])
                outer_label.append(outer["multi_" + i])
                outer_color.append(color_map[i])
                n += outer["multi_" + i]
        inner_val.append(n)
        inner_label.append(inner["multi"])
        inner_color.append(11)

    # 开始绘制
    ax1.pie(outer_val, labels=outer_label, colors=cmap1(outer_color), labeldistance=0.75, radius=1,
            wedgeprops=dict(edgecolor='w'))
    ax1.pie(inner_val, labels=inner_label, colors=cmap2(inner_color), labeldistance=0.66, radius=0.66,
            wedgeprops=dict(edgecolor='w'))
    ax1.pie([1], radius=0.33, colors='w')
    # 绘制colorbar
    # 第一个0.5参数表示垂直高度，第二个和第三个参数0.7、0.9表示色块的长度。参照系是xlim和ylim的长度。
    ax2.hlines(0.5, 0.7, 0.9, color=cmap2(11), linewidth=12)
    ax2.text(0.95, 0.5, "input", fontsize=10)
    ct = 1
    for i in color_map:
        ax2.hlines(0.5 - 0.05 * ct, 0.7, 0.9, colors=cmap1(color_map[i]), linewidth=12)
        ax2.text(0.95, 0.5 - 0.05 * ct, i, fontsize=10)
        ct += 1
    if 19 in outer_color:
        ax2.hlines(0.5 - 0.05 * ct, 0.7, 0.9, colors=cmap1(19), linewidth=12)
        ax2.text(0.95, 0.5 - 0.05 * ct, "none", fontsize=10)
        ct += 1
    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.set_axis_off()
    ax2.set_xlim([0.7, 1])
    ax2.set_ylim([0, 1 - (ct + 1 - 2) * 0.05])
    plt.savefig(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "defects_ratio.jpg"))

    # 2. 绘制置信度直方图
    for i in detected_types:
        plt.figure(figsize=(6, 6))
        sns.distplot(scores[i])
        plt.savefig(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "hist_{}.jpg".format(i)))

    # 3. 绘制热力图
    [r, c] = loc_matrix.shape
    plt.figure(figsize=(c / 100, r / 100))
    plt.pcolor(loc_matrix, cmap=plt.cm.Reds)
    plt.xticks([])
    plt.yticks([])
    # plt.grid("True")
    plt.savefig(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "heatmap_alg_results.jpg"))

    # --------------------
    # markdown字符串
    module_str = """
## 二、测试结果统计

共输入**{}**张测试图片<br>
检测出**{}**张图片有缺陷

1. 测试缺陷分布

    |缺陷名称|检测个数|
    | --- | --- |""".format(sum(inner.values()), sum(inner.values()) - inner['none'])

    for i in detected_types:
        module_str += """
    |{}|{}|""".format(i, each_defect_num[i])

    module_str += """

    ![]({})

2. 置信度区间分布
    """.format(os.path.join('pics', "defects_ratio.jpg"))

    for i in detected_types:
        module_str += """
    ![]({})""".format(os.path.join('pics', "hist_{}.jpg".format(i)))

    module_str += """

3. 测试结果缺陷分布热力图
    ![]({})
<br>
    """.format(os.path.join('pics', "heatmap_alg_results.jpg"))

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
        defect_dict
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


def pre_evaluation(standard_testsets, filter_csv_bbox, defect_name):
    # 大图松
    answer_defect1 = get_piclist_defect(standard_testsets, defect_name, False)
    paper_defect1 = get_piclist_defect(filter_csv_bbox, defect_name, False)
    miss_list1 = answer_defect1 - paper_defect1
    overkill_list1 = paper_defect1 - answer_defect1
    true_list1 = answer_defect1 & paper_defect1
    # 小图
    answer_defect2 = get_piclist_defect(standard_testsets, defect_name, True)
    paper_defect2 = get_piclist_defect(filter_csv_bbox, defect_name, True)
    miss_list2 = answer_defect2 - paper_defect2
    overkill_list2 = paper_defect2 - answer_defect2
    true_list2 = answer_defect2 & paper_defect2
    # 大图紧
    miss_list3 = set([miss_pic.split('-')[0] for miss_pic in miss_list2])
    overkill_list3 = set([over_pic.split('-')[0] for over_pic in overkill_list2])
    true_list3 = answer_defect1 - miss_list3 - overkill_list3

    return miss_list1, overkill_list1, true_list1, miss_list2, overkill_list2, true_list2, miss_list3, overkill_list3, true_list3


def module_evaluation(config_info, all_defect_types, inner, outer, gold_loc, result_df):
    """
    第四部分，评估测试结果
    1. 答案基本情况。多少组件，多少缺陷。
    2. 各个置信度下过检漏检情况，折线图。
    3. 最终选择的置信度下，过检漏检情况，统计表。
    4.
    :return:
    """
    # 检测到的缺陷类型与数量
    detected_types = []
    each_defect_num = {}
    for i in outer:
        if outer[i] > 0:
            name = i.split("_")[-1]
            detected_types.append(name)
            if name in each_defect_num.keys():
                each_defect_num[name] += 1
            else:
                each_defect_num[name] = 1
    detected_types = set(detected_types)

    # 1. 饼图
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(1, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[0, -1])
    cmap1 = plt.get_cmap("Pastel1")
    cmap2 = plt.get_cmap("tab20c")

    # 绘制饼图的val，label，color
    outer_val = []
    outer_label = []
    outer_color = []
    inner_val = []
    inner_label = []
    inner_color = []

    # 各个缺陷的颜色映射表
    color_map = {}
    for ind, i in enumerate(detected_types):
        if ind >= 2:  # 2为内图的颜色
            ind = ind + 1
        color_map[i] = ind

    # 过检
    n1 = 0
    for k in detected_types:
        if outer["over_" + k] > 0:
            outer_val.append(outer["over_" + k])
            outer_label.append(outer["over_" + k])
            outer_color.append(color_map[k])
            n1 += outer["over_" + k]
    if n1 > 0:
        inner_val.append(n1)
        inner_label.append(inner['over'])
        inner_color.append(16)

    # 漏检
    n2 = 0
    for k in detected_types:
        if outer["miss_" + k] > 0:
            outer_val.append(outer["miss_" + k])
            outer_label.append(outer["miss_" + k])
            outer_color.append(color_map[k])
            n2 += outer["miss_" + k]
    if n2 > 0:
        inner_val.append(n2)
        inner_label.append(inner['miss'])
        inner_color.append(4)

    # 正确检出
    n3 = 0
    for k in detected_types:
        if outer["ok_" + k] > 0:
            outer_val.append(outer["ok_" + k])
            outer_label.append(k + ":" + str(outer["ok_" + k]))
            outer_color.append(color_map[k])
            n3 += outer["ok_" + k]
    if n3 > 0:
        inner_val.append(n3)
        inner_label.append(inner['ok'])
        inner_color.append(8)

    # 其它
    n4 = 0
    for k in detected_types:
        if outer["other_" + k] > 0:
            outer_val.append(outer["other_" + k])
            outer_label.append(outer["other_" + k])
            outer_color.append(color_map[k])
            n4 += outer["other_" + k]
    if n4 > 0:
        inner_val.append(n4)
        inner_label.append(inner['other'])
        inner_color.append(0)

    # 开始绘制
    ax1.pie(outer_val, labels=outer_label, colors=cmap1(outer_color), labeldistance=0.75, radius=1,
            wedgeprops=dict(edgecolor='w'))
    ax1.pie(inner_val, labels=inner_label, colors=cmap2(inner_color), labeldistance=0.66, radius=0.66,
            wedgeprops=dict(edgecolor='w'))
    ax1.pie([1], radius=0.33, colors='w')
    # 绘制colorbar
    # 第一个0.5参数表示垂直高度，第二个和第三个参数0.7、0.9表示色块的长度。参照系是xlim和ylim的长度。
    ct = 1
    if n1 > 0:
        ax2.hlines(0.5 + 0.05 * ct, 0.7, 0.9, color=cmap2(16), linewidth=12)
        ax2.text(0.95, 0.5 + 0.05 * ct, "overkill", fontsize=10)
        ct += 1
    if n2 > 0:
        ax2.hlines(0.5 + 0.05 * ct, 0.7, 0.9, color=cmap2(4), linewidth=12)
        ax2.text(0.95, 0.5 + 0.05 * ct, "miss", fontsize=10)
        ct += 1
    if n3 > 0:
        ax2.hlines(0.5 + 0.05 * ct, 0.7, 0.9, color=cmap2(8), linewidth=12)
        ax2.text(0.95, 0.5 + 0.05 * ct, "normal", fontsize=10)
        ct += 1
    if n4 > 0:
        ax2.hlines(0.5 + 0.05 * ct, 0.7, 0.9, color=cmap2(0), linewidth=12)
        ax2.text(0.95, 0.5 + 0.05 * ct, "other", fontsize=10)
        ct += 1
    ct = 0
    for i in color_map:
        ax2.hlines(0.5 - 0.05 * ct, 0.7, 0.9, colors=cmap1(color_map[i]), linewidth=12)
        ax2.text(0.95, 0.5 - 0.05 * ct, i, fontsize=10)
        ct += 1
    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.set_axis_off()
    ax2.set_xlim([0.7, 1])
    ax2.set_ylim([0, 1 - (ct + 1 - 2) * 0.05])
    plt.savefig(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "golden_defects_ratio.jpg"))

    # 所有缺陷类型

    # 答案中的缺陷分布
    gold_defects = {}
    for v in gold_loc.values():
        for w in v:
            name = w.split("_")[0]
            if name in gold_defects:
                gold_defects[name] += 1
            else:
                gold_defects[name] = 1

    # --------------------
    # 选择合适置信度
    types_and_ratios = {}
    for i in config_info.model_infos.values():
        for j in i["threshold"]:
            types_and_ratios[j] = i["threshold"][j]
    if config_info.EVALUATION['select_th']['switch'] is True:
        step = config_info.EVALUATION['select_th']['step']
        miss_num = config_info.EVALUATION['select_th']['miss_number']
        overkill_num = config_info.EVALUATION['select_th']['overkill_number']
        if_preprocess = config_info.PRE_PROCESS['switch']
        standard_testsets = pd.read_csv(config_info.EVALUATION['csv_path'])

        csv_bbox = result_df.copy()
        csv_bbox.loc[csv_bbox['class'] == 'shixiao', 'class'] = 'yinlie'
        dtype_ratio = {}
        total_missover = None
        for select_defect in all_defect_types:
            defect_th, total_missover = select_defect_thresh(select_defect, types_and_ratios,
                                                             standard_testsets, csv_bbox, step,
                                                             miss_num, overkill_num, if_preprocess)
            dtype_ratio[select_defect] = defect_th

        print(total_missover)
        plt.figure(figsize=(6, 6))
        length = len(total_missover['defect_th'].values)

        x = range(1, length + 1)
        x_label = [str(i) for i in total_missover['defect_th'].values]
        y1 = total_missover['miss_num'].values
        y2 = total_missover['overkill_num'].values
        y3 = total_missover['ng_num'].values

        plt.bar(x, y1, align="center", tick_label=x_label, color=cmap1(1), label="miss")
        plt.bar(x, y2, align="center", bottom=y1, color=cmap1(2), label="overkill")
        plt.bar(x, y3, align="center", bottom=y1 + y2, color=cmap1(4), label="ng")
        plt.plot(range(-1, length + 2), [config_info.EVALUATION['select_th']['miss_number']] * (length + 3), 'r--',
                 label="maximum of missed")
        plt.xlabel("threshold")
        plt.ylabel("number")
        plt.xlim(0, length + 1)

        plt.legend()
        plt.savefig(os.path.join(config_info.OUTPUT_FOLDER, 'pics', "line_threshold.jpg"))

    # --------------------
    # markdown字符串
    module_str = """
## 三、测试评估

1. 答案基本情况

    答案中总图片数量: {}<br>
    答案中各缺陷数量：

    |缺陷名称|缺陷个数|
    | --- | --- |""".format(len(gold_loc))

    for i in gold_defects:
        module_str += """
    |{}|{}|""".format(i, gold_defects[i])

    module_str += """

    ![]({})""".format(os.path.join('pics', "golden_defects_ratio.jpg"))

    module_str += """

2. 不同置信度下的过检与漏检
    
    ![]({})  
""".format(os.path.join('pics', "line_threshold.jpg"))

    return module_str
