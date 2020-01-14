import os
import pandas as pd
import cv2
from EL.jin_neng.cut_zujian_pic import grid_cut


def get_defect_loc(standard_df, defect_type):
    if defect_type == "yinlie":
        defect_loc = standard_df[(standard_df["缺陷名"] == "yinlie") | (standard_df["缺陷名"] == "shixiao")]
    else:
        defect_loc = standard_df[standard_df["缺陷名"] == defect_type]
    defect_loc = defect_loc.copy()
    # 将row和col组合成loc
    defect_loc["loc"] = (defect_loc["行"]).astype(str) + "_" + (defect_loc["列"]).astype(str)
    # 对loc按serial_number进行聚合
    defect_grouped = defect_loc['loc'].groupby(defect_loc['序列号']).apply(lambda x: list(x))

    res = {}
    for s_n in defect_grouped.index:
        res[s_n] = defect_grouped[s_n]
    return res


def location_correction(defect_loc):
    """
    客户端检测csv文件中的行列号存在异常，对其进行修复
    :param defect_loc:
    :return:
    """
    res = {}
    for name, loc in defect_loc.items():
        loc = set(loc)

        loc_corrected = []
        for i in loc:
            r, c = i.split("_")
            r, c = 6 - int(r), int(c) - 1
            loc_corrected.append("_".join([str(r), str(c)]))

        res[name] = loc_corrected

    return res


def loc_transfer_2_dcp(location):
    """
    将组件级别的缺陷，转换为电池片级别
    {zj_name1: ['1_2', '3_5']} --> ['zj_name1=1_2', 'zj_name1=3_5']
    :param location:
    :return:
    """
    res = []
    for name, loc in location.items():
        for i in loc:
            res.append(name + "=" + i)
    return res


#
jinneng_dir = r"D:\OneDrive\MyDrive\OneDrive\Projects\jinneng\standard_testsets"
test_csv = r"D:\OneDrive\MyDrive\OneDrive\Projects\jinneng\model_evaluation\flaw_statistic_0515_yl.v3_xh.v4.csv"
save_dir = r"D:\OneDrive\MyDrive\OneDrive\Projects\jinneng\model_evaluation\0515_yl.v3_xh.v4"
"""--------------------------------------------------------------------------------------------------------
标准答案
目前只有yinlie和xuhan
--------------------------------------------------------------------------------------------------------"""
# 有缺陷的组件
gt_df = pd.read_csv(os.path.join(jinneng_dir, "defect_location.csv"))
gt_zj_list = set(gt_df["序列号"].tolist())
print("测试集中有缺陷的组件数：", len(gt_zj_list))

# 获取各缺陷类型的位置
gt_xuhan_loc = get_defect_loc(gt_df, "xuhan")
gt_yinlie_loc = get_defect_loc(gt_df, "yinlie")
print("    xuhan缺陷的组件数：", len(gt_xuhan_loc))
print("    yinlie缺陷的组件数：", len(gt_yinlie_loc))
print("    有{}个组件既有xuhan也有yinlie".format(len(gt_yinlie_loc) + len(gt_xuhan_loc) - len(gt_zj_list)))

"""--------------------------------------------------------------------------------------------------------
算法检测结果
--------------------------------------------------------------------------------------------------------"""
test_df = pd.read_csv(test_csv)
# 总测试组件
test_zj_list = set(test_df["序列号"].tolist())
print("--------------------------------------")
print("总测试组件数：", len(test_zj_list))
# 测试组件中，有缺陷的组件数
detected_df = test_df[(test_df["缺陷名"] == "yinlie") |
                      (test_df["缺陷名"] == "shixiao") |
                      (test_df["缺陷名"] == "xuhan")]
detected_zj_list = set(detected_df["序列号"].tolist())
# print("检测出有缺陷的组件数：", len(detected_zj_list))

detected_xuhan_loc = get_defect_loc(test_df, "xuhan")
detected_yinlie_loc = get_defect_loc(test_df, "yinlie")
# 由于客户端bug，对位置进行校正
detected_xuhan_loc = location_correction(detected_xuhan_loc)
detected_yinlie_loc = location_correction(detected_yinlie_loc)
# print("    其中检测出有xuhan缺陷的组件数：", len(detected_xuhan_loc))
# print("    其中检测出有yinlie缺陷的组件数：", len(detected_yinlie_loc))

"""--------------------------------------------------------------------------------------------------------
结果统计结果
--------------------------------------------------------------------------------------------------------"""
# 1. 组件级别
right_zj_n = len(set(gt_zj_list) & set(detected_zj_list))
overkill_zj_n = len(set(detected_zj_list) - set(gt_zj_list))
miss_zj_n = len(set(gt_zj_list) - set(detected_zj_list))
print("检测出缺陷组件数：", right_zj_n)
print("过检组件数：", overkill_zj_n)
print("漏检组件数：", miss_zj_n)
print("漏检组件：", set(gt_zj_list) - set(detected_zj_list))
print("过检率：{}/{}={}".format(overkill_zj_n, len(test_zj_list), overkill_zj_n / len(test_zj_list)))
print("漏检率：{}/{}={}".format(miss_zj_n, len(test_zj_list), miss_zj_n / len(test_zj_list)))
print("--------------------------------------")


# 2. 电池片级别
def draw_rectangle(proj_dir, save_path, overkill_or_miss, draw_type, defect_type):
    """
    对overkill或者miss进行可视化
    :param overkill_or_miss:
    :return:
    """
    res_list = {}
    for over in overkill_or_miss:
        img_name, row_col = over.split("=")
        if img_name in res_list.keys():
            res_list[img_name].append(row_col)
        else:
            res_list[img_name] = [row_col]

    for name, loc in res_list.items():
        # print(name, loc)
        pic_file = os.path.join(proj_dir, "all", name + ".jpg")
        img_data = cv2.imread(pic_file)
        _, (row_lines, col_lines) = grid_cut(img_data, 6, 12, edge_removed=False)
        for i in loc:
            row, col = i.split("_")
            row, col = int(row), int(col)
            if draw_type == "overkill":
                color = (0, 255, 255)
            elif draw_type == "miss":
                color = (255, 0, 255)
            else:
                color = (255, 255, 0)

            cv2.rectangle(img_data, (col_lines[col - 1], row_lines[row - 1]), (col_lines[col], row_lines[row]),
                          color, 5)

        cv2.imwrite(os.path.join(save_path, draw_type + "_" + defect_type + "_" + name + ".jpg"), img_data)


# 将组件级别的缺陷，转换为电池片级别
gt_xuhan_dcp = loc_transfer_2_dcp(gt_xuhan_loc)
gt_yinlie_dcp = loc_transfer_2_dcp(gt_yinlie_loc)
detected_xuhan_dcp = loc_transfer_2_dcp(detected_xuhan_loc)
detected_yinlie_dcp = loc_transfer_2_dcp(detected_yinlie_loc)

right_xh = set(gt_xuhan_dcp) & set(detected_xuhan_dcp)
overkill_xh = set(detected_xuhan_dcp) - set(gt_xuhan_dcp)
miss_xh = set(gt_xuhan_dcp) - set(detected_xuhan_dcp)

right_yl = set(gt_yinlie_dcp) & set(detected_yinlie_dcp)
overkill_yl = set(detected_yinlie_dcp) - set(gt_yinlie_dcp)
miss_yl = set(gt_yinlie_dcp) - set(detected_yinlie_dcp)

print("答案中xuhan电池片个数：", len(gt_xuhan_dcp))
print("检测出的xuhan电池片个数：", len(detected_xuhan_dcp))
print("检测正确的xuhan电池片个数：", len(right_xh))
print("过检xuhan电池片个数：", len(overkill_xh))
print("漏检xuhan电池片个数：", len(miss_xh))
print("\n")
print("答案中yinlie电池片个数：", len(gt_yinlie_dcp))
print("检测出的yinlie电池片个数：", len(detected_yinlie_dcp))
print("检测正确的yinlie电池片个数：", len(right_yl))
print("过检yinlie电池片个数：", len(overkill_yl))
print("漏检yinlie电池片个数：", len(miss_yl))

draw_rectangle(jinneng_dir, save_dir, overkill_xh, "overkill", "xuhan")
draw_rectangle(jinneng_dir, save_dir, miss_xh, "miss", "xuhan")
draw_rectangle(jinneng_dir, save_dir, overkill_yl, "overkill", "yinlie")
draw_rectangle(jinneng_dir, save_dir, miss_yl, "miss", "yinlie")

# print("检测出xuhan数：", right_xh)
# print("过检xuhan数：", overkill_xh)
# print("漏检xuhan数：", miss_xh)
