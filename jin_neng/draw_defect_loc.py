import os
import cv2
import pandas as pd
from jin_neng.cut_zujian_pic import grid_cut


def get_defect_loc(standard_df):
    defect_loc = standard_df[(standard_df["缺陷名"] == "yinlie")
                             | (standard_df["缺陷名"] == "shixiao")
                             | (standard_df["缺陷名"] == "xuhan")]

    defect_loc = defect_loc.copy()
    # 将row和col组合成loc
    defect_loc["loc"] = defect_loc["缺陷名"] + "_" \
                        + (defect_loc["行"]).astype(str) + "_" \
                        + (defect_loc["列"]).astype(str)
    # 对loc按serial_number进行聚合
    defect_grouped = defect_loc['loc'].groupby(defect_loc['序列号']).apply(lambda x: set(x))

    res = {}
    for s_n in defect_grouped.index:
        res[s_n] = defect_grouped[s_n]
    return res


def draw_rectangle(img_path, save_path, defect_loc):
    """
    可视化
    :param defect_loc:
    :return:
    """

    for name, type_loc in defect_loc.items():
        # print(name, loc)
        pic_file = os.path.join(img_path, name + ".jpg")
        if os.path.exists(pic_file):
            img_data = cv2.imread(pic_file)
            _, (row_lines, col_lines) = grid_cut(img_data, 6, 12, edge_removed=False)

            for i in type_loc:
                defect_type, row, col = i.split("_")
                row, col = int(row), int(col)

                if defect_type == "yinlie":
                    color = (0, 255, 255)
                elif defect_type == "shixiao":
                    color = (255, 0, 255)
                else:  # xuhan
                    color = (255, 255, 0)

                cv2.rectangle(img_data, (col_lines[col - 1], row_lines[row - 1]), (col_lines[col], row_lines[row]),
                              color, 5)

            cv2.imwrite(os.path.join(save_path, name + ".jpg"), img_data)


# csv路径
test_csv = r"D:\Desktop\jn\flaw_statistic.csv"
# 图片路径
img_dir = r"D:\Desktop\jn\overkill"
# 保存路径
save_dir = r"D:\Desktop\jn"

# 可视化
test_df = pd.read_csv(test_csv)
detected_loc = get_defect_loc(test_df)
draw_rectangle(img_dir, save_dir, detected_loc)

