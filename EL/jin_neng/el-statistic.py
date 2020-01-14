import os
import pandas as pd
import shutil

df_data = pd.read_csv(r"./flaw_statistic_0508.csv")
data = df_data[['过检时间', '序列号', '缺陷名']]

# ----------------------------------------------------
# ai检测结果list
ai_list = []
for index, row in data.iterrows():
    if row['过检时间'].split(" ")[0] == "2019/5/8":
        ai_list.append(str(int(row['序列号'])))
ai_list = list(set(ai_list))

# ----------------------------------------------------
# 工人检测 list
df_data_fixed = pd.read_csv("./0508.csv")
data_fixed = df_data_fixed[(df_data_fixed['defect_names'] == "xuhan") | (df_data_fixed['defect_names'] == "yinlie")]

fixed_data = []
miss = 0
overkill = 0
right = 0
for data in data_fixed:
    fixed_data.append(str(int(data)))
fixed_data = list(set(fixed_data))

print("fixed_data:", fixed_data)
print("ai_list:", ai_list)

for data in fixed_data:
    if data in ai_list:
        right += 1
    else:
        miss += 1
        pic_name = data + ".jpg"
        print(pic_name)
        # try:
        #     shutil.copy(r"F:\20190417\all-pic/"+pic_name, r"F:\20190417\miss/" + pic_name)
        # except Exception as e:
        #     continue

print("算法检出缺陷总数:", len(ai_list))
# all_pic_list = os.listdir(r"F:\20190314\NightFlight")
for data2 in ai_list:
    if data2 not in fixed_data:
        overkill += 1
        # pic_name = data2 + ".jpg"
        # if pic_name not in all_pic_list:
        #     print(pic_name)
        #     continue
        # shutil.copy(r"F:\20190417\all-pic/" + pic_name, r"F:\20190417\overkill/" + pic_name)

print("漏检图片数量:", miss, "过检图片数量:", overkill, "正确图片数量:", right, "返修图片总量：", len(fixed_data))
