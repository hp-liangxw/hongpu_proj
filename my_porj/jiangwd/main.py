csv_file = "ZP2018-08-02_00-01-48LV2(1).csv"
his_file1 = "PRE_VIS_01.his"
his_file2 = "PRE_VIS_02.his"

# read his file
with open(his_file1, "r") as f_his1:
    his1_lines = f_his1.readlines()

with open(his_file2, "r") as f_his2:
    his2_lines = f_his2.readlines()

his = []
his.extend(his1_lines[2:])
his.extend(his2_lines[2:])

his_dict = {}
for i in his:
    strs = i.strip().split("\t")
    his_dict[strs[1][:-3]] = [strs[2], strs[3]]
# print(his_dict)

# read csv file
with open(csv_file, "r") as f_csv:
    csv_lines = f_csv.readlines()

res_11 = []
res_12 = []
res_13 = []
res_14 = []

for i in csv_lines[2:]:
    strs = i.strip().split(",")
    t = i.strip() + "," + ",".join(his_dict[strs[1][:-3]]) + "," + str(float(strs[40]) - float(strs[36])) + "\n"
    if strs[2] == "11":
        res_11.append(t)
    elif strs[2] == "12":
        res_12.append(t)
    elif strs[2] == "13":
        res_13.append(t)
    elif strs[2] == "14":
        res_14.append(t)
    else:
        pass

res_11.insert(0, csv_lines[0].strip() + ",VIS,MIN," + "温度差\n")
res_12.insert(0, csv_lines[0].strip() + ",VIS,MIN," + "水汽差\n")
res_13.insert(0, csv_lines[0].strip() + ",VIS,MIN," + "湿度差\n")
res_14.insert(0, csv_lines[0].strip() + ",VIS,MIN," + "液态水含量差\n")

# write file
with open("11.csv", "w") as f_11:
    f_11.writelines(res_11)
with open("12.csv", "w") as f_12:
    f_12.writelines(res_12)
with open("13.csv", "w") as f_13:
    f_13.writelines(res_13)
with open("14.csv", "w") as f_14:
    f_14.writelines(res_14)