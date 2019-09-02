# coding: utf8
import os
import time
import xlrd
import shutil
import datetime

# strs = ["漏检", "miss"]
strs = ["过检", "overkill"]
dates = ["20190530", "20190531", "20190601", "20190602"]

# ---------------------
# get name and time
excel_book = xlrd.open_workbook(r'D:\OneDrive\MyDrive\OneDrive\Projects\jinneng\statistics\对比.xlsx')
sheet = excel_book.sheet_by_name(strs[0])
img_names = sheet.col_values(1)
modify_time = sheet.col_values(3)
name_time = {}
n = 0
for i in zip(img_names, modify_time):
    if n == 0:
        pass
    else:
        name_time[str(i[0]).strip() + ".jpg"] = i[1]
    n += 1

print(name_time)

# ---------------------
# get specific image
finded_imgs = []
for d in dates:
    file_path = os.path.join(r"D:\OneDrive\MyDrive\OneDrive\Projects\jinneng\image_only", d)
    for root, dir, files in os.walk(file_path):
        for file in files:
            full_path = os.path.join(root, file)
            img_name = os.path.basename(full_path)
            if img_name in name_time.keys():
                mtime = os.stat(full_path).st_mtime
                file_modify_time = time.strftime('%Y/%m/%d %H:%M', time.localtime(mtime))

                date1 = datetime.datetime.strptime(file_modify_time, "%Y/%m/%d %H:%M")
                date2 = datetime.datetime.strptime(name_time[img_name], "%Y/%m/%d %H:%M")

                if img_name in ['3271339123289.jpg', '3271409123176.jpg', '3271339123532.jpg']:
                    print(img_name, date2, date1)

                if date1.year == date2.year and date1.month == date2.month and date1.day == date2.day \
                        and date1.hour == date2.hour and abs(date1.minute - date2.minute) <= 2:
                    finded_imgs.append(img_name)
                    # print(d, img_name, file_modify_time)
                    shutil.copyfile(
                        full_path,
                        os.path.join(r"D:\OneDrive\MyDrive\OneDrive\Projects\jinneng\statistics", strs[1], img_name)
                    )

# ---------------------
# 打印未找到的图片
print(set(finded_imgs) - set(name_time.keys()))
print(set(name_time.keys()) - set(finded_imgs))
