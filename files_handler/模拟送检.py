import re
import os
import glob
import shutil
import time


def slice_inspection():
    slices_img_path = r'.\\origin'  # 分段图放在这里

    pic_dict = {re.split('_', os.path.basename(k))[0]: {} for k in glob.glob(os.path.join(slices_img_path, r'*H.jpg'))}
    for p in glob.glob(os.path.join(slices_img_path, r'*H.jpg')):
        pic_dict[re.split('_', os.path.basename(p))[0]][re.findall('([A-C]\d)H', p)[0]] = p
    print(len(pic_dict))

    for k, v in pic_dict.items():
        if len(v) == 12:
            for pic in v.values():
                name = os.path.basename(pic)
                shutil.move(pic, '.\\test\\' + name)
                time.sleep(10)


def component_inspection():
    # root_path = r"D:\OneDrive\MyDrive\OneDrive\Projects\longi\AI_results\2019.08.03_day"
    root_path = r"D:\OneDrive\MyDrive\OneDrive\Projects\longi\production_line_images\repair_lost_images"
    shifts = [
        "20190804", "20190805"
        #     "DayFlight",
        #     "NightFlight",
        #     "CFlight"
    ]
    ng_ok = [
        #     "NG",
        #     "OK"
        "DayFlight",
        "NightFlight",
    ]

    for s in shifts:
        for x in ng_ok:
            full_path = os.path.join(root_path, s, x)
            img_list = os.listdir(full_path)

            for img in img_list:
                print(img)
                shutil.copyfile(os.path.join(full_path, img),
                                os.path.join(r'D:\Desktop\test', img))
                time.sleep(15)

            # 重命名
            os.rename(r"D:\Desktop\result\2019.08.07\ng",
                      os.path.join(r"D:\Desktop\result\2019.08.07", "ng_" + s + "_" + x))
            os.rename(r"D:\Desktop\result\2019.08.07\ng_with_box",
                      os.path.join(r"D:\Desktop\result\2019.08.07", "ng_with_box_" + s + "_" + x))
            # 删除test中的文件
            test_path = r"D:\Desktop\test"
            test_imgs = os.listdir(test_path)
            for t in test_imgs:
                os.remove(os.path.join(test_path, t))
