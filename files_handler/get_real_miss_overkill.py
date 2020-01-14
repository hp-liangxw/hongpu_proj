# 文件拷贝
import os
import time
import shutil

# 沛德机台图片路径
all_dir = r"D:\OneDrive\MyDrive\OneDrive\Projects\longi\production_line_images\20190807"
# 软件检测结果
ai_result_dir = r"D:\OneDrive\MyDrive\OneDrive\Projects\longi\AI_results\2019.08.07_day"


def get_real_miss(all_dir, ai_result_dir):
    """
    查找真实miss的文件
    :param all_dir:
    :param ai_result_dir:
    :return:
    """
    miss_dir = os.path.join(ai_result_dir, r"missing")
    actually_dir = os.path.join(ai_result_dir, r"actually_miss")
    if not os.path.exists(actually_dir):
        os.makedirs(actually_dir)

    miss_images = os.listdir(miss_dir)

    for img_name in miss_images:
        miss_path = os.path.join(miss_dir, img_name)
        t1 = os.path.getmtime(miss_path)

        for root, dirs, files in os.walk(all_dir):  # 遍历所有目录，包括自身
            for name in files:  # 遍历文件，抓取指定文件
                if "_" in img_name:
                    name_1 = img_name.split("_")[0]
                else:
                    name_1 = img_name
                if "_" in name:
                    name_2 = name.split("_")[0]
                else:
                    name_2 = name

                if name_1 == name_2:
                    original_path = os.path.join(root, name)
                    t2 = os.path.getmtime(original_path)
                    #                     print(original_path)
                    #                     print(miss_path)
                    #                     print(root)
                    #                     print(t2 - t1)
                    if 0 < t2 - t1 <= 2 * 60 and "NG" in root:
                        #                     print(root)
                        print(img_name)
                        shutil.copyfile(
                            miss_path,
                            os.path.join(actually_dir, img_name)
                        )


def get_real_overkill(all_dir, ai_result_dir):
    """
    查找真实的overkill
    :param all_dir:
    :param ai_result_dir:
    :return:
    """
    over_dir = os.path.join(ai_result_dir, r"ng_with_box")
    actually_dir = os.path.join(ai_result_dir, r"actually_overkill")
    if not os.path.exists(actually_dir):
        os.makedirs(actually_dir)

    overkill_images = os.listdir(over_dir)

    for img_name in overkill_images:
        miss_path = os.path.join(over_dir, img_name)
        t1 = os.path.getmtime(miss_path)

        for root, dirs, files in os.walk(all_dir):  # 遍历所有目录，包括自身
            for name in files:  # 遍历文件，抓取指定文件
                if "_" in img_name:
                    name_1 = img_name.split("_")[0]
                else:
                    name_1 = img_name
                if "_" in name:
                    name_2 = name.split("_")[0]
                else:
                    name_2 = name
                if img_name in name:
                    original_path = os.path.join(root, name)
                    t2 = os.path.getmtime(original_path)
                    if 0 < t2 - t1 <= 2 * 60 and "OK" in root:
                        #                     print(root)
                        print(img_name)
                        shutil.copyfile(
                            miss_path,
                            os.path.join(actually_dir, img_name)
                        )


def rework_component():
    """
    查找返修组件
    :return:
    """
    root_path = r"D:\OneDrive\MyDrive\OneDrive\Projects\longi\production_line_images\20190805"
    shifts = [
        "DayFlight",
        #     "NightFlight",
        #     "CFlight"
    ]
    ng_ok = [
        "NG",
        "OK"
    ]

    img_dict = {}
    lost_path = r"D:\OneDrive\MyDrive\OneDrive\Projects\longi\production_line_images\repair_lost_images\20190805\DayFlight"
    for s in shifts:
        for x in ng_ok:
            full_path = os.path.join(root_path, s, x)
            img_list = os.listdir(full_path)

            for img in img_list:
                img_path = os.path.join(full_path, img)

                if "_" in img:
                    name = img.split("_")[0]
                else:
                    name = img.split(".")[0]

                modify_t = os.path.getmtime(img_path)
                if name not in img_dict:
                    img_dict[name] = [modify_t, img_path, img]
                else:
                    if img_dict[name][0] > modify_t:
                        shutil.copyfile(img_dict[name][1], os.path.join(lost_path, img_dict[name][2]))
                        img_dict[name] = [modify_t, img_path, img]
                    else:
                        shutil.copyfile(img_path, os.path.join(lost_path, img))


def merge_ng_ok():
    folder = r"D:\longi_production_line_images"

    dates = ["20190731", "20190801", "20190802", "20190803",
             "20190804", "20190805", "20190806", "20190807"]
    # dates = ["20190730"]
    for d in dates:
        if d == "20190731":
            shifts = ["NightFlight", "CFlight"]
        elif d == "20190807":
            shifts = ["DayFlight"]
        else:
            shifts = ["DayFlight", "NightFlight", "CFlight"]

        for s in shifts:
            # 创建文件夹
            day_dir = os.path.join(folder, d, "Day")
            night_dir = os.path.join(folder, d, "Night")
            if not os.path.exists(day_dir):
                os.makedirs(day_dir)
            if not os.path.exists(night_dir):
                os.makedirs(night_dir)

            root_dir = os.path.join(folder, d, s)
            #
            for root, dirs, files in os.walk(root_dir):  # 遍历所有目录，包括自身
                for name in files:  # 遍历文件，抓取指定文件
                    src_path = os.path.join(root, name)
                    if s == "DayFlight":
                        to_dir = day_dir
                    else:
                        to_dir = night_dir

                    if not os.path.exists(os.path.join(to_dir, name)):
                        dst_path = os.path.join(to_dir, name)
                        shutil.move(src_path, dst_path)
                    else:
                        if "OK" in src_path:
                            dst_path = os.path.join(to_dir, name.split(".")[0] + "_" + \
                                                    str(time.time())[6:15].replace(".", "11") + ".jpg")
                        else:
                            dst_path = os.path.join(to_dir, name.split(".")[0] + "_" + \
                                                    str(time.time())[6:15].replace(".", "77") + ".jpg")
                        shutil.move(src_path, dst_path)
