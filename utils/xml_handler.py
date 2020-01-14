"""
parse_xml: 解析单个xml文件
get_xml_info: 得到一个文件夹下，所有xml文件的信息
print_all_labels: 输出所有标签名称及对应的标签个数
del_specific_label: 删除特定的标签(bbox)
del_specific_xml: 删除特定的xml文件
del_empty_xml: 删除没有缺陷标签的xml文件
change_label_name: 批量修改标签名称
del_space: 删除xml文件名中的空格，并用下划线代替。
merge_xml: 将两处的xml文件进行合并
get_image_in_bbox: 保存标签图片
change_ext_in_xml: 修改xml中文件名的后缀
"""

# -*- coding: utf-8 -*-

import glob
import os
import shutil
from xml.dom.minidom import parse as xmlparser
from PIL import Image


class LabelHandler:
    def __init__(self):
        pass

    @classmethod
    def parse_xml(cls, xml_file):
        """
        解析单个xml文件
        :param xml_file:
        :return: 返回标签名和对应的bbox位置。
            [
                {"name": "aodian", "bbox": [10, 20, 30, 30]},
                {"name": "aodian", "bbox": [10, 20, 30, 30]},
                ...
            ]
        """

        # 得到文档元素对象
        root = xmlparser(xml_file).documentElement
        object_list = root.getElementsByTagName('object')

        info = []
        for ob in object_list:
            # 标签名称
            class_name = ob.getElementsByTagName('name')[0].childNodes[0].data
            # 标签位置
            box = ob.getElementsByTagName('bndbox')[0]
            xmin = int(box.getElementsByTagName('xmin')[0].childNodes[0].data)
            ymin = int(box.getElementsByTagName('ymin')[0].childNodes[0].data)
            xmax = int(box.getElementsByTagName('xmax')[0].childNodes[0].data)
            ymax = int(box.getElementsByTagName('ymax')[0].childNodes[0].data)

            # 返回信息
            info += [{"name": class_name, "bbox": [xmin, ymin, xmax, ymax]}]

        return info

    @classmethod
    def get_xml_info(cls, xml_dir_path):
        """
        得到一个文件夹下，所有xml文件的信息
        :param xml_dir_path:
        :return:
            {
                "img_name1": info,  (info见parse_xml函数)
                "img_name2": info,
                ...
            ]
        """
        xml_list = glob.glob(os.path.join(xml_dir_path, "*.xml"))

        xml_info = {}
        for n in xml_list:
            pic_name = os.path.basename(n).split('_')[1]
            info = cls.parse_xml(n)
            xml_info[pic_name] = info

        return xml_info

    @classmethod
    def print_all_labels(cls, xml_dir_path):
        """
        输出所有标签名称及对应的标签个数
        :param xml_dir_path:
        :return:
        """
        # 存储标签列表
        label_num = {}

        # 所有的xml
        xml_list = glob.glob(os.path.sep.join([xml_dir_path, '*.xml']))

        for xml_file in xml_list:
            print(xml_file)
            # xml_name = os.path.basename(xml_file)
            xml_parser = xmlparser(xml_file)
            annotation = xml_parser.documentElement

            # 获取标签列表
            object_list = annotation.getElementsByTagName('object')
            # flag = False
            for ob in object_list:
                # 获取标签名
                object_name = ob.getElementsByTagName('name')[0].childNodes[0].data
                # if object_name == "huahen":
                #     print(xml_file)
                if object_name in label_num.keys():
                    label_num[object_name] += 1
                else:
                    label_num[object_name] = 1

        for k, v in label_num.items():
            print(k, ':', v)

        return list(label_num.keys())

    @classmethod
    def del_specific_label(cls, xml_path, save_path, label_list):
        """
        删除特定的标签，比如dahuahen
        :param xml_path:
        :param save_path:
        :param label_list: list, 需要删除的标签
        :return:
        """

        xml_list = glob.glob(os.path.sep.join([xml_path, '*.xml']))

        for xml_file in xml_list:

            xml_name = os.path.basename(xml_file)

            xml_parser = xmlparser(xml_file)
            annotation = xml_parser.documentElement
            # 获取标签列表
            object_list = annotation.getElementsByTagName('object')

            for ob in object_list:
                # 获取标签名
                if ob.getElementsByTagName('name')[0].childNodes[0].data in label_list:
                    annotation.removeChild(ob)

            with open(os.path.sep.join([save_path, xml_name]), 'w', encoding='utf8') as f_w:
                annotation.writexml(f_w)

    @classmethod
    def del_specific_xml(cls, xml_path):
        """
        删除特定的xml文件
        :param xml_path:
        :return:
        """
        # 所有的xml
        xml_list = glob.glob(os.path.sep.join([xml_path, '*.xml']))

        n = 0
        for xml_file in xml_list:
            # xml_name = os.path.basename(xml_file)
            xml_parser = xmlparser(xml_file)
            annotation = xml_parser.documentElement

            # 获取标签列表
            object_list = annotation.getElementsByTagName('object')
            flag = False
            for ob in object_list:
                # 获取标签名
                object_name = ob.getElementsByTagName('name')[0].childNodes[0].data
                if object_name == "chengshu":
                    n += 1
                    flag = True
                    continue

            if flag:
                print("{}, {}".format(n, xml_file))
                os.remove(os.path.join(xml_path, xml_file))
                continue

    @classmethod
    def del_empty_xml(cls, xml_dir):
        """
        删除文件夹中没有缺陷标签的xml文件
        :param xml_dir:
        :return:
        """
        xml_list = glob.glob(os.path.sep.join([xml_dir, '*.xml']))

        for xml_file in xml_list:
            print(xml_file)
            # 获取标签列表
            xml_parser = xmlparser(xml_file)
            annotation = xml_parser.documentElement
            # 获取标签列表
            object_list = annotation.getElementsByTagName('object')

            # 如果object_list为空
            if not object_list:
                print("=====================WARNING! EMPTY XML!=====================")
                print("del empty label:{}".format(xml_file))
                # 如果没有标签，则删除该xml文件
                # 出现这种情况的原因是，第一次对该图片打了标签，第二次复检时，将该标签删除了
                os.remove(xml_file)

        print("=====================NICE! NO EMPTY XML!=====================")

    @classmethod
    def change_label_name(cls, in_path, out_path, change_dict):
        """
        批量修改标签名称
        :param in_path:
        :param change_dict: {(old_name): new_name}
        :return:
        """
        xml_list = glob.glob(os.path.sep.join([in_path, '*.xml']))

        for xml in xml_list:
            xml_name = os.path.basename(xml)

            with open(xml, 'r', encoding="utf8") as f_r:  # 读文件
                lines = f_r.readlines()

            for old_set, new_name in change_dict.items():
                for old_name in old_set:
                    for i in range(len(lines)):
                        if '<name>' + old_name + '</name>' in lines[i]:
                            # print(lines[i])
                            lines[i] = lines[i].replace(old_name, new_name)  # 替换

            # 替换后重新写入文件
            # print(xml)
            with open(os.path.sep.join([out_path, xml_name]), 'w', encoding="utf8") as f_w:
                f_w.writelines(lines)

    @classmethod
    def remove_space(cls, xml_path, save_path, char_before, char_after):
        """
        删除xml文件名中的空格，并用下划线代替。
        文件名修改后，xml文件中的filename和path标签也要修改。
        :param xml_path:
        :param save_path:
        :return:
        """

        xml_list = glob.glob(os.path.sep.join([xml_path, '*.xml']))

        for xml in xml_list:
            xml_name = os.path.basename(xml)

            # 开始解析xml
            annotation = xmlparser(xml).documentElement

            # 如果文件名中有空格
            if char_before in xml_name:
                # 修改文件名，用_代替空格
                xml_name = xml_name.replace(char_before, char_after)

                # 修改xml中的文件名
                ob_filename = annotation.getElementsByTagName('filename')
                ob_path = annotation.getElementsByTagName('path')
                # 获取filename和path对象的值
                # filename_value = ob_filename[0].childNodes[0].data
                path_value = ob_path[0].childNodes[0].data

                # 赋予filename和path对象新值
                abs_path = '_'.join(path_value.split(' '))
                ob_path[0].childNodes[0].data = abs_path
                ob_filename[0].childNodes[0].data = os.path.basename(abs_path)
                # = '_'.join(filename_value.split(' '))

            with open(os.path.sep.join([save_path, xml_name]), 'w', encoding='utf8') as f_w:
                annotation.writexml(f_w)

    @classmethod
    def get_objects(cls, xml_file):
        with open(xml_file, 'r', encoding='utf8') as f_xml:
            lines = f_xml.readlines()

        start = 0
        for i in range(len(lines)):
            if lines[i].strip() == '<segmented>0</segmented>':
                start = i

        return lines[start + 1: -1]

    @classmethod
    def merge_xml(cls, src_path, dst_path):
        """
        将两处的标签进行合并
        :param src_path:
        :param dst_path:
        :return:
        """

        # 原文件夹文件列表
        src_list = os.listdir(src_path)

        for src_name in src_list:

            src_file = os.path.sep.join([src_path, src_name])

            dst_file = os.path.sep.join([dst_path, src_name])
            # 如果目标文件夹存在该文件
            if os.path.exists(dst_file):

                src_info = cls.get_objects(src_file)

                with open(dst_file, 'r', encoding='utf8') as f_xml:
                    # print(dst_file)
                    dst_info = f_xml.readlines()

                new_info = dst_info[:-1] + src_info + [dst_info[-1]]

                # 写入新文件
                with open(dst_file, 'w', encoding='utf8') as f_xml:
                    f_xml.writelines(new_info)

            else:
                # 复制
                shutil.copyfile(os.path.sep.join([src_path, src_name]), os.path.sep.join([dst_path, src_name]))

    @classmethod
    def get_image_in_bbox(cls, xml_path, img_path, save_path, defect_names):
        """
        保存所有的缺陷小图
        :param xml_path:
        :param img_path:
        :param save_path:
        :param defect_names: 缺陷名称，列表类型
        :return:
        """
        # 按标签名创建文件夹
        for d_n in defect_names:
            # 分标签保存
            bbox_save_dir = os.path.join(save_path, d_n)
            if not os.path.exists(bbox_save_dir):
                os.mkdir(bbox_save_dir)

        xml_list = glob.glob(os.path.sep.join([xml_path, '*.xml']))

        num = 1
        for xml in xml_list:
            # 文件名
            name = os.path.basename(xml).split('.')[0]
            print("\t{}\t:\t{}".format(num, name))
            img_file = os.path.sep.join([img_path, '%s.jpg' % name])

            info = cls.parse_xml(xml)

            i = 0
            for f in info:
                object_name = f["name"]
                x1, y1, x2, y2 = f["bbox"]
                if x2 > x1 and y2 > y1:
                    if object_name in defect_names:
                        img_data = Image.open(img_file)
                        img_roi = img_data.crop((x1, y1, x2, y2))
                        out_name = '-'.join([name, object_name, str(i) + '.jpg'])
                        img_roi.save(os.path.sep.join([save_path, object_name, out_name]))
                        i += 1
                else:
                    print("wrong xml file: %s" % name)

            num += 1

    @classmethod
    def change_ext_in_xml(cls, in_path, out_path):
        """
        修改xml中文件名的后缀
        :param in_path:
        :param out_path:
        :return:
        """
        xml_list = glob.glob(os.path.sep.join([in_path, '*.xml']))

        for i in xml_list:
            xml_name = os.path.basename(i)

            with open(i, 'r', encoding='utf8') as fr_xml:
                value = fr_xml.readlines()

            temp = value[2].replace('.bmp', '.jpg')
            value[2] = temp

            temp = value[3].replace('.bmp', '.jpg')
            value[3] = temp

            with open(os.path.sep.join([out_path, xml_name]), 'w', encoding='utf8') as fw_xml:
                fw_xml.writelines(value)

    @classmethod
    def copy_images_according_to_xml(cls, xml_path, img_path, dst_path):
        """
        根据xml文件名，复制出对应的图片。
        :param img_path: xml文件路径
        :param xml_path: 图片路径
        :param dst_path: 拷贝图片的目的路径
        :return:
        """
        # 获取所有图片的列表
        img_list = []
        for rt, dirs, files in os.walk(img_path + os.sep):
            if files:
                for file in files:
                    if file.split(".")[-1] == "jpg":
                        img_list.append(os.path.join(rt, file))

        # 根据xml的名称拷贝图片
        for rt, dirs, files in os.walk(xml_path + os.sep):
            if files:
                for file in files:
                    xml_name = os.path.splitext(file)[0]
                    for img in img_list:
                        if xml_name in img:
                            dst_img = os.path.sep.join([dst_path, xml_name + '.jpg'])
                            if not os.path.exists(dst_img):
                                shutil.copyfile(img, dst_img)

    @classmethod
    def get_img_xml_pair(cls, in_dir, out_dir):
        
        # 生成保存文件夹
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        os.makedirs(os.path.join(out_dir, 'img'))
        os.makedirs(os.path.join(out_dir, 'xml'))

        # 检验是否是xml文件
        xml_list = os.listdir(os.path.join(in_dir, "xml"))
        for i in xml_list:
            if not i.endswith(".xml"):
                print("{} is not xml file".format(i))
                xml_list.remove(i)

        # 
        for i in xml_list:
            name = i[:-4]
            img_path = os.path.join(in_dir, "img", name + ".jpg")
            xml_path = os.path.join(in_dir, "xml", name + ".xml")
            if os.path.exists(img_path):
                shutil.copyfile(img_path, os.path.join(out_dir, "img", name + ".jpg"))
                shutil.copyfile(xml_path, os.path.join(out_dir, "xml", name + ".xml"))
        
        



# LabelHandler.del_empty_xml(r"D:\Desktop\501_v3_class_xuhan_2\xml")
# LabelHandler.change_label_name(
#     r"D:\Desktop\dongfang_4+5\3_xuhan\xuhan\xml", 
#     r"D:\Desktop\dongfang_4+5\3_xuhan\xuhan\xml2", 
#     {("negative", ): "neg"}
# )

LabelHandler.get_img_xml_pair(
    r"D:\Desktop\501_v3_class_xuhan_2\all",
    r"D:\Desktop\501_v3_class_xuhan_2\pair"
)
# LabelHandler.print_all_labels(r"D:\Desktop\501_v3_class_xuhan_2\xml")
