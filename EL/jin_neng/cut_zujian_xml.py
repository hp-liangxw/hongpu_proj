# coding:utf-8
# 组件切图（不含层后正畸、串检）后修改xml为小图，并保存图片与xml（可选保存OK小图）
# import
import os
import glob
import cv2
import numpy as np
import xml.dom.minidom as minidom
from EL.jin_neng.cut_zujian_pic import grid_cut

# path
pic_path = r'D:\JingNeng_zj_0509\img'
xml_path = r'D:\JingNeng_zj_0509\xml'
save_path = r'D:\JingNeng_dcp_0509'
os.mkdir(os.path.join(save_path, 'img'))
os.mkdir(os.path.join(save_path, 'xml'))
save_ok_pic = False
rows = 6
cols = 12


def getdefect(xml_file, xml_name, cut_images, row_lines, col_lines):
    parsexml = minidom.parse(xml_file)
    root = parsexml.documentElement
    nodes_length = len(root.getElementsByTagName('object'))
    defect = {'pic_name': xml_name + '.jpg'}
    for i in range(nodes_length):
        xmin = eval(root.getElementsByTagName('xmin')[i].firstChild.data)
        ymin = eval(root.getElementsByTagName('ymin')[i].firstChild.data)
        xmax = eval(root.getElementsByTagName('xmax')[i].firstChild.data)
        ymax = eval(root.getElementsByTagName('ymax')[i].firstChild.data)
        name = root.getElementsByTagName('name')[i].firstChild.data
        centorid_x = int((xmax + xmin) / 2)
        centorid_y = int((ymax + ymin) / 2)
        loc_x = np.where(np.array(col_lines) <= centorid_x)[0][-1]
        loc_y = np.where(np.array(row_lines) <= centorid_y)[0][-1]
        h_stride, w_stride = cut_images[row_lines[loc_y], col_lines[loc_x]].shape[:2]
        xmin = max(0, xmin - col_lines[loc_x])
        ymin = max(0, ymin - row_lines[loc_y])
        xmax = min(w_stride, xmax - col_lines[loc_x])
        ymax = min(h_stride, ymax - row_lines[loc_y])
        if (loc_y, loc_x) in defect:
            defect[(loc_y, loc_x)].append({'class': name, 'coord': (xmin, ymin, xmax, ymax)})
        else:
            defect[(loc_y, loc_x)] = [
                {'size': (w_stride, h_stride, 1), 'class': name, 'coord': (xmin, ymin, xmax, ymax)}]

    return defect


def xml2xml(defect, save_path):
    for item in defect:
        if type(item) == tuple:
            doc = minidom.Document()
            annotations = doc.createElement('annotations')
            doc.appendChild(annotations)
            folder = doc.createElement('folder')
            annotations.appendChild(folder)
            folder.appendChild(doc.createTextNode(os.path.basename(save_path)))
            filename = doc.createElement('filename')
            annotations.appendChild(filename)
            savename = defect['pic_name'] + '-' + str(item[0]) + '-' + str(item[1])
            filename.appendChild(doc.createTextNode(savename + '.jpg'))
            path = doc.createElement('path')
            annotations.appendChild(path)
            path.appendChild(doc.createTextNode(save_path + r'\\' + savename + '.jpg'))
            source = doc.createElement('source')
            annotations.appendChild(source)
            database = doc.createElement('database')
            source.appendChild(database)
            database.appendChild(doc.createTextNode('Unknown'))

            size = doc.createElement('size')
            annotations.appendChild(size)
            width = doc.createElement('width')
            size.appendChild(width)
            width.appendChild(doc.createTextNode(str(defect[item][0]['size'][0])))

            height = doc.createElement('height')
            size.appendChild(height)
            height.appendChild(doc.createTextNode(str(defect[item][0]['size'][1])))

            depth = doc.createElement('depth')
            size.appendChild(depth)
            depth.appendChild(doc.createTextNode(str(defect[item][0]['size'][2])))

            segmented = doc.createElement('segmented')
            annotations.appendChild(segmented)
            segmented.appendChild(doc.createTextNode('0'))

            for defects in defect[item]:
                # if defects['class'] not in ('zhengxuhan','fuxuhan'):
                object_ = doc.createElement('object')
                annotations.appendChild(object_)
                name = doc.createElement('name')
                object_.appendChild(name)
                name.appendChild(doc.createTextNode(defects['class']))
                pose = doc.createElement('pose')
                object_.appendChild(pose)
                pose.appendChild(doc.createTextNode('Unspecified'))
                truncated = doc.createElement('truncated')
                object_.appendChild(truncated)
                truncated.appendChild(doc.createTextNode('0'))
                difficult = doc.createElement('difficult')
                object_.appendChild(difficult)
                difficult.appendChild(doc.createTextNode('0'))

                bndbox = doc.createElement('bndbox')
                object_.appendChild(bndbox)
                xmin = doc.createElement('xmin')
                bndbox.appendChild(xmin)
                xmin.appendChild(doc.createTextNode(str(defects['coord'][0])))
                ymin = doc.createElement('ymin')
                bndbox.appendChild(ymin)
                ymin.appendChild(doc.createTextNode(str(defects['coord'][1])))
                xmax = doc.createElement('xmax')
                bndbox.appendChild(xmax)
                xmax.appendChild(doc.createTextNode(str(defects['coord'][2])))
                ymax = doc.createElement('ymax')
                bndbox.appendChild(ymax)
                ymax.appendChild(doc.createTextNode(str(defects['coord'][3])))

            f = open(os.path.join(save_path, 'xml') + "/" + savename + '.xml', 'w')
            doc.writexml(f)
            f.close()


for xml_file in glob.glob(os.path.join(xml_path, '*.xml')):
    xml_name = xml_file.split('\\')[-1].split('.')[0]
    print(xml_name)
    # 根据xml搜索对应pic，并进行切图
    pic_file = glob.glob(os.path.join(pic_path, '%s.jpg' % xml_name))[0]
    img_data = cv2.imread(pic_file)
    img_data_cut = img_data[100:-135, 100:-100, :].copy()
    cut_images, (row_lines, col_lines) = grid_cut(img_data, rows, cols, edge_removed=False)
    try:
        defect = getdefect(xml_file, xml_name, cut_images, row_lines, col_lines)
        xml2xml(defect, save_path)
        for item in defect:
            if type(item) == tuple:
                save_name = defect['pic_name'] + '-' + str(item[0]) + '-' + str(item[1])
                cv2.imwrite(os.path.join(save_path, 'img') + '\\' + save_name + '.jpg',
                            cut_images[row_lines[item[0]], col_lines[item[1]]])
    except:
        print('error', xml_name)
