# coding:utf-8

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os, sys
import cv2
# import utils.shanxian_utils as su
import json
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
import time
import logging
import scipy.signal as signal

import src.yinlie_utils.extract_yinlie_feature as extract

def find_shanxian_row(img, shanxian_num=5, order_percentage=0.113):
    # 1.get the sharpened pic and count the column avg

    rows = np.mean(img, axis=1)
    h ,w = img.shape
    # 2.fill the start and end
    l = int(0.04 * h)
    pads = np.ones(l) * np.mean(rows)
    rows = np.hstack((pads, rows[l:-l], pads))
    #rows[:l] = np.mean(rows)
    #rows[-l:] = np.mean(rows)

    # 3.find shanxian row
    order = int(img.shape[0] * order_percentage)
    shanxian_row = signal.argrelmin(rows, order=order)[0]
    num_row = len(shanxian_row)

    return num_row, shanxian_row, None

def _model_path():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),'model_joblib_yinlie.pkl')
def _scalar_path():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),'scalar_joblib.pkl')

def yinlie_post_process(img, row, col, bbox, prob):
    """
    隐裂缺陷的后处理

    :param img: 初次预判为隐裂的电池片图像
    :param row: 电池片在大图中所在的行，从1开始
    :param col: 电池片在大图中所在的列，从1开始
    :param bbox: 隐裂的bounding box, tuple格式为 (x1, y1, x2, y2)
    :param prob: 初次预判为隐裂的置信度
    :returns: 后处理完成后判断为隐裂的置信度
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if not hasattr(yinlie_post_process, 'model'):
        yinlie_post_process.model = joblib.load(_model_path())

    if not hasattr(yinlie_post_process, 'scalar'):
        yinlie_post_process.scalar = joblib.load(_scalar_path())

    near_thre = 0.1
    xmin, ymin, xmax, ymax = bbox
    h, w = img.shape
    x_max_pos = xmax / w
    x_min_pos = xmin / w
    height = ymax - ymin
    width = xmax - xmin

    # pass if the size is big enough
    if height*1.0/h > 0.3 or width*1.0/w > 0.3:
        return 1.0

    max_size_recip = 1 / np.max([height, width])
    near_vertical_side = bool((x_min_pos < near_thre) + (x_max_pos > (1 - near_thre)))

    dict_all = {
        'prob': prob,
        'max_size_recip': max_size_recip,
        'near_vertical_side': near_vertical_side,
    }
    cols = ['prob', 'near_vertical_side', 'max_size_recip']
    df = pd.DataFrame(dict_all, index=[0])
    df = df[cols]
    df = yinlie_post_process.scalar.transform(df)

    pred = yinlie_post_process.model.predict_proba(df)[0, 1]
    #     pred = model.predict(df)[0]
    logging.info('POST PROCESS: a yinlie with prob={} at ({},{})'.format(pred, row, col))
    return pred

def yinlie_xizhi(img, row, col, bbox, yinlie_xizhi_thresh):
    """
    隐裂缺陷的后处理

    :param img: 初次预判为隐裂的电池片图像
    :param row: 电池片在大图中所在的行，从1开始
    :param col: 电池片在大图中所在的列，从1开始
    :param bbox: 隐裂的bounding box, tuple格式为 (x1, y1, x2, y2)
    :param yinlie_xizhi_thresh: 反应隐裂粗细的阈值，认为隐裂的粗细小于阈值的就是细直的假隐裂
    :returns: 输出一个数值，是“细直程度”的度量
    """
 
    # 获取局部的隐裂形态信息 labels3
    yinlieMask, shanxian, labels3 = extract.getYinlieInfo(img, bbox, verbose=False, padding=10)
    # status 直接表示 是否判断为一个假隐裂， False 表示 假， True 表示认为是真的
    status, solidity, angle, ratio_y_x = extract.getThicknessInfo(img, bbox, shanxian, labels3, verbose=False, thick=yinlie_xizhi_thresh, thres=10)
    
    return status
