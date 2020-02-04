import cv2
import os
import glob
import re
import numpy as np
from scipy.signal import argrelmin, argrelmax
from scipy import optimize
import pickle
import logging


def joint_image(img_list, row, h=1000, pic_size=[3, 3, 3, 3]):
    """
    :img_list: 图片矩阵的列表
    :h: 待拼接高度
    :pic_size: 待拼接的图片片数
    returns: 去边并水平拼接好的图像
    """
    piece1, piece2, piece3, piece4 = pic_size
    concat_dict = {}
    for i in range(1, 5):
        img = img_list[i-1]
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = _undistort(img, str(i))
        img_down = _pyrDown(img, 3)
        dots = _get_grid_corners(img_down, row, col="%d" %
                                 i, pic_size=[2, 3, 3, 2], verbose=False)
        concated_piece = _cut_and_paste_one_piece(
            img_down, img, row, col="%d" % i, h=1000, pic_size=[2, 3, 3, 2])
        concat_dict['piece%i' % i] = concated_piece
    concat_4 = np.hstack([concat_dict['piece%i' % i] for i in range(1, 5)])
    return concat_4


def _get_grid_corners(img_down, row, col, pic_size, verbose=False):
    """
    获得ROI的四个角点
    :img_down: 输入图像的下采样图
    row：图片在大图中的行位置，从下至上ABC
    col：图片在大图中的列位置，从左到右1234
    pic_size :[3,3,3,3]代表共有3行4列小图
    """

    try:
        # 预处理
        dst = img_down.copy()
        hline_coef, b, c = _get_hline_coef(dst, row)
        vline_coef = _get_vline_coef(dst, col, pic_size)
        hline_coef[0][1], vline_coef[0][1]
        dot1 = [vline_coef[0][1], hline_coef[0][1]]  # 坐标格式（w,h）
        dot2 = [vline_coef[1][1], hline_coef[0][1]]
        dot3 = [vline_coef[0][1], hline_coef[1][1]]
        dot4 = [vline_coef[1][1], hline_coef[1][1]]
        dots = [dot1, dot2, dot3, dot4]
    except:
        logging.info('error in finding dots, use default dots position.')

        dots = [ERROR_DOTS_DICT['%s%i%i' %
                                (row, int(col), i)] for i in range(1, 5)]

    if verbose:
        print(dots)
        dot1, dot2, dot3, dot4 = dots
        plt.imshow(dst)
        plt.scatter(dot1[0], dot1[1], color="r")
        plt.scatter(dot2[0], dot2[1], color="r")
        plt.scatter(dot3[0], dot3[1], color="r")
        plt.scatter(dot4[0], dot4[1], color="r")
        plt.show()
    return dots


def _undistort(img, col):
    '''
    恢复图像的径向畸变
    '''
    global camera_parameter_pkl_path, distort_matrix
    path = os.path.dirname(os.path.abspath(__file__))
    camera_parameter_pkl_path = os.path.join(
        path, 'jn_camera_all.pkl')

    try:
        distort_matrix
    except:
        distort_matrix = pickle.load(open(camera_parameter_pkl_path, "rb"))

    mtx, dist = distort_matrix["camera" +
                               col]["mtx"], distort_matrix["camera" + col]["dist"]
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


def _perspective_transform(img_down, img, dots, h=1000):
    """
    对四个角点的透视变换，返回变换后图
    :img_down: 输入图像的下采样图
    img：输入图像原图
    h：获取所拼图的高
    """
    dots = 8 * np.array(dots)
    dots = np.float32(dots)
    w = h * 3 / 2
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(dots, dst)
    img_trans = cv2.warpPerspective(img, M, (int(w), int(h)))
    return img_trans


def _cut_and_paste_one_piece(img_down, img, row, col, h=1000, pic_size=[3, 3, 3, 3]):
    """
    透视变换整理以适配各种小图分布情况，返回变换后图
    :img_down: 输入图像的下采样图
    img：输入图像原图
    row：图片在大图中的行位置，从下至上ABC
    col：图片在大图中的列位置，从左到右1234
    h：获取所拼图的高
    pic_size :[3,3,3,3]代表共有3行4列小图
    """
    dots = _get_grid_corners(img_down, row, col, pic_size, verbose=False)
    piece1, piece2, piece3, piece4 = pic_size
    assert int(col) <= 4
    if col == "1":
        w = h / 2 * piece1
    elif col == "2":
        w = h / 2 * piece2
    elif col == "3":
        w = h / 2 * piece3
    else:
        w = h / 2 * piece4
    pers_transed_img = _perspective_transform(img_down, img, dots, h=1000)
    return pers_transed_img


def _get_hline_coef(img_down, row):  # 读入图务必是cv2.imread(pic,0)
    """
    拟合横向曲线，并返回对应坐标
     :img_down: 输入图像的下采样图
    col：图片在大图中的列位置，从上到下ABC
    pic_size :[3,3,3,3]代表共有3行4列小图
    """
    # 切边，做掩模
    ver_cut = int(img_down.shape[0] * 0.07)
    img_down = img_down * _get_mask(img_down)
    img_down = img_down[ver_cut + 5:-ver_cut, :]

    thre_bin = cv2.adaptiveThreshold(img_down.astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                     99, 55)

    positive_lines_bin = argrelmax(np.diff(thre_bin.mean(axis=1)), order=50)[
        0] + 5 + ver_cut

    # 预处理图片
    img_down = cv2.filter2D(img_down.astype(np.uint8),
                            cv2.CV_8U, np.ones((1, 11)) / 15)
    img_down = cv2.GaussianBlur(img_down, (11, 11), 1)

    # 获取三条横线的基础位置
    positive_lines = argrelmax(np.diff(img_down.mean(axis=1)), order=50)[
        0] + 5 + ver_cut
    negative_lines = argrelmin(np.diff(img_down.mean(axis=1)), order=50)[
        0] + 5 + ver_cut

    if len(thre_bin.mean(axis=1)[thre_bin.mean(axis=1) < 0.2]) < 3:
        # 二值化结果不可用
        total_lines = sorted(np.hstack((positive_lines, negative_lines)))
    else:
        total_lines = sorted(
            np.hstack((positive_lines, negative_lines, positive_lines_bin)))

    child_list = _get_child_list(total_lines)
    pn_lines = np.array([[np.mean(pn), len(pn)] for pn in child_list])

    assert len(pn_lines) > 2, 'basic lines has faced some problem'
    if len(pn_lines) == 3:
        basic_hline_pos = pn_lines.T[0]
    else:
        basic_hline_pos = _filter_outlier(pn_lines)
        basic_hline_pos = np.array(basic_hline_pos)

        assert len(basic_hline_pos) == 3, 'basic lines has faced some problem'

    basic_hline_pos = basic_hline_pos + 1
    hline_coef = np.vstack(([1e-5] * 3, basic_hline_pos))

    return hline_coef.T[[0, 2]], positive_lines, negative_lines


def _get_vline_coef(img_down, col, pic_size):
    """
    拟合纵向曲线，并返回对应坐标
     :img_down: 输入图像的下采样图
    col：图片在大图中的列位置，从左到右1234
    pic_size :[3,3,3,3]代表共有3行4列小图
    """
    piece1, piece2, piece3, piece4 = pic_size

    h, w = img_down.shape

    # 1. 自适应二值化找竖线
    thre = cv2.adaptiveThreshold(
        img_down, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
    kern = np.zeros((11, 11), dtype=np.uint8)
    kern[:, 5] = 1
    thre1 = cv2.dilate(thre, kern)
    thre1 -= 1

    # 2. 利用mask和自适应二值化的结果过滤掉多余的直线
    dilation = _get_mask(img_down)
    thre_new = thre1 * dilation

    # 3. 寻找竖线可能在的范围，放在x_dots里
    thre_sum = thre_new.sum(0) + np.random.randn(w) * 10e-3
    x_dots = argrelmax(thre_sum, order=50)[0]

    # 4. 根据x_dots的不同特点进行重新分类
    y_line = np.where(thre_new == 0)
    y_line = np.array([y_line[0], y_line[1]]).T
    stick = 5
    vline = {}
    vline["left"] = y_line[(y_line[:, 1] < x_dots[0] + stick)
                           * (y_line[:, 1] > x_dots[0] - stick)]

    if col == "2" or col == "3":
        vline["right"] = y_line[
            (y_line[:, 1] < x_dots[int(piece2)] + stick) * (y_line[:, 1] > x_dots[int(piece2)] - stick)]
    else:
        vline["right"] = y_line[
            (y_line[:, 1] < x_dots[int(piece1)] + stick) * (y_line[:, 1] > x_dots[int(piece1)] - stick)]

    def fit_f3_vlines(lines):
        '''拟合竖直一次曲线用'''

        def f_3(x, A, D):
            return A * x + D

        keep = int(len(lines) * 0.05)  # 去除异常值
        lines = lines[np.argsort(lines[:, 1])][keep:np.min([-keep, -1]), :]
        return optimize.curve_fit(f_3, lines[:, 0], lines[:, 1])[0]

    vline_coef = {}
    for i in ['left', 'right']:
        vline_coef[i] = fit_f3_vlines(vline[i])
    vline_coef = np.array([vline_coef['left'], vline_coef['right']])
    return vline_coef


def _get_mask(img_down):
    """
    用于找img_down的mask,为后面排除多余点集
    """
    th, ret = cv2.threshold(img_down, 120, 255, cv2.THRESH_OTSU)
    th, thre = cv2.threshold(img_down, th + 20, 255, cv2.THRESH_BINARY)
    thre = cv2.morphologyEx(thre, cv2.MORPH_CLOSE, np.ones((7, 7)))
    _, c, _ = cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 找函数点
    max_size_idx = np.array([cv2.contourArea(c1) for c1 in c]).argmax()
    convex_c = cv2.convexHull(c[max_size_idx])

    corner_list = _getCorners(np.squeeze(convex_c))
    diff_list = abs(corner_list.T[0][0] - corner_list.T[0][1]), abs(corner_list.T[0][2] - corner_list.T[0][3]), abs(
        corner_list.T[1][0] - corner_list.T[1][3]), abs(corner_list.T[1][1] - corner_list.T[1][2])
    if max(diff_list) < 15:
        thre_new = cv2.fillConvexPoly(np.zeros(thre.shape), convex_c, 1)
        kernel = np.ones((10, 10))
        dilation = cv2.dilate(thre_new, kernel, iterations=1)
    else:
        dilation = np.ones(img_down.shape)
    return dilation


def _pyrDown(img, n):
    '''用于指定下采样的比例，图片会被下采样2^n倍'''
    out = img.copy()
    for i in range(n):
        out = cv2.pyrDown(out)
    return out


def _get_child_list(test):
    child_list = []
    for index in range(len(np.where(np.diff(test) > 10)[0])):
        if index == 0:
            child_list.append(
                test[:np.where(np.diff(test) > 10)[0][index] + 1])
        else:
            child_list.append(
                test[np.where(np.diff(test) > 10)[0][index - 1] + 1:np.where(np.diff(test) > 10)[0][index] + 1])
    if np.where(np.diff(test) > 10)[0][index] != len(test) - 1:
        child_list.append(test[np.where(np.diff(test) > 10)[0][index] + 1:])
    return child_list


def _filter_outlier(pn_lines):
    tolerance = 70
    start_id = 0
    basic_hline_pos = [pn_lines.T[0][start_id]]
    while start_id < len(pn_lines) - 1:
        check_list = pn_lines.T[0][start_id + 1:]
        close_index = np.where(
            abs(check_list - tolerance - basic_hline_pos[-1]) < 5)[0]
        if len(close_index) != 0:
            close_pos_num = [pn_lines.T[1][start_id + 1:][i]
                             for i in close_index]
            basic_hline_pos.append(
                check_list[close_index[np.argmax(close_pos_num)]])
            start_id = start_id + 1
        elif len(basic_hline_pos) > 1:
            break
        else:
            start_id = start_id + 1
            basic_hline_pos = [pn_lines.T[0][start_id]]
    return basic_hline_pos


def _calcLeaveDist(pt0, pt1, points):
    '''
    用于计算点到直线的距离，其中直线由pt0和pt1两点确定
    param:
        pt0, pt1 距离最远两个点
        points 连通分量上点
    return:
        d 距离
    '''
    x0, y0 = pt0
    x1, y1 = pt1
    xc, yc = points
    A = 1 / (x1 - x0 + 1e-8)
    B = -1 / (y1 - y0 + 1e-8)
    C = y0 / (y1 - y0 + 1e-8) - x0 / (x1 - x0 + 1e-8)
    d = (A * xc + B * yc + C) / np.sqrt(A ** 2 + B ** 2)

    return d


def _getCorners(contour_dots):
    '''
    输入contours, 获取两边最远的点
    '''
    # 寻找contour上最远的两个点
    cross_term = contour_dots.dot(contour_dots.T)
    dist = np.power(contour_dots, 2).sum(1) + np.power(contour_dots, 2).sum(1).T - \
        2 * contour_dots.dot(contour_dots.T)
    max_dist_idx = dist.argmax()
    ncd = contour_dots.shape[0]
    max_dist_idx = [int(max_dist_idx / ncd), max_dist_idx % ncd]
    corner1, corner2 = contour_dots[max_dist_idx]

    # 计算contour上每个点到直线的距离，区分正负
    leave_dist = _calcLeaveDist(corner1, corner2, contour_dots.T)
    leave_dist = leave_dist + np.random.rand(len(leave_dist)) * 1e-8
    corner3, corner4 = contour_dots[[leave_dist.argmax(), leave_dist.argmin()]]

    corner = np.array([corner1, corner2, corner3, corner4])
    median = np.median(corner.T, axis=1)
    bool_med = np.sum((corner < median) *
                      np.array([1, 2]), axis=1).argsort()[[3, 1, 0, 2]]

    return corner[bool_med]


ERROR_DOTS_DICT = {'A11': [15.093462550496637, 35.96125907990315],
                   'A12': [228.23932366492517, 35.96125907990315],
                   'A13': [15.093462550496637, 175.20338983050848],
                   'A14': [228.23932366492517, 175.20338983050848],
                   'A21': [13.933712858619362, 32.663438256658594],
                   'A22': [228.55683980990472, 32.663438256658594],
                   'A23': [13.933712858619362, 175.63438256658597],
                   'A24': [228.55683980990472, 175.63438256658597],
                   'A31': [15.38556442221209, 29.346246973365616],
                   'A32': [230.2833992186749, 29.346246973365616],
                   'A33': [15.38556442221209, 174.61501210653753],
                   'A34': [230.2833992186749, 174.61501210653753],
                   'A41': [16.070576377632644, 33.43341404358353],
                   'A42': [230.1099032844763, 33.43341404358353],
                   'A43': [16.070576377632644, 173.6004842615012],
                   'A44': [230.1099032844763, 173.6004842615012],
                   'B11': [15.422953266717009, 34.87469879518072],
                   'B12': [227.89568686378027, 34.87469879518072],
                   'B13': [15.422953266717009, 174.06024096385542],
                   'B14': [227.89568686378027, 174.06024096385542],
                   'B21': [14.285406209054146, 31.36144578313253],
                   'B22': [227.92456190386886, 31.36144578313253],
                   'B23': [14.285406209054146, 174.15903614457832],
                   'B24': [227.92456190386886, 174.15903614457832],
                   'B31': [15.659333043009106, 27.983132530120482],
                   'B32': [229.67410996480777, 27.983132530120482],
                   'B33': [15.659333043009106, 173.04578313253012],
                   'B34': [229.67410996480777, 173.04578313253012],
                   'B41': [16.43843731738943, 32.11807228915663],
                   'B42': [229.5014311596249, 32.11807228915663],
                   'B43': [16.43843731738943, 172.0674698795181],
                   'B44': [229.5014311596249, 172.0674698795181],
                   'C11': [14.69220230019799, 34.390361445783135],
                   'C12': [228.07966105858756, 34.390361445783135],
                   'C13': [14.69220230019799, 172.94939759036146],
                   'C14': [228.07966105858756, 172.94939759036146],
                   'C21': [13.61257768850209, 30.903614457831324],
                   'C22': [228.0605001470102, 30.903614457831324],
                   'C23': [13.61257768850209, 173.0265060240964],
                   'C24': [228.0605001470102, 173.0265060240964],
                   'C31': [15.042191408040424, 27.426506024096387],
                   'C32': [229.7917299466244, 27.426506024096387],
                   'C33': [15.042191408040424, 171.78313253012047],
                   'C34': [229.7917299466244, 171.78313253012047],
                   'C41': [15.917923073978574, 31.790361445783134],
                   'C42': [229.52325960229305, 31.790361445783134],
                   'C43': [15.917923073978574, 170.86746987951807],
                   'C44': [229.52325960229305, 170.86746987951807]}
