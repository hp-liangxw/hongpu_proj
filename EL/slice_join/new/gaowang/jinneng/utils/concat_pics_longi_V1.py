import numpy as np
import cv2, glob, os, re
import pickle
import logging

global GOLDEN_UNIT_DICT, GOLDEN_CORNER_DICT
with open(os.path.join(os.path.dirname(__file__), 'ats_tawan.pkl'), 'rb') as f:
    golden_parameters = pickle.load(f)

GOLDEN_UNIT_DICT = golden_parameters['GOLDEN_UNIT_DICT']
GOLDEN_CORNER_DICT = golden_parameters['GOLDEN_CORNER_DICT']

def _pyrDown(img, n):
    for i in range(n):
        img = cv2.pyrDown(img)
    return img

def _undistort(img, col):
    '''
    恢复图像的径向畸变
    '''
    global camera_parameter_pkl_path, distort_matrix
    path = os.path.dirname(os.path.abspath(__file__))
    camera_parameter_pkl_path = os.path.join(path, 'ats_camera_all.pkl')
    
    try:
        distort_matrix
    except:
        distort_matrix = pickle.load(open(camera_parameter_pkl_path, "rb" )) 
    
    
    mtx,dist = distort_matrix["camera"+col]["mtx"],distort_matrix["camera"+col]["dist"]
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst

def get_unbiased_corners(img, row_symbol, col_num, n_pyrDown = 1):
    x_pad, y_pad = np.uint([50, 100] / np.power(2, n_pyrDown))#
    templ = GOLDEN_UNIT_DICT['%s_%i' % (row_symbol, col_num)][y_pad:-y_pad, x_pad:-x_pad]
    corners = GOLDEN_CORNER_DICT['%s%i' % (row_symbol, col_num)]
    
    img = _pyrDown(img, n_pyrDown)
    mt = cv2.matchTemplate(img.astype(np.uint8), templ.astype(np.uint8), cv2.TM_CCOEFF_NORMED) 
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(mt)
    xbias, ybias = maxloc #

    xbias -= x_pad
    ybias -= y_pad
    unbiased_corners = corners + np.array([xbias, ybias])
#     xs, ys = unbiased_corners.T
    if maxval < 0.8:#
        return corners * np.power(2, n_pyrDown)
    return unbiased_corners * np.power(2, n_pyrDown)

def joint_image_2(img_list, row_symbol, module_type = 'half', piece_list = [4,6,6,4]):
    """
    去边并水平拼接图像
    :param img_list: 图片矩阵的列表
    :module_type: 半片or全片组件
    :piece_list: 四段分段图分别有几列组件
    returns: 去边并水平拼接好的图像
    """
    # generate the output size of 4 pieces
    h = 1000
    if module_type == 'half':
        w = 250 * np.array(piece_list)
    elif module_type == 'full':
        w = 500 * np.array(piece_list)
        
    concat_4 = np.zeros((h , 5), dtype = np.uint8)
    for i in range(4):
        img_dst = _undistort(img_list[i], str(i+1))
        if len(img_dst.shape) == 3:
            img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
        corners = get_unbiased_corners(img_dst, row_symbol, i + 1)
        pers_transed_img = perspective_transform_one_image(img_dst, corners, h, w[i])
        pers_transed_img = np.hstack((pers_transed_img, np.ones((h, 5))))
        concat_4 = np.hstack([concat_4, pers_transed_img])

    w = concat_4.shape[1]
    sep = np.ones((5, w)) 
    concat_4 = np.vstack((sep, concat_4, sep))
    return concat_4


def perspective_transform_one_image(img, dots, h, w):
    dst = np.float32( [ [0,0],[w,0],[w,h],[0,h] ] )
    M = cv2.getPerspectiveTransform(dots.astype(np.float32), dst)
    img_trans = cv2.warpPerspective(img, M, (w, h))
    return img_trans
