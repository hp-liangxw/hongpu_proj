# -*- coding:utf-8 -*-
# 准备两个基础方法
import logging
import cv2
import numpy as np
import scipy.stats as stats
# import matplotlib.pyplot as plt
from skimage import measure
from scipy.signal import argrelmax
import sys, os 

d = os.path.dirname(__file__)  
parent_path = os.path.dirname(d)
parent_path = os.path.dirname(parent_path)
sys.path.append(parent_path)

import src.yinlie_utils.grid_cut_utils as cut
import src.yinlie_utils.shanxian_utils as su
from src.yinlie_utils.del_shanxian import *



# from matplotlib.pyplot import subplot as sb
# from matplotlib.pyplot import imshow as sh

from src.yinlie_utils.shanxian_utils import find_shanxian_row

from collections import Counter
from scipy.spatial import distance as dist

import math

def get_coord(rps, label, option=1):
    min_row, min_col, max_row, max_col = rps[label-1].bbox
    if option == 1:
        x1,y1 = min_col,min_row
        x2,y2 = max_col,max_row
    else:
        # 右上x1,y1，左下x2,y2
        x1, y1 = max_col, min_row
        x2, y2 = min_col, max_row
        
    return x1, y1, x2, y2

def relative_coord(x1, y1, x2, y2, w, h):
        # bbox 左上，右下
        return 'left' if x1/w < (1-x2/w) else 'right'

def rectContains(rect, pt):
     # bbox 左上，右下
    logic = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
    return logic
    
def getYinlieInfo(subimg, bx, verbose=True, padding=10, thre=0.3, del_sx_simple=True):
    
    # 去除栅线
    if del_sx_simple:
        del_img = del_shanxian_simple(subimg, bx, verbose)
    else:
        del_img = del_shanxian(subimg)
    
    # 拿到坐标
    xmin, ymin, xmax, ymax = bx 
    
    # 进行坐标放缩
    ymin = max(ymin, padding)
    ymax = min(ymax, subimg.shape[0] - padding)

    # get shanxian
    shanxian = del_img[ymin:ymax, xmin:xmax]

    size = int(0.1 * min(shanxian.shape[:2]) )
    w, h = shanxian.shape[:2]

    kernel1 = np.diag([1.0]*size) / size
    kernel2 = np.flip(kernel1, 1)

    if w < h:

        kernel1 = cv2.resize(kernel1, (max(1, int(w * 1.0 *size/ h )), size))
        kernel2 = cv2.resize(kernel2, (max(1, int(w * 1.0 *size/ h )), size))

    else:
        kernel1 = cv2.resize(kernel1, (size, max(1, int(w * 1.0 *size/ h ))))
        kernel2 = cv2.resize(kernel2, (size, max(1, int(w * 1.0 *size/ h ))))      

    kernel1 = kernel1 / kernel1.sum()
    kernel2 = kernel2 / kernel2.sum()


    # get 2 blur
    blur1 = cv2.filter2D(shanxian, -1, kernel1)
    blur2 = cv2.filter2D(shanxian, -1, kernel2)


    if max( (w * 1.0 / h ) , (h * 1.0 / w )) > 1.75:
        shanxian_c = shanxian.copy()
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #shanxian_c = cv2.morphologyEx(shanxian_c,cv2.MORPH_CLOSE,k)
        shanxian_c = cv2.dilate(shanxian_c, k)
        ret1 = ret2 =  255 -  cv2.adaptiveThreshold(shanxian_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
    else:
        # get 2 binary image
        ret1  = 255 -  cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
        ret2  = 255 -  cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
    
    index_i = 0 
    labels1 = measure.label(ret1,connectivity=2)
    labels2 = measure.label(ret2,connectivity=2)
    labeldic1 = Counter(labels1.flatten())
    labeldic2 = Counter(labels2.flatten())
    maxArea1Dic = sorted(labeldic1.items() , key=lambda x:x[1])
    maxArea2Dic = sorted(labeldic2.items() , key=lambda x:x[1])
    rps = measure.regionprops(labels1, cache=False)
    min_row, min_col, max_row, max_col = rps[maxArea1Dic[-2][0]-1].bbox
    area = max(maxArea1Dic[-2][1], maxArea2Dic[-2][1]) /((xmax-xmin)*(ymax-ymin))
    #     print(maxArea1Dic[-2][1],maxArea2Dic[-2][1])
    if area > thre or min(len(maxArea1Dic), len(maxArea2Dic)) <= 6:
        labels1 = (labels1 == maxArea1Dic[-2][0])
        labels2 = (labels2 == maxArea2Dic[-2][0])
        if labels1.sum()> labels2.sum():
            labels3 = labels1
        else:
            labels3 = labels2
    
    elif max((w * 1.0 / h ), (h * 1.0 / w )) > 2.5:
        rps = measure.regionprops(labels1, cache=False)
        aa = [r.area for r in rps]
        index = [maxArea1Dic[-i][0] for i in range(2,7) if maxArea1Dic[-i][1]/max(aa) > 0.3]
        labels3 = np.in1d(labels1, index)
        labels3 = labels3.reshape(labels1.shape)
        
        
    else:  
        # first
        rps = measure.regionprops(labels1, cache=False)
        aa = [r.area for r in rps]
        #tuple 
        index = [maxArea1Dic[-i][0] for i in range(3,7) if maxArea1Dic[-i][1]/max(aa) >0.3]
        distance = 100000000
        max_x1, max_y1, max_x2, max_y2 = get_coord(rps,maxArea1Dic[-2][0])
        max_rel = relative_coord(max_x1, max_y1, max_x2, max_y2, h, w)
        if max_rel == 'left':
            for i in range(len(index)):
                x1,y1,x2,y2 = get_coord(rps,index[i])
                dists = dist.euclidean((max_x2, max_y2), (x1, y1))
                center_x, center_y = (x2+x1)/2, (y1+y2)/2
                #if dists < distance and x1 >= max_x2 and y1 >= max_y2:
                if (dists < distance and center_x +10 >= max_x2 and center_y +10 >= max_y2) or (rps[index[i]-1].area>max(aa)>0.5):
                    distance = dists
                    index_i = index[i]

        elif max_rel == 'right':
            for i in range(len(index)):
                x1, y1, x2, y2 = get_coord(rps, index[i])
                dists = dist.euclidean((max_x1, max_y1), (x2, y2))
                center_x, center_y = (x2+x1)/2, (y1+y2)/2
                #if dists < distance and x2 <= max_x1 and y2 <= max_y1:
                if (dists < distance and center_x -10 <= max_x1 and center_y -10 <= max_y1) or  (rps[index[i]-1].area>max(aa)>0.5):
                    distance = dists
                    index_i = index[i]

                    
        # draw the mask
        if index_i != 0:
            labels3 = ((labels1 == index_i)+ (labels1 == maxArea1Dic[-2][0]))
        else:
            labels3 = (labels1 == maxArea1Dic[-2][0])
            
        if maxArea1Dic[-2][1] <= maxArea2Dic[-2][1]:
              
            # choose second
            rps = measure.regionprops(labels2, cache=False)
            aa = [r.area for r in rps]
            #tuple 
            index = [maxArea2Dic[-i][0] for i in range(3,7) if maxArea2Dic[-i][1]/max(aa) >0.3]
            distance = 100000000
            max_x1, max_y1, max_x2, max_y2 = get_coord(rps, maxArea2Dic[-2][0], 2)
            max_rel = relative_coord(max_x1, max_y2, max_x2, max_y1, h, w)
            if max_rel == 'left':
                for i in range(len(index)):
                    x1, y1, x2, y2 = get_coord(rps, index[i], 2)
                    dists = dist.euclidean((max_x1, max_y1), (x2, y2))
                    center_x, center_y = (x2+x1)/2, (y1+y2)/2
                    #if dists < distance and x2 >= max_x1 and y2 <= max_y1:
                    if (dists < distance and center_x +10 > max_x1 and center_y -10 < max_y1) or (rps[index[i]-1].area>max(aa)>0.5):
                        distance = dists
                        index_i = index[i]


            elif max_rel == 'right':
                for i in range(len(index)):
                    x1, y1, x2, y2 = get_coord(rps, index[i], 2)
                    dists = dist.euclidean((max_x2, max_y2), (x1, y1))
                    center_x, center_y = (x2+x1)/2, (y1+y2)/2
                    #if dists < distance and x1 <= max_x2 and y1 >= max_y2:
                    if (dists < distance and center_x -10 <= max_x2 and center_y +10 >= max_y2) or (rps[index[i]-1].area>max(aa)>0.5):
                        distance = dists
                        index_i = index[i]

            # draw the mask
            if index_i != 0:
                labels3 = ((labels2 == index_i)+ (labels2 == maxArea2Dic[-2][0]))  
            else:
                labels3 = (labels2 == maxArea2Dic[-2][0])

    kernel = np.ones((3,3),np.uint8)  
    labels3 = cv2.bilateralFilter(labels3.astype(np.uint8), 9, 75, 75)
    erosion = cv2.erode(labels3, kernel, iterations=1)
    labels3 = cv2.dilate(erosion, kernel, iterations=1)

    yinlieMask = np.zeros(shape = subimg.shape[:2])
    yinlieMask[ymin:ymax, xmin:xmax] = labels3
    
#     if verbose == True:
#         plt.imshow(del_img[ymin:ymax,xmin:xmax])
#         plt.show()
#         sb(1,2,1)
#         sh(blur1)
#         sb(1,2,2)
#         sh(blur2)
#         plt.show()
#         sb(1,2,1)
#         sh(ret1)
#         sb(1,2,2)
#         sh(ret2)
#         plt.show()
#         sh(labels3)
#         plt.show()
  
    return yinlieMask, shanxian, labels3

def getConvexDefectInfo(shanxian, subimg_mask, verbose=False):
    subimg_mask = subimg_mask.astype(np.int16)
    
    # cv2方法
    yinlie = subimg_mask.copy()
    yinlie = np.uint8(yinlie)
    h, w = subimg_mask.shape
    yinlie = np.uint8(yinlie*255)
    img, contours, hierarchy = cv2.findContours(yinlie, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    
    # 筛选轮廓
    contours = [cnt for cnt in contours if cnt.shape[0] > 2]
    
    # 画出轮廓
    painter = np.uint8(np.ones((h, w))) 
    cv2.drawContours(painter, contours, -1, (0,0,255), 1)
    
    # 画出凸包
    hulls = [cv2.convexHull(contour) for contour in contours]
    cv2.drawContours(painter, hulls, -1, (0,255,0), 1)
    
    # 计算凸包面积，轮廓面积，轮廓凸包面积比
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    hull_areas = [cv2.contourArea(hull) for hull in hulls]
    solidity = np.sum(contour_areas) / np.sum(hull_areas)    
     
    # 计算距离
    hulls_on_contour = [cv2.convexHull(contour, returnPoints=False) for contour in contours]
    defects = [cv2.convexityDefects(contours[i], hulls_on_contour[i]) for i in range(len(contours))]
    
    depths = [defect[:, 0] for defect in defects if type(defect) != type(None)]
    depth = [i for j in depths for i in j]
    
    # 将多块mask的contour合并
    cnt = np.concatenate(contours, axis=0)
    
    # 计算旋转角度，确定隐裂走向
    (center_x, center_y), (w, h), _ = cv2.minAreaRect(cnt)  
        
    # 距离变换
    dt = cv2.distanceTransform(subimg_mask.astype(np.uint8), 1, 5)

    # 拟合直线
    [vx, vy, x, y] = cv2.fitLine(np.float32(cnt), cv2.DIST_L1, 0, 0.01, 0.01)

    angle = np.arctan2(vy, vx) * 180 / np.pi
    
    # 直接使用distanceTransform 来计算l1 信息
    loc_simple = np.argwhere(dt != 0).copy()
    temp = []
    for  i in range(len(loc_simple)):
        temp.append(dt[loc_simple[i][0], loc_simple[i][1] ])
    # loc 的第三列为权值
    loc_simple = np.hstack([loc_simple, np.array(temp).reshape(-1,1)])
    # 计算l1 和 l2
    l1_simple = np.max(
        np.abs(
                vx*(loc_simple[:, 0]-y)-vy*(loc_simple[:, 1]-x)
            ) * loc_simple[:, 2] 
        ) / np.max(loc_simple[:, 2])
    l2_simple = np.sum(
        np.power(
            vx*(loc_simple[:, 0]-y)-vy*(loc_simple[:, 1]-x) , 2
            ) * loc_simple[:, 2] 
        ) / loc_simple[:, 2].sum()

    # 显示
    if verbose:
        print('solidity: {}'.format(solidity))
        print('rotate angle: {}'.format(angle))
        print('max diviation: {}'.format(l1_simple))
        print('l2 diviation: {}'.format(l2_simple))
        
        rows, cols = shanxian.shape[:2]
        lefty=int(-(x*vy/vx)+y)
        righty=int(((cols-x)*vy/vx)+y)

#         skeleton_with_line = np.uint8(np.ones(dt.shape))
#         skeleton_with_line[np.uint8(loc_simple[:, 0]), np.uint8(loc_simple[:, 1])] = 255
#         img = cv2.line(skeleton_with_line, (cols-1, righty), (0,lefty), (0,0,0), 1)
#         sb(2,2,1)
#         sh(shanxian)
#         sb(2,2,2)
#         sh(painter)
#         sb(2,2,3)
#         sh(dt)
#         sb(2,2,4)
#         sh(img)
#         plt.show()

    return depth, contour_areas, hull_areas, solidity, angle, l1_simple, l2_simple

def getThicknessInfo(subimg, bx ,shanxian, labels3, verbose=False, thick=2.5, thres=10):

    thick = 1.0 * thick * subimg.shape[1] / 630.0

    xmin, ymin, xmax, ymax = bx
    shanxians = find_shanxian_row(subimg)[1]
    # 首先：获取 l1 距离

    depth, contour_areas, hull_areas, solidity, angle, average_diviation_l1, average_diviation_l2 = getConvexDefectInfo(shanxian, labels3, verbose)

    # 判断隐裂形态是否找准
    # 计算labels3 中的有效面积占比
    labels3_not_zero_x = np.argwhere(labels3 !=0)[:,0]
    labels3_not_zero_y = np.argwhere(labels3 !=0)[:,1]
    mask_xmin = labels3_not_zero_x.min()
    mask_xmax = labels3_not_zero_x.max()
    mask_ymin = labels3_not_zero_y.min()
    mask_ymax = labels3_not_zero_y.max()

    areaRatio =  min( 1.0 * (mask_ymax - mask_ymin) / labels3.shape[1], 1.0 * (mask_xmax - mask_xmin) / labels3.shape[0]) 
    # h/w比
    ratio_y_x  = 1.0 * (ymax - ymin) / (xmax - xmin)
    
    Status = -1
    logging.info('this yinlie angle is {}'.format(angle))
    logging.info('this yinlie is straight or not {}, {}'.format(average_diviation_l1, solidity))
    logging.info('if is a thin defect {}'.format((xmax - xmin ) < 0.3 * np.diff(shanxians).mean()))
    logging.info('yinlie hw ratio {}'.format(ratio_y_x))
    
    if abs(angle) > 80 and (solidity > 0.7 or (average_diviation_l1 < thick) or (average_diviation_l2 < thick)):
        if ymax - ymin >= 0.5 * subimg.shape[0]:
            if ratio_y_x > 2 and (xmax - xmin ) < 0.5 * np.diff(shanxians).mean():
                Status = False
            else:
                Status = True
        elif ymax <= shanxians[0] or ymin >= shanxians[-1]:
            if ratio_y_x > 1.3 and (xmax - xmin ) < 0.3 * np.diff(shanxians).mean():
                Status = False
            else:
                Status = True
        elif ratio_y_x > 1.5 and (xmax - xmin ) < 0.3 * np.diff(shanxians).mean():
            Status = False
        else:
            Status = True
    else:
        Status = True
    
    if verbose:
        print('areaRatio: {}'.format(areaRatio))
        print('ratio_y_x: {}'.format(ratio_y_x))
        print('w: {}, th: {}'.format(xmax - xmin, np.diff(shanxians).mean()))
        print('h: {}, th: {}'.format(ymax - ymin, subimg.shape[0]))
    
    return Status, solidity, angle, ratio_y_x
