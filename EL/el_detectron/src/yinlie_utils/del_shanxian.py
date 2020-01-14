#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import cv2
import os, shutil
# import matplotlib.pyplot as plt
import scipy.signal as ss
import os , sys 


d = os.path.dirname(__file__)  
parent_path  = os.path.dirname(d)
parent_path  = os.path.dirname(parent_path)
sys.path.append(parent_path)

import src.yinlie_utils.shanxian_utils as su 

def del_shanxian_simple(img,bx,verbose = True):
    # deal with img
    img = img*(np.mean(img)/np.mean(img,axis = 0))
    img = np.clip(img,0,255)
    img = np.uint8(img)
    img_fill = img.copy()
    xmin,ymin,xmax,ymax = bx 
    shanxians = su.find_shanxian_row(img)[1]

    # 判断栅线是否在ymin - ymax 中间，如果是，则把栅线两侧的位置进行均值处理
    # 只对框内的区域进行inpaint
    # 思路：
    # 构建一个全局的mask    
    mask = np.zeros(shape =  img_fill.shape )

    # 然后把栅线的位置填充掉

    for shanxian in su.find_shanxian_row(img_fill)[1]:
        mask[shanxian - 8 : shanxian + 8] = 1

    # 取出mask的局部
    sub_mask = mask[ymin:ymax,xmin:xmax]
    sub_mask = np.uint8(sub_mask)
    # inpaint

    sub_img_fill = cv2.inpaint(img_fill[ymin:ymax,xmin:xmax],sub_mask,5,cv2.INPAINT_NS)

    # 放进原图
    img_fill[ymin:ymax , xmin:xmax] = sub_img_fill
    
#     if verbose:
#         subimg  = img_fill.copy()
#         subimg  = cv2.rectangle(subimg, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
#         plt.imshow(subimg)
#         plt.show()



    return img_fill

def del_shanxian(img):
    img = img*(np.mean(img)/np.mean(img,axis = 0))
    img = np.clip(img,0,255)
    img = np.uint8(img)
    img_fill = img.copy()
    shanxians = su.find_shanxian_row(img)[1]
    h,w = img.shape
    img_bool = np.zeros((h,w))
    for s in shanxians:
        img_bool[s-8:s+8] = 255
 

    img_bool = np.uint8(img_bool)
    img_fill = cv2.inpaint(img,img_bool,5,cv2.INPAINT_NS)
    #img_fill = img_fill*(np.mean(img_fill)/np.mean(img_fill,axis = 0))
    return img_fill
