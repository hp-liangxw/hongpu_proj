#-*- coding:utf-8 -*-
import cv2
# import matplotlib.pyplot as plt
import os
import numpy as np
import time
import scipy.signal as signal

# #get the picture information
# def imshow(img):
#     plt.imshow(img,'gray')
# def imghist(img):
#     hist = cv2.calcHist([img],[0],None,[256],[0,256])
#     plt.plot(hist)
# def imshownhist(img):
#     plt.subplot(121)
#     imshow(img)
#     plt.subplot(122)
#     imghist(img)
    
def find_shanxian_row(img, shanxian_num = 5, order_percentage = 0.113):
    # 1.get the sharpened pic and count the column avg
    rows = np.mean(img, axis = 1)
    h, w = img.shape
    # 2.fill the start and end
    l = int(0.04 *  h)
    pads = np.ones(l) * np.mean(rows)
    rows = np.hstack((pads, rows[l:-l], pads))
    '''rows[:l] = np.mean(rows)
    rows[-l:] = np.mean(rows)'''
    # 3.find shanxian row
    order = int(img.shape[0] * order_percentage)
    shanxian_row = signal.argrelmin(rows,order = order)[0]
    num_row = len(shanxian_row)

    return num_row, shanxian_row, None
