#-*- coding:utf-8 -*-
import cv2
import os
import numpy as np
from scipy import signal
import logging
import time


def fix_jk_cengqian(line_ver, line_hor, num_hor_pieces, num_ver_pieces):
    '''
    对应晶科层前图的修复utils。
    晶科层前图一般会在四周检不出线条，所以用中间的来修复四个周边值。
    在修复之前，还需要保证中间的5×11条线都被检出。
    '''
    

    
    ## 保证所有该检出的线都被检出。只考虑少一条线的情况，如果少了两条线，或者多了，就调用之前的切图函数。
    def fix_lines(lines, num_pieces):
        lines_diff = np.diff(lines, n = 1)
        if len(lines) == num_pieces - 2:  
            ##########第一种情况：首尾缺线###############################
            if np.std(lines_diff) < 8:   
                print('type1: first/last line lost')
                if lines[0] - 2 * np.mean(lines_diff) > 0: # 情况1.1:第一根线没检出来
                    lines = np.hstack((int(lines[0] -  np.mean(lines_diff)), 
                                       lines))
                else:   # 情况1.2：最后一根线没检出来
                    lines = np.hstack((lines, int(lines[-1] + np.mean(lines_diff))))
            ############第二种情况：中间缺线#############################
            else:
                print('type2: internal line lost')
                lines_diff_list = list(lines_diff)
                lost_line_id = lines_diff_list.index(max(lines_diff_list))
                lines = np.hstack((lines[:lost_line_id + 1], 
                                   int(np.mean(lines[lost_line_id:lost_line_id + 2])), 
                                   lines[lost_line_id + 1:]))
        else:
            raise Exception('算法不太适应这张图，调用之前的切图函数来解决它吧。如果并非晶科的图片，那就先放弃吧。')
            pass
        return lines
    
    ## 修复原始的横纵线条
    if len(line_hor) < num_hor_pieces - 1:
        print('hor less')
        line_hor = fix_lines(line_hor, num_hor_pieces)
    if len(line_ver) < num_ver_pieces - 1:
        print('ver_less')
        line_ver = fix_lines(line_ver, num_ver_pieces)
    
    ## 修复两头的线条
    line_hor_diff = np.diff(line_hor, n = 1)
    line_ver_diff = np.diff(line_ver, n = 1)
    #assert np.std(line_hor_diff[:11]) < 8 and len(line_hor) == num_hor_pieces - 1, '水平方向切割出错'
    #assert np.std(line_ver_diff[:5]) < 30 and len(line_ver) == num_ver_pieces - 1, '垂直方向切割出错'
    
    # 用avg来生成第一个值和最后一个值
    # 晶科10片会在头尾提前检出不对的线条
    
    hor_first_line, hor_last_line = line_hor[0] - int(np.median(line_hor_diff[1:3])), line_hor[-1] + int(np.median(line_hor_diff[-3:-1]))
    ver_first_line, ver_last_line = line_ver[0] - int(np.median(line_ver_diff[1:3])), line_ver[-1] + int(np.median(line_ver_diff[-3:-1]))
    line_hor = np.hstack((hor_first_line, line_hor, hor_last_line))
    line_ver = np.hstack((ver_first_line, line_ver, ver_last_line))
    
    return line_hor, line_ver
    

def fix_ats_cenghou(line_ver, line_hor, num_hor_pieces = 12, num_ver_pieces = 6):
    '''
    对应阿特斯层后图的修复utils。
    阿特斯层后图一般会在左右的边缘出现问题，可能会少检，所以用中间的来修复周边的。
    '''
    line_hor_diff = np.diff(line_hor, n = 1)
    line_ver_diff = np.diff(line_ver, n = 1)
    #assert np.std(line_hor_diff[:11]) < 8
    #assert np.std(line_ver_diff[:5]) < 30
    gap_hor_avg = int(np.mean(line_hor_diff[1:-1]))
    gap_ver_avg = int(np.mean(line_ver_diff[1:-1]))
    line_hor[0] = line_hor[1] - gap_hor_avg
    if len(line_hor) <= num_hor_pieces:
        line_hor = np.hstack((line_hor, line_hor[num_hor_pieces - 1] + gap_hor_avg))
    line_hor[num_hor_pieces] = line_hor[num_hor_pieces - 1] + gap_hor_avg
    line_ver[0] = line_ver[1] - gap_ver_avg
    if len(line_ver) <= num_ver_pieces:
        line_ver = np.hstack((line_ver, line_ver[num_ver_pieces - 1] + gap_ver_avg))
    line_ver[num_ver_pieces] = line_ver[num_ver_pieces - 1] + gap_ver_avg

    #assert len(line_hor) == 13
    #assert len(line_ver) == 7
    
    return line_hor, line_ver 

def get_cut_line(img, num_hor_pieces = 12, num_ver_pieces = 6, auto_fix = None):
    tic = time.time()
    order_hor = num_hor_pieces + 2
    order_ver = num_ver_pieces + 2
    # 如果是10片，就在边缘切掉10个像素
    if num_hor_pieces == 10:
        img = img[:, 50:-50]
    ## 1.。。。
    # horizontal
    horizontal = np.sum(img, axis = 0)
    line_hor = signal.argrelmin(horizontal, order = int(len(horizontal) / order_hor))[0]
    # vertical
    vertical = np.sum(img, axis = 1)
    line_ver = signal.argrelmin(vertical, order = int(len(vertical) / order_ver))[0]
    toc1 = time.time()
    
    #对于10片组件，再把10像素偏移量加回来
    if num_hor_pieces == 10:
        line_hor = line_hor + 50
    
    # 根据传入的图片厂家不同，采用不同的修复方案
    if auto_fix == 'ats_cenghou':
        line_hor, line_ver = fix_ats_cenghou(line_ver, line_hor, num_hor_pieces, num_ver_pieces)
        
    if auto_fix == 'jk_cengqian':
        line_hor, line_ver = fix_jk_cengqian(line_ver, line_hor, num_hor_pieces, num_ver_pieces)
    
    
    toc2 = time.time()
    print('first step time is %.2f secs.' % (toc1 - tic), 'auto fix time is %.2f secs.' % (toc2 - toc1))

    return line_hor, line_ver 
