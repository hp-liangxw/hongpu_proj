# coding:utf-8
## 组件切图（不含层后正畸、串检）
# import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import glob
import time
import logging
from scipy import signal

class ElPreCuter:
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    def __init__(self):
        self._scale = 4
        tempath = self._template_dir()
        all_templs = glob.glob(os.path.join(tempath, '*'))
        self.all_templs = []  
        for tmpl in all_templs:
            template = {}
            for tmpimg in glob.glob(os.path.join(tempath, "{}/*.png".format(tmpl))):
                tmpname = os.path.basename(tmpimg).split('.')[0]
                templimg = cv2.imread(tmpimg,cv2.IMREAD_COLOR)
                templimg = cv2.resize(templimg, (int(templimg.shape[1]/self._scale), int(templimg.shape[0]/self._scale)) )
                template[tmpname] = templimg
            self.all_templs.append(template)

        self.last_templ = 0

    def _template_dir(self):
        return os.path.join(os.path.abspath(os.path.dirname(__file__)),'temps')

    def _tempates2mathch(self):
        rv = list(range(len(self.all_templs)))
        del rv[self.last_templ]
        rv.insert(0, self.last_templ)
        return rv

    def _is_good_corners(self,corners, w, h):

        tl = corners['tl'][0]
        tr = corners['tr'][0]
        bl = corners['bl'][0]
        br = corners['br'][0]

        if tr[0] - tl[0] < w*2/3:
            return False
        if abs(tr[1] - tl[1]) > h/12:
            return False

        if br[0] - bl[0] < w*2/3:
            return False
        if abs(br[1] - bl[1]) > h/12:
            return False

        if bl[1] - tl[1] < h*2/3:
            return False
        if abs(bl[0] - tl[0]) > w/16:
            return False

        if br[1] - tr[1] < h*2/3:
            return False
        if abs(br[0] - tr[0]) > w/16:
            return False

        return True

    def _get_corners_of_templ(self, img, tmpl):
        corners = {}
        for tmpname, templimg in self.all_templs[tmpl].items():
            meth = 'cv2.TM_SQDIFF'
            method = eval(meth)
            tic = time.time()
            d,w,h = templimg.shape[::-1]

            # 将图片切分成 3 行 4 列 12个分区， 只在对于的分区进行模板匹配
            rows, cols = 3, 4
            min_x, min_y, max_x, max_y = 0, 0 ,img.shape[1] ,img.shape[0]
            if tmpname=='tl':
                min_x, min_y, max_x, max_y = 0, 0 ,img.shape[1]//cols ,img.shape[0]//rows
            elif tmpname=='tr':
                min_x, min_y, max_x, max_y = img.shape[1]*(cols-1)//cols, 0 , img.shape[1],img.shape[0]//rows
            elif tmpname=='bl':
                min_x, min_y, max_x, max_y = 0, img.shape[0]*(rows-1)//rows ,img.shape[1]//cols ,img.shape[0]
            elif tmpname=='br':
                min_x, min_y, max_x, max_y = img.shape[1]*(cols-1)//cols, img.shape[0]*(rows-1)//rows ,img.shape[1] ,img.shape[0]
            else:
                assert False, "bad "
            

            res = cv2.matchTemplate(img[min_y:max_y, min_x:max_x, :],templimg,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            top_left = (top_left[0] + min_x, top_left[1] + min_y)
            bottom_right = (top_left[0] + w, top_left[1] + h)
            toc = time.time()
            #print('timing: {} seconds'.format(toc-tic))
            corners[tmpname] = (top_left,bottom_right)

        if self._is_good_corners(corners,img.shape[1], img.shape[0]):
            return True, corners

        return False, corners
        
    def _get_corner(self, img_data):
        for templ in self._tempates2mathch():
            ok,corners = self._get_corners_of_templ(img_data, templ)
            if ok:
                logging.warning('get corners by template {}'.format(templ))
                self.last_templ = templ
                return corners
            
        return None


    def _show_img(self, title, img):
        plt.title(title)
        plt.imshow(img,cmap= "gray")
        plt.show()
        #time.sleep(1)

    def get_precut_corner_by_template(self, img_data):
        img_data = cv2.copyMakeBorder(img_data,50,50,50,50,cv2.BORDER_CONSTANT,value = [0,0,0])

        img = cv2.resize(img_data, (int(img_data.shape[1]/self._scale), int(img_data.shape[0]/self._scale))) 
        corners = self._get_corner(img)
        if corners is None:
            logging.info('no corner found')
            return None
        
        tl = corners['tl']
        tl = [(tl[0][0]+tl[1][0])*self._scale//2, (tl[0][1]+tl[1][1])*self._scale//2]
        tr = corners['tr']
        tr = [(tr[0][0]+tr[1][0])*self._scale//2, (tr[0][1]+tr[1][1])*self._scale//2]
        bl = corners['bl']
        bl = [(bl[0][0]+bl[1][0])*self._scale//2, (bl[0][1]+bl[1][1])*self._scale//2]
        br = corners['br']
        br = [(br[0][0]+br[1][0])*self._scale//2, (br[0][1]+br[1][1])*self._scale//2]

        return (tl, tr, bl, br)


    def simple_cut(self, img_data):
        img = img_data[:,:,0]
        h,w = img.shape
        thres,img2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
        index = np.where(img2 == 255)
        ymin , xmin = np.min(index,axis = 1)
        ymax , xmax = np.max(index,axis = 1)
        r_sum = np.sum(img2,axis = 1)
        ymax_t = np.where(r_sum>.3*255*img.shape[1])[0]
        ymax_t = ymax_t[-1] if len(ymax_t)>0 else ymax
        newimg = img_data[ymin:ymax_t,xmin:xmax,:]
        return newimg, (xmin, ymin)

    def cut_by_grid(self, img_data, rows, cols):
        # 对比较正的层前图，利用栅线信息进行边缘切割
        col_lines, row_lines = get_cut_line_none_precut(img_data, rows, cols)
        #print(np.array(col_lines).var(), np.array(row_lines).var())
        
        col_step = (col_lines[-2] - col_lines[1])//(cols-2)
        row_step = (row_lines[-2] - row_lines[1])//(rows-2)
        xmin, xmax = col_lines[1]-col_step, col_lines[-2]+col_step
        ymin, ymax = row_lines[1]-row_step, row_lines[-2]+row_step
        tl, tr, bl, br = (xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)
        return tl, tr, bl, br

    def get_precut(self, img_data, rows, cols):
        '''
        '''
        offset = 10
        hh, ww = img_data.shape[:2]

        if rows > 4 and cols > 4:
            # 对比较正的层前图，利用栅线信息进行边缘切割
            try:
                tl, tr, bl, br = self.cut_by_grid(img_data, rows, cols)
                min_x = max(min(tl[0], bl[0]) -offset, 0)
                max_x = min(max(tr[0], br[0]) +offset, ww-1)
                min_y = max(min(tl[1], tr[1]) -offset, 0)
                max_y = min(max(bl[1], br[1]) +offset, hh-1)
                return img_data[min_y:max_y, min_x:max_x, :], (min_x, min_y)
            except:
                logging.exception('cut_by_grid failed')

        try:
            return self.simple_cut(img_data)
        except:
                logging.exception('simple_cut')
        
        corners = None
        try:
            corners = self.get_precut_corner_by_contour(img_data)  
        except:
            logging.exception('get_precut_corner_by_contour')         
        if corners == None:
            try:
                corners = self.get_precut_corner_by_template(img_data)
            except:
                logging.exception('get_precut_corner_by_template')



        if corners == None:
            logging.info('failed get good cornner by template, return origin image')
            return img_data, (0,0)

        logging.info('success get good cornner.')
        
        tl, tr, bl, br = corners

        hh, ww = img_data.shape[:2]
        min_x = max(min(tl[0], bl[0]) -offset, 0)
        max_x = min(max(tr[0], br[0]) +offset, ww-1)
        min_y = max(min(tl[1], tr[1]) -offset, 0)
        max_y = min(max(bl[1], br[1]) +offset, hh-1)

        return img_data[min_y:max_y, min_x:max_x, :], (min_x, min_y)

    def getLinePara(self, line):
        '''简化交点计算公式'''
        a = line[0][1] - line[1][1]
        b = line[1][0] - line[0][0]
        c = line[0][0] *line[1][1] - line[1][0] * line[0][1]
        return a,b,c

    def getCrossPoint(self, line1,line2):
        '''计算交点坐标，此函数求的是line1中被line2所切割而得到的点，不含端点'''
        a1,b1,c1 = self.getLinePara(line1)
        a2,b2,c2 = self.getLinePara(line2)
        d = a1* b2 - a2 * b1
        p = [0,0]
        if d == 0:#d为0即line1和line2平行
            return ()
        else:
            p[0] = int(round((b1 * c2 - b2 * c1)*1.0 / d,2))#工作中需要处理有效位数，实际可以去掉round()
            p[1] = int(round((c1 * a2 - c2 * a1)*1.0 / d,2))
        p = tuple(p)
        return p

    def _fix_line(self,img, offset):
        _, cts, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        cts = sorted(cts, key=cv2.contourArea, reverse=True)
        ct = cts[0]
        [vx,vy,x,y] = cv2.fitLine(ct, cv2.DIST_L2,0,0.01,0.01)
        x,y = x+ offset[0] , y+offset[1]
        line = [(np.asscalar(x),np.asscalar(y)), (np.asscalar(x+vx*100), np.asscalar(y+vy*100))]
        return line

    def get_precut_corner_by_contour(self, image_data):
        if len(image_data.shape)>2:
            img_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # 给图片加一圈黑色，有利于最大类间方差法
        pad = 100
        img_data = np.pad(img_data, pad, mode='constant')
        img_data = cv2.resize(img_data, (int(img_data.shape[1]/self._scale), int(img_data.shape[0]/self._scale))) 

        h,w=img_data.shape[:2]

        # 去除太亮的像素，因为太亮的像素很可能是电池片外的白斑或左下角的序列化
        #avg = np.average(img_data[h//4:h*3//4, w//4:w*3//4])
        #img_data[img_data>avg*2]=0

        #最大类间方差法做阈值
        threshold,imgOtsu = cv2.threshold(img_data,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imgOtsu_Org = imgOtsu

        kernel = np.ones((3,3),np.uint8)
        _, contours, hierarchy = cv2.findContours(imgOtsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0]
        
        # 对于平滑简单的图像，进行一次腐蚀
        if len(contours)<600:
            logging.info("make a erode as len(contours)<600")
            imgOtsu = cv2.erode(imgOtsu,kernel,iterations = 4)
            _, contours, hierarchy = cv2.findContours(imgOtsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour = contours[0]

        # 边缘太复杂，一般出现在电池片较为暗淡的情况，将边缘线的区域填充
        if contour.size>5000:
            logging.info("contour.size>5000: add contour")
            contourimg = cv2.drawContours(imgOtsu.copy()*0,[contour],0,(10,10,10),10)   
            imgOtsu = imgOtsu + contourimg
            imgOtsu[imgOtsu>0]=255
            _, contours, hierarchy = cv2.findContours(imgOtsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour = contours[0]             
            
        contour_Area = cv2.contourArea(contour)
        contourRatio = contour_Area/(h*w)
        contourHull_Area = cv2.contourArea(cv2.convexHull(contour,returnPoints=True))
        rect = cv2.minAreaRect(contour)
        contourMinRect_Area = cv2.contourArea(np.int0(cv2.boxPoints(rect)))
        # 如果情况不佳，尝试进行一次图像膨胀
        if contourRatio<0.6 or contourMinRect_Area/contourHull_Area > 1.1 and contourHull_Area/contour_Area>1.1 :
            logging.info("make a dilate")
            imgOtsu = cv2.dilate(imgOtsu,kernel,iterations = 4)
            _, contours, hierarchy = cv2.findContours(imgOtsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour = contours[0]
            
            contour_Area = cv2.contourArea(contour)
            contourRatio = contour_Area/(h*w)
            contourHull_Area = cv2.contourArea(cv2.convexHull(contour,returnPoints=True))
            rect = cv2.minAreaRect(contour)
            contourMinRect_Area = cv2.contourArea(np.int0(cv2.boxPoints(rect)))

            # 如果情况任不佳，尝试对原图像进行一次图像腐蚀
            if contourRatio<0.6 or contourMinRect_Area/contourHull_Area > 1.1 and contourHull_Area/contour_Area>1.1 :
                logging.info("make a erode")
                imgOtsu = cv2.erode(imgOtsu_Org,kernel,iterations = 4)
                _, contours, hierarchy = cv2.findContours(imgOtsu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                contour = contours[0]

        # 求追踪的边缘的凸边缘，并生成仅仅包含凸边缘的图像，用来计算边缘直线
        contour = cv2.convexHull(contour,returnPoints=True) 
        contourimg = cv2.drawContours(imgOtsu.copy()*0,[contour],0,(255,255,255),5)
        
        # 凸边缘的范围
        pt_x = contour[:,:,0]
        pt_y = contour[:,:,1]
        min_x, max_x = np.min(pt_x), np.max(pt_x)
        min_y, max_y = np.min(pt_y), np.max(pt_y)        
        
        # ratiio 用来调节用那部分的边缘像素值计算直线，数值越大，
        ratiio = 5
        dh, dw = (max_y-min_y)//ratiio, (max_x-min_x)//ratiio
        # padding 用来向外调节边框值，通过调节 padding值可以扩大最终的边框范围
        pading = 2
        # 计算上下左右四条线
        up_img_org = contourimg[min_y : min_y+dh,min_x +dw:max_x -dw]
        up_line = self._fix_line(up_img_org,(min_x +dw, min_y-pading))

        down_img_org = contourimg[max_y-dh:max_y, min_x +dw:max_x -dw]
        down_line = self._fix_line(down_img_org,(min_x +dw, max_y-dh +pading))

        left_img_org = contourimg[min_y +dh:max_y -dh,min_x :min_x +dw]
        left_line = self._fix_line(left_img_org,(min_x-pading, min_y +dh))

        right_img_org = contourimg[min_y +dh:max_y -dh,max_x -dw :max_x]
        right_line = self._fix_line(right_img_org,(max_x -dw+pading, min_y +dh))

        # 计算四个角点的位置
        tl=self.getCrossPoint(up_line, left_line)
        tr= self.getCrossPoint(up_line, right_line)
        bl= self.getCrossPoint(down_line, left_line)
        br= self.getCrossPoint(down_line, right_line)

        tl=[x*self._scale - pad for x in tl]
        tr=[x*self._scale - pad  for x in tr]
        bl=[x*self._scale - pad  for x in bl]
        br=[x*self._scale - pad  for x in br]

        if not self._is_good_corners({'tl':[tl,],'tr':[tr,],'bl':[bl,],'br':[br,] }, w*self._scale, h*self._scale):
            return None

        return (tl,tr, bl, br)

def get_cut_line(img, rows, cols, edge_removed):
    ybias, xbias = 0, 0
    if not edge_removed:
        precutter = ElPreCuter()
        img, bias = precutter.get_precut(img, rows, cols)
        xbias, ybias = bias

    col_lines, row_lines = get_cut_line_none_precut(img, rows, cols)

    if xbias>0:
        col_lines = [x+xbias for x in col_lines]
    if ybias>0:
        row_lines = [x+ybias for x in row_lines]


    return col_lines, row_lines

def get_cut_line_none_precut(img, rows, cols):
    img_gray = img[:,:,0]
    hh, ww = img.shape[:2]
    xstep = ww//cols
    ystep = hh//rows
    horizontal = np.sum(img_gray, axis = 0)

    vertical = np.sum(img_gray, axis = 1)
    # 电池片水平栅格线有可能干扰到电池片的水平切割，特别处理下
    conv_width_half = hh//(rows*90)
    vertical = np.convolve(vertical, np.ones(conv_width_half*2 +1).astype('int'))[conv_width_half : -1*conv_width_half]

    col_lines = [0]
    xstart = xstep*3//4
    for x in range(cols-1):
        xstart = np.argmin(horizontal[xstart: xstart + xstep//2]) + xstart
        col_lines.append(xstart)
        xstart += xstep*3//4
    col_lines.append(ww-1)

    row_lines = [0]
    ystart = ystep*3//4
    for y in range(rows-1):
        ystart = np.argmin(vertical[ystart: ystart + ystep//2]) + ystart
        row_lines.append(ystart)
        ystart += ystep*3//4
    row_lines.append(hh-1)

    # verify
    assert abs(col_lines[-1]-col_lines[-2]-xstep)/xstep < 0.4
    assert abs(row_lines[-1]-row_lines[-2]-ystep)/ystep < 0.4

    return col_lines, row_lines

def grid_cut(img_data, rows, cols, edge_removed):
    """
    按晶格切割大图，返回切割后的图像及其位置，注意返回的位置为晶格图像在大图片中的位置
    
    :param img_data: 图像数据
    :param rows: 电池片的行数
    :param cols: 电池片列数
    :edge_removed: 默认False，是否无黑边
    :returns: 切割后的图像及其位置，注意返回的位置为晶格图像在大图片中的位置
    """
    try:
        col_lines, row_lines = get_cut_line(img_data, rows, cols, edge_removed)
    except:   
        logging.exception('el cut failed, cut it by average.') 
        stride_row, stride_col = int(img_data.shape[0]/rows), int(img_data.shape[1]/cols)
        row_lines=[x*stride_row for x in range(rows)]
        row_lines.append(img_data.shape[0])
        col_lines=[x*stride_col for x in range(cols)]
        col_lines.append(img_data.shape[1])

    cut_images = {}
    for row in range(len(row_lines)-1):
        for col in range(len(col_lines)-1):
            key = (row_lines[row], col_lines[col])
            img = img_data[row_lines[row]:row_lines[row+1], col_lines[col]:col_lines[col+1], :]
            cut_images[key] = img

    logging.info('image cut: row_lines={}, col_lines={}'.format(str(row_lines), str(col_lines)))

    return cut_images, (row_lines, col_lines)
