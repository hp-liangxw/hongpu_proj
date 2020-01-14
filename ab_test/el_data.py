""" 定义平台需要的数据结构"""

from typing import Dict, List
import base64
import logging
import uuid
import pickle
import math

import numpy as np

from .dict_data import DictData, DictDataField
from ab_test.hp_utils import img_bytes_2_cv


class Defect(DictData):
    ''' 缺陷信息
    '''
    name = DictDataField(str, '')  # 缺陷名称
    confidence = DictDataField(float, 0.0)  # 置信度
    row = DictDataField(int, 6)   # 缺陷所在行，从 1 开始
    col = DictDataField(int, 12)   # 缺陷所在列，从 1 开始
    coordinate = DictDataField(list, [])  # 缺陷位置 （x1, y1, x2, y2）
    ext_features = DictDataField(dict, {})  # 扩展缺陷特征值


class Recipe(DictData):
    '''代表一个配方'''
    name = DictDataField(str, '')  # 配方名称
    defect_keepers = DictDataField(list, [])          # 缺陷过滤器
    defect_filters = DictDataField(list, [])          # 缺陷过滤器
    cell_filters = DictDataField(list, [])         # 电池片过滤器
    comp_filters = DictDataField(list, [])          # 组件过滤器

    features_str = ['class']
    features_number = ['prob', 'area_ratio', 'h_w_ratio']

    def execute(self, img_info, defects, features):
        '''执行配方系统'''
        result_defects = []
        for defect in defects:
            fetured_defect = self.defect_feturing(img_info, defect, features)
            if self.persist_defect(fetured_defect) or not self.filter_defect(fetured_defect):
                result_defects.append(defect)
            result_defects = self.filter_cell(result_defects)
            result_defects = self.filter_component(result_defects)

        logging.info('执行配方之后的缺陷列表: %s', result_defects)
        return result_defects

    def _match_conditions(self, defect, conditions):
        if 'locations' in conditions and defect['loc'] not in conditions['locations']:
            return False

        for cond in Recipe.features_str:
            if cond in conditions and defect[cond] != conditions[cond]:
                return False

        for cond in Recipe.features_number:
            if cond in conditions and not (conditions[cond][0] < defect[cond] < conditions[cond][1]):
                return False

        return True

    def persist_defect(self, defect):
        ''' 缺陷级别保留
        '''
        for conditions in self.defect_keepers:
            if self._match_conditions(defect, conditions):
                return True

        return False

    def filter_defect(self, defect):
        ''' 缺陷级别过滤
        '''
        for conditions in self.defect_filters:
            if self._match_conditions(defect, conditions):
                return True

        return False

    def filter_cell(self, defects):
        ''' 电池片级别的过滤
        '''
        defects_by_cell = {}
        for item in defects:
            defects_by_cell.setdefault(tuple(item['loc']), []).append(item)

        allow_defects = self.cell_filters[0]['defects'][1] if self.cell_filters else 0

        rv = []
        for _, defects in defects_by_cell.items():
            if len(defects) > allow_defects:  # 缺陷数量小于阈值是，过滤掉该电池片上
                rv.extend(defects)
        return rv

    def filter_component(self, defects):
        ''' 缺陷级别过滤
        '''
        allow_cells = self.comp_filters[0]['cells'][1] if self.comp_filters else 0
        cells = {tuple(defect['loc']) for defect in defects}  # 有缺陷的电池片位置集合
        if len(cells) > allow_cells:
            return defects

        return []

    def _get_defect_info(self, img_info, coord, grid_coord, features):
        defect_info_dict = {}
        defect_info_dict['grid_h'] = grid_coord[3] - grid_coord[1]
        defect_info_dict['grid_w'] = grid_coord[2] - grid_coord[0]
        defect_info_dict['grid_area'] = defect_info_dict['grid_h'] * defect_info_dict['grid_w']  # 面积
        defect_info_dict['grid_diag'] = np.sqrt(defect_info_dict['grid_h']**2 + defect_info_dict['grid_w']**2)  # 对角线长
        defect_info_dict['grid_coord'] = grid_coord

        grid_x1, grid_y1, grid_x2, grid_y2 = coord

        # 还原大图坐标
        defect_info_dict['defect_h'] = grid_y2 - grid_y1  # 高、宽及比例
        defect_info_dict['h_ratio'] = defect_info_dict['defect_h'] / defect_info_dict['grid_h']

        defect_info_dict['defect_w'] = grid_x2 - grid_x1
        defect_info_dict['w_ratio'] = defect_info_dict['defect_w'] / defect_info_dict['grid_w']

        defect_info_dict['defect_area'] = defect_info_dict['defect_h'] * defect_info_dict['defect_w']  # 面积及比例
        defect_info_dict['area_ratio'] = defect_info_dict['defect_area'] / defect_info_dict['grid_area']

        defect_info_dict['defect_diag'] = math.sqrt(defect_info_dict['defect_h']**2 + defect_info_dict['defect_w']**2)  # 对角线长及比例
        defect_info_dict['diag_ratio'] = defect_info_dict['defect_diag'] / defect_info_dict['grid_diag']

        defect_info_dict['angle'] = math.atan2(defect_info_dict['defect_h'], defect_info_dict['defect_w']) * 180.0 / np.pi  # 角度

        defect_info_dict['h_w_ratio'] = defect_info_dict['defect_h'] / defect_info_dict['defect_w']  # 高宽比

        # 距离四顶点距离
        defect_info_dict['corner_dist_lt'] = math.sqrt((coord[0]-grid_coord[0])**2 + (coord[1]-grid_coord[1])**2)  # 左上
        defect_info_dict['corner_dist_lb'] = math.sqrt((coord[0]-grid_coord[0])**2 + (coord[3]-grid_coord[3])**2)  # 左下
        defect_info_dict['corner_dist_rt'] = math.sqrt((coord[2]-grid_coord[2])**2 + (coord[1]-grid_coord[1])**2)  # 右上
        defect_info_dict['corner_dist_rb'] = math.sqrt((coord[2]-grid_coord[2])**2 + (coord[3]-grid_coord[3])**2)  # 右下

        defect_info_dict['corner_dist_ratio_lt'] = defect_info_dict['corner_dist_lt'] / defect_info_dict['grid_diag']  # 左上
        defect_info_dict['corner_dist_ratio_lb'] = defect_info_dict['corner_dist_lb'] / defect_info_dict['grid_diag']  # 左下
        defect_info_dict['corner_dist_ratio_rt'] = defect_info_dict['corner_dist_rt'] / defect_info_dict['grid_diag']  # 右上
        defect_info_dict['corner_dist_ratio_rb'] = defect_info_dict['corner_dist_rb'] / defect_info_dict['grid_diag']  # 右下

        defect_info_dict['corner_dist_ratio_min'] = min(defect_info_dict['corner_dist_ratio_lt'], defect_info_dict['corner_dist_ratio_lb'],
                                                        defect_info_dict['corner_dist_ratio_rt'], defect_info_dict['corner_dist_ratio_rb'])

        # 距离四边的距离
        defect_info_dict['edge_dist_top'] = abs(coord[1]-grid_coord[1])  # 上边
        defect_info_dict['edge_dist_bottom'] = abs(coord[3]-grid_coord[3])  # 下边
        defect_info_dict['edge_dist_left'] = abs(coord[0]-grid_coord[0])  # 左边
        defect_info_dict['edge_dist_right'] = abs(coord[2]-grid_coord[2])  # 右边

        defect_info_dict['edge_dist_ratio_top'] = abs(coord[1]-grid_coord[1]) / defect_info_dict['grid_h']  # 上边
        defect_info_dict['edge_dist_ratio_bottom'] = abs(coord[3]-grid_coord[3]) / defect_info_dict['grid_h']  # 下边
        defect_info_dict['edge_dist_ratio_left'] = abs(coord[0]-grid_coord[0]) / defect_info_dict['grid_w']  # 左边
        defect_info_dict['edge_dist_ratio_right'] = abs(coord[2]-grid_coord[2]) / defect_info_dict['grid_w']  # 右边

        defect_info_dict['edge_dist_ratio_min_hor'] = min(defect_info_dict['edge_dist_ratio_left'], defect_info_dict['edge_dist_ratio_right'])  # 右边
        defect_info_dict['edge_dist_ratio_min_ver'] = min(defect_info_dict['edge_dist_ratio_top'], defect_info_dict['edge_dist_ratio_bottom'])  # 右边
        defect_info_dict['edge_dist_ratio_min'] = min(defect_info_dict['edge_dist_ratio_min_hor'], defect_info_dict['edge_dist_ratio_min_ver'])  # 右边

        if set(['grayscale_ratio', 'defect_grayscale', 'grayscale_ratio']).intersection(set(features)):
            img = img_bytes_2_cv(img_info.img_bytes)
            grid_img_gray = img[grid_coord[1]:grid_coord[3], grid_coord[0]:grid_coord[2], 0]
            img_gray_re = grid_img_gray.reshape((1, -1))
            grid_min = np.percentile(img_gray_re, 25)
            grid_max = np.percentile(img_gray_re, 75)
            defect_info_dict['grid_grayscale'] = img_gray_re[(img_gray_re >= grid_min) & (img_gray_re <= grid_max)].mean()  # 灰度值
            if set(['defect_grayscale', 'grayscale_ratio']).intersection(set(features)):
                detect_img_gray = img[coord[1]:coord[3], coord[0]:coord[2], 0]
                detect_gray_re = detect_img_gray.reshape((1, -1))
                detect_min = np.percentile(detect_gray_re, 25)
                detect_max = np.percentile(detect_gray_re, 75)
                defect_info_dict['defect_grayscale'] = detect_gray_re[(detect_gray_re >= detect_min) & (detect_gray_re <= detect_max)].mean()  # 灰度值及比例
                defect_info_dict['grayscale_ratio'] = defect_info_dict['defect_grayscale'] / defect_info_dict['grid_grayscale']

        return defect_info_dict

    def defect_feturing(self, img_info, defect, features):
        '''缺陷特征工程,计算缺陷特征.'''
        row_lines = img_info.row_lines
        col_lines = img_info.col_lines

        row, col = defect['loc']
        row, col = row-1, col-1
        coord = defect['coord']
        grid_coord = col_lines[col], row_lines[row], col_lines[col+1], row_lines[row+1]

        fetured_defect = self._get_defect_info(img_info, coord, grid_coord, features)
        fetured_defect.update(defect)

        return fetured_defect


class ModuleInfo(DictData):
    ''' 组件信息'''
    id = DictDataField(str, '')  # 组件序列号
    rows = DictDataField(int, 6)   # 组件电池片行数
    cols = DictDataField(int, 12)   # 组件电池片列数
    monocrystal = DictDataField(bool, False)   # 是否为单晶
    half_plate = DictDataField(bool, False)    # 是否为半片
    station_stage = DictDataField(str, '')     # 机台类型：层前，层后
    module_type = DictDataField(str, '')     # 组件类型：5BB，9BB，叠瓦
    shift = DictDataField(str, '')  # 班次： 2010.06.08-白班
    layout = DictDataField(str, '')  # 版面类型： 5BB, 9BB, 叠瓦
    ext_info = DictDataField(dict, {})  # 扩展信息


class ImageInfo(DictData):
    ''' 图像信息'''
    img_name = DictDataField(str, '')   # 图像名称
    img_class = DictDataField(str, 'el')     # 图像类别, 外观：wg; EL：el。
    _img_bytes = DictDataField(bytes, b'')    # 图像字节流
    rows = DictDataField(int, 6)   # 图片中电池片行数
    cols = DictDataField(int, 12)   # 图片中电池片列数
    row_lines = DictDataField(list, [])  # 行分割线位置
    col_lines = DictDataField(list, [])  # 列分割线位置
    section_idx = DictDataField(list, [0, 0])   # 分段图在组件图中位置:从(0,0)开始
    section_num = DictDataField(list, [3, 4])   # 组件图分段图数量:(行分段数，列分段数)
    edge_removed = DictDataField(bool, False)  # 图像黑边是否已经切除
    col_flipped = DictDataField(bool, False)  # 图像的的列位置是否倒序，及图像的最后被为第一列
    ext_info = DictDataField(dict, {})  # 扩展信息

    def cache_image_bytes(self):
        '''将图像数据转移到redis缓存。并将缓存的key暂存在self._img_bytes中。
        ImageInfo对象频繁的放到消息队列，会造成较大的系统负担，在放入消息队列去应该进行缓存。
        '''
        if isinstance(self._img_bytes, uuid.UUID):
            return
        if not self._img_bytes:
            return
        uid = uuid.uuid1()
        key = str(uid)
        from ab_test.redis_utils import REDIS_CACHE
        if REDIS_CACHE.exists(key) and len(REDIS_CACHE.get(key)) > 1000:
            logging.info('expand expire img: %s with dumped string: %s, and ttl: %s, img_bytes: %s', key, len(REDIS_CACHE.get(key)), REDIS_CACHE.ttl(key), len(self.img_info.img_bytes))
            REDIS_CACHE.expire(key, 120)
        else:
            if REDIS_CACHE.exists(key):
                logging.warning('exist cache: %s with bytes: %s', key, len(REDIS_CACHE.get(key)))
            logging.info('cache img: %s with bytes: %s', key, len(self._img_bytes))
            REDIS_CACHE.set(key, pickle.dumps(self._img_bytes), 120)
        self._img_bytes = uid

    def restore_img_bytes(self):
        '''从redis缓存中恢复图像数据。'''
        if isinstance(self._img_bytes, uuid.UUID):
            from ab_test.redis_utils import REDIS_CACHE
            key = str(self._img_bytes)
            img_bytes = b''
            if REDIS_CACHE.exists(key):
                img_bytes = pickle.loads(REDIS_CACHE.get(key))
                self._img_bytes = img_bytes

    @property
    def img_bytes(self):
        '''img_bytes访问器'''
        if isinstance(self._img_bytes, uuid.UUID):
            from ab_test.redis_utils import REDIS_CACHE
            key = str(self._img_bytes)
            img_bytes = b''
            if REDIS_CACHE.exists(key):
                img_bytes = pickle.loads(REDIS_CACHE.get(key))
            return img_bytes
        return self._img_bytes

    @img_bytes.setter
    def img_bytes(self, value):
        self._img_bytes = value


class MesInfo(DictData):
    ''' 设备信息
    '''
    equip_id = DictDataField(str, '')   # 设备号
    product_line = DictDataField(str, '')  # 产线号
    facility_id = DictDataField(str, '')    # 车间号
    ext_info = DictDataField(dict, {})  # 扩展信息，


class ELData(DictData):
    '''EL 数据'''
    module_info = DictDataField(ModuleInfo, None)     # 组件信息
    img_info = DictDataField(ImageInfo, None)       # 图像信息
    mes_info = DictDataField(MesInfo, None)   # 设备信息

    @property
    def module_identity(self):
        '''构造组件标识'''
        return self.mes_info.facility_id + self.mes_info.product_line + \
            self.mes_info.equip_id + self.module_info.station_stage + self.module_info.id

    @property
    def image_identity(self):
        '''构造图像标识'''
        return self.module_identity + self.img_info.img_name + str(self.img_info.section_idx) + str(self.img_info.section_num)


class VIData(DictData):
    ''' VI 检测相关数据
    '''
    station_server = DictDataField(str, '')   # 机台的服务地址（用于接受vi检测结果信号）
    vi_server = DictDataField(str, '')      # 算法服务器地址 192.168.1.10:8000
    confirm_server = DictDataField(str, '')   # 复检服务器地址 192.168.1.10:8000
    confirm_mode = DictDataField(str, 'all')  # 复检模式: all:全复检; ng_only:坏件复检;
    # ok_only:只复检好品; none: 不复检
    platform = DictDataField(str, '')      # 中台地址 192.168.1.10:8000
    defects = DictDataField(list, [])          # 待检测缺陷名称列表
    features = DictDataField(list, [])  # 应获取缺陷特征列表
    recipe = DictDataField(Recipe, [])  # 缺陷配方
    ext_info = DictDataField(dict, {})   # 其它式列表


class VIResult(DictData):
    '''VI 检测相关数据
    '''
    error = DictDataField(int, 0)  # 错误信息，如果成功则为空字符串
    defects = DictDataField(list, [])  # 检测到的缺陷列表
    ext_info = DictDataField(dict, {})   # 其它式列表


class VIItem:
    '''定义一个检测项目的相关数据。'''

    def __init__(self, el_data, vi_data: VIData = None, vi_result: VIResult = None, confirmed_result: VIResult = None):
        self.el_data = el_data  # el检测数据源
        self.vi_data = vi_data  # vi 检测定义
        self.vi_result = vi_result  # vi 检测结果
        self._confirmed_result = confirmed_result  # 人工复检的结果

    @property
    def confirmed_result(self):
        '''confirmed_result访问器'''
        return self._confirmed_result

    @confirmed_result.setter
    def confirmed_result(self, val):
        self._confirmed_result = val

    @property
    def module_id(self):
        '''module_id 访问器'''
        return self.el_data.module_info.id

    @property
    def img_name(self):
        '''img_name 访问器'''
        return self.el_data.img_info.img_name

    @property
    def img_bytes(self):
        '''img_bytes 访问器'''
        return self.el_data.img_info.img_bytes

    def __eq__(self, other):
        '''是否为同一批检测图片'''
        if not other:
            return False
        return self.el_data.image_identity == other.el_data.image_identity

    def __hash__(self):
        '''实现通过image_identity标识VIItem'''
        return hash(self.el_data.image_identity)
