# coding: utf-8
'''一些工具函数'''
import os
import os.path as osp
import logging
import logging.handlers
import sys
import json
import numpy
import cv2
import yaml
import time
from collections import Iterable
import glob
import re
from threading import Lock


def open_file(fp, *args, **kargs):
    '''open file, 适配python 2 和 3'''
    if sys.version_info[0] >= 3:
        kargs['encoding'] = 'utf8'
        return open(fp, *args, **kargs)
    else:
        return open(fp, *args, **kargs)


class ConfigYaml:
    '''基于Yaml的配置类.'''

    def __init__(self, config_dir):
        self.config_dir = config_dir
        self._cache = None
        self.mutex = Lock()

    def __setitem__(self, key, value):
        '''实现访问器.'''
        self.set_config(key, value)

    def __getitem__(self, key):
        '''实现访问器.'''
        return self.get_config(key, None)

    def get(self, key, defval=None):
        '''实现访问器. 带默认值。'''
        return self.get_config(key, defval)

    def _update_config(self, from_, to_):
        if isinstance(from_, dict) and isinstance(to_, dict):
            for key, val_to in to_.items():
                if key not in from_:
                    continue
                val_from = from_[key]
                if isinstance(val_from, dict) and isinstance(val_to, dict):
                    self._update_config(val_from, val_to)
                else:
                    to_[key] = val_from

    @property
    def config_list(self):
        '''返回所有配置文件列表。'''
        rv = [osp.basename(file_path)[8:-4] for file_path in
              glob.glob(os.path.join(self.config_dir, 'config__*.yml'))]
        if self.local_config_path:
            rv.insert(0, '<本地>')
        return rv

    @property
    def current_config(self):
        '''当前配置文件。'''
        fp = osp.join(self.config_dir, 'active_config.txt')
        if osp.exists(fp):
            with open_file(fp) as fd:
                rv = fd.read()
                if rv in self.config_list:
                    return rv

        if self.config_list:
            return self.config_list[0]

        return None

    @current_config.setter
    def current_config(self, value):
        '''设置当前配置文件。'''
        if value == self.current_config:
            return

        assert value in self.config_list, '配置 %r 不存在。' % value
        with self.mutex:
            fp = osp.join(self.config_dir, 'active_config.txt')
            with open_file(fp, 'w+') as fd:
                fd.write(value)

            self._cache = None

    @property
    def config_path(self):
        '''当前配置文件的路径'''
        if self.current_config == '<本地>':
            return self.local_config_path

        return os.path.join(self.config_dir, 'config__{}.yml'.format(self.current_config))

    @property
    def local_config_path(self):
        '''返回本地配置文件路径名.'''
        locals_files = glob.glob(os.path.join(
            self.config_dir, 'config_local_*.yml'))
        assert len(locals_files) <= 1, 'there is more than one local config files'
        if locals_files:
            return locals_files[0]
        fp = os.path.join(self.config_dir, 'config_local.yml')
        return fp if osp.exists(fp) else None

    def get_config(self, key, defval):
        '''返回配置项的值，如果配置不存在，返回defval.'''
        if self._cache is None:
            with self.mutex:
                fp = os.path.abspath(os.path.join(
                    self.config_dir, 'config.yml'))
                if not os.path.exists(fp):
                    return defval

                with open_file(fp) as ff:
                    self._cache = yaml.load(ff) or {}

                fp = self.config_path
                if os.path.exists(fp):
                    with open_file(fp) as ff:
                        cfg_local = yaml.load(ff) or {}
                        self._update_config(cfg_local, self._cache)

        v = self._cache
        for k in key.split('.'):
            if not isinstance(v, Iterable):
                return defval
            if k not in v:
                return defval
            v = v[k]

        if isinstance(v, list):
            return v[:]
        if isinstance(v, dict):
            return v.copy()
        return v

    def flush(self):
        '''将配置写入磁盘文件。'''
        with self.mutex:
            fp = self.config_path
            with open_file(fp, 'w+') as ffw:
                yaml.dump(self._cache, ffw, allow_unicode=True,
                          indent=4, default_flow_style=False)

    def set_config(self, key, val, flush=False):
        '''设置配置'''
        with self.mutex:
            # update the cache
            if self._cache:
                v = self._cache
                for k in key.split('.')[:-1]:
                    if k not in v:
                        v[k] = {}
                    v = v[k]
                v[key.split('.')[-1]] = val

            if flush:
                fp = self.config_path
                with open_file(fp, 'w+') as ffw:
                    yaml.dump(self._cache, ffw, allow_unicode=True,
                              indent=4, default_flow_style=False)

# logging util


def initlogging(
        filepath,
        log_name=None,
        logerlevel=logging.DEBUG,
        consolelevel=logging.DEBUG,
        filelevel=logging.INFO,
        propagate=False):
    '''初始化日志，创建一个文件日志和一个控制台日志.'''
    thelogger = logging.getLogger(log_name)
    thelogger.setLevel(logerlevel)
    thelogger.propagate = propagate

    fmt = logging.Formatter(
        '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

    # 清除handler
    thelogger.handlers = []

    # 添加console handler
    cl = logging.StreamHandler()
    cl.setLevel(consolelevel)
    cl.setFormatter(fmt)
    thelogger.addHandler(cl)

    # 添加文件handler
    if filepath:
        if not osp.exists(osp.dirname(filepath)):
            os.makedirs(osp.dirname(filepath))

        fl = logging.handlers.RotatingFileHandler(
            filepath, mode='w', maxBytes=2 * 1024 * 1024, backupCount=100)
        fl.setLevel(filelevel)
        fl.setFormatter(fmt)
        thelogger.addHandler(fl)


def compose_path(fn, *p):
    '''路径组合'''
    fd = osp.dirname(fn)
    return osp.join(fd, *p)


def img_bytes_2_cv(img_bytes):
    '''将图像字节流装换为cv2数组(BGR格式)'''
    nparr = numpy.fromstring(img_bytes, numpy.uint8)
    if nparr.any():
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    else:
        return None


def img_cv_2_bytes(img_cv):
    '''将open cv数组(BGR格式)装换为图像字节流'''
    img_encode = cv2.imencode('.jpg', img_cv)[1]
    data_encode = numpy.array(img_encode)
    str_encode = data_encode.tostring()
    return str_encode


def timestamp_2_tring(sp):
    '''time stamp 格式化'''
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(sp))
