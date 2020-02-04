"""提供json decoder和encode，支持numpy类型和dictata类型"""
import json
from datetime import datetime
import base64
import uuid

import numpy

from multi_threading.dict_data import DictData, DICTDATA_CLASSES


class HPJsonEncoder(json.JSONEncoder):
    """记录所有已经加载的DictData."""

    def default(self, obj):
        """Override"""
        if isinstance(obj, bytes):
            return {'__type__': 'bytes', 'value': str(base64.b64encode(obj), encoding='utf-8')}

        if isinstance(obj, uuid.UUID):
            return {'__type__': 'UUID', 'value': str(obj)}

        if isinstance(obj, DictData):
            return {'__type__': obj.__class__.__name__, 'value': obj.to_dict()}

        if isinstance(obj, numpy.ndarray):
            return obj.tolist()

        if isinstance(obj, datetime):
            return {'__type__': 'datetime', 'value': obj.timestamp()}

        if hasattr(numpy, 'integer') and isinstance(obj, numpy.integer):
            return int(obj)

        if hasattr(numpy, 'float') and isinstance(obj, numpy.float):
            return float(obj)

        if hasattr(numpy, 'float32') and isinstance(obj, numpy.float32):
            return float(obj)

        if hasattr(numpy, 'float64') and isinstance(obj, numpy.float64):
            return float(obj)

        return super().default(obj)


class HPJsonDecoder(json.JSONDecoder):
    """定制json decoder, 支持自有类型。"""

    def __init__(self, *args, **kargs):
        """构造器"""
        super().__init__(object_hook=self.dict_to_object,
                         *args, **kargs)

    def dict_to_object(self, obj):
        """对象构造回调函数"""
        if '__type__' not in obj:
            return obj
        type_ = obj['__type__']
        if type_ == 'bytes':
            return base64.b64decode(bytes(obj['value'], encoding='utf-8'))
        elif type_ == 'UUID':
            return uuid.UUID(obj['value'])
        elif type_ == 'datetime':
            return datetime.fromtimestamp(obj['value'])
        elif type_ in DICTDATA_CLASSES:
            cls = DICTDATA_CLASSES[type_]
            obj = cls.from_dict(obj['value'])
            return obj
        else:
            raise TypeError('不支持加载类型 %s。' % type_)

        return obj
