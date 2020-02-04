"""DictData 框架实现，提供 DictData 基础类，实现 to_dict 和 from_dict，达到对象和dict的相互装换。
"""
import json

# 所有DictData 类型系列
DICTDATA_CLASSES = {}


class DictDataField:
    """DictData的数据成员定义"""

    def __init__(self, cls, default):
        self.name = ''  # 属性名称
        self.cls = cls  # 属性数据类型
        self.default = default  # 属性默认值


class DictDataMetaclass(type):
    """DictData 元类.
    记录类型的成员信息，包括类型和默认值，将其放到类型属性 __dict_data_attrs__ 当中，
    再配合 DictData，实现DictData及其子类和字典之间的转换，并强制暴露对象属性定义属性定义
    """

    def __new__(cls, name, bases, attrs):
        """为TaskBase添加 data_cls 和 result_cls 属性, 实现子类继承父类的属性定义."""
        dictdata_fields = []
        new_attrs = {}
        for key, val in attrs.items():
            if isinstance(val, DictDataField):
                val.name = key
                dictdata_fields.append(val)
            else:
                new_attrs[key] = val
        new_attrs['__dictdata_fields__'] = dictdata_fields  # dictdata_fields 属性定义
        new_attrs['__slots__'] = [x.name for x in dictdata_fields]  # 固定对象属性

        # 添加默认初始化函数
        def init_func(self, *args, **kwargs):
            DictData.__init__(self, *args, **kwargs)

        if '__init__' not in new_attrs:
            new_attrs['__init__'] = init_func
        cls = super().__new__(cls, name, bases, new_attrs)
        # 注册到json_coders
        DICTDATA_CLASSES[name] = cls

        return cls


class DictData(metaclass=DictDataMetaclass):
    """DictData基础类.
    实现to_dict 和 from_dict
    注： 所有DictData子类，如果自定义了__init__, 需要首先调用 super().__init__
    """

    def __init__(self, *args, **kwargs):
        for dictdata_field in self.dictdata_fields():
            setattr(self, dictdata_field.name,
                    kwargs[dictdata_field.name] if dictdata_field.name in kwargs else dictdata_field.default)

    @classmethod
    def dictdata_fields(cls):
        """返回所有数据定义对象。"""
        rv = []
        base = cls
        while hasattr(base, '__dictdata_fields__'):
            for field in base.__dictdata_fields__:
                rv.append(field)
            base = base.__base__
        return rv

    def to_dict(self):
        """转换成字典."""
        rv = {}
        for dictdata_field in self.dictdata_fields():
            val = getattr(self, dictdata_field.name)
            if isinstance(val, DictData):
                rv[dictdata_field.name] = val.to_dict()
            else:
                rv[dictdata_field.name] = val
        return rv

    def to_json(self):
        """转换到json字符串."""
        import multi_threading.json_coders as json_coders
        return json.dumps(self, cls=json_coders.HPJsonEncoder)

    @classmethod
    def from_dict(cls, dict_val: dict):
        """从字典生成对象."""
        if dict_val is None:
            return None
        result = cls()
        for dictdata_field in cls.dictdata_fields():
            if dictdata_field.name not in dict_val:
                continue
            if issubclass(dictdata_field.cls, DictData):
                setattr(result, dictdata_field.name, dictdata_field.cls.from_dict(dict_val[dictdata_field.name]))
            else:
                setattr(result, dictdata_field.name, dict_val[dictdata_field.name])
        return result

    @classmethod
    def from_json(cls, json_str: str):
        """从json字符串生成对象. 需配合HPJsonDecoder."""
        import multi_threading.json_coders as json_coders
        return json.loads(json_str, cls=json_coders.HPJsonDecoder)

    def __str__(self):
        """用于logging"""
        return self.__class__.__name__ + self.to_json()
