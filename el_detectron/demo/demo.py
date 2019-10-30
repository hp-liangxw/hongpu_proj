import os
import yaml
import shutil
import cv2
from deep_learning_evaluation.src.utils import get_logger

logger = get_logger()


class Config:
    def __init__(self, config_path):
        self.config_path = config_path

        self.yml_str = None

        self.types_and_ratios = None
        self.model_paths = None

        self.INFO = None
        self.PRE_PROCESS = None
        self.IMG_CUT_FOLDER = None
        self.OUTPUT_FOLDER = None
        self.MODEL = None
        self.AUTO_LABELING = None
        self.POST_PROCESS = None
        self.EVALUATION = None

        self.is_valid = self.check_if_valid()

    def check_if_valid(self):
        """
        确认配置文件的正确性
        :param yml_str:
        :return:
        """
        if not os.path.exists(self.config_path):
            logger.info("the configuration file: {} is not exists!".format(self.config_path))
            return 1

        # 读取配置文件
        self.yml_str = open(self.config_path).read()
        try:
            cfg_info = yaml.load(self.yml_str)

            self.INFO = cfg_info['info']

            self.PRE_PROCESS = cfg_info['pre_process']
            self.POST_PROCESS = cfg_info['post_process']

            self.IMG_CUT_FOLDER = cfg_info['img_folder']
            self.OUTPUT_FOLDER = cfg_info['output_folder']

            self.MODEL = cfg_info['model']

            self.AUTO_LABELING = cfg_info['auto_labeling']
            self.EVALUATION = cfg_info['evaluation']
        except:
            logger.info("failed to read configuration file")
            return 1

        # 有前处理时：
        if self.PRE_PROCESS['switch'] is True:
            # 判断前处理文件夹是否存在
            if not os.path.exists(self.PRE_PROCESS['origin_folder']):
                logger.info("path: {} not exists!".format(self.PRE_PROCESS['origin_folder']))
                return 1

        # 判断图片输入路径是否存在
        if not os.path.exists(self.IMG_CUT_FOLDER):
            logger.info("path: {} not exists!".format(self.IMG_CUT_FOLDER))
            return 1

        # 判断模型路径是否存在
        self.types_and_ratios, self.model_paths = self.get_model_info()
        for i in self.model_paths.values():
            if not os.path.exists(i):
                logger.info("path: {} not exists!".format(i))
                return 1

        # 需要评估时：
        if self.EVALUATION['switch'] is True:
            # 判断答案文件是否存在
            if not os.path.exists(self.EVALUATION['csv_path']):
                logger.info("path: {} not exists!".format(self.EVALUATION['csv_path']))
                return 1
            # 需要筛选置信度时：
            if self.EVALUATION['select_th']['switch'] is True:
                select_th_keys = self.EVALUATION['select_th'].keys()
                # 判断参数是否正确
                if 'step' not in select_th_keys or \
                        'miss_number' not in select_th_keys or \
                        'overkill_number' not in select_th_keys:
                    logger.info("paramaters: {} not exists!".format("step, miss_number, overkill_number"))
                    return 1

        # 创建输出文件夹
        if os.path.exists(self.OUTPUT_FOLDER):
            shutil.rmtree(self.OUTPUT_FOLDER)
        os.mkdir(self.OUTPUT_FOLDER)
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'csv'))
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'img_with_box'))
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'xml'))
        os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'pics'))

        # 再次查看配置信息正确性
        logger.info("\n\n>>>>>>>>----------------------------------------------------------------<<<<<<<<\n")
        logger.info(self.yml_str)
        logger.info("\n>>>>>>>>----------------------------------------------------------------<<<<<<<<\n\n")

        answer = input("Is the configuration file correct? [yes/no]:")
        while answer.upper() not in ["YES", "Y", "NO", "N"]:
            logger.info("Please input yes or no!")
            answer = input()

        if answer.upper() in ["YES", "Y"]:
            return 0
        else:
            return 1

    def get_model_info(self):
        """
        获取缺陷类型、缺陷置信度、模型路径
        model_infos:{
            "yinlie": {
                "model_dir" : /root/el/yinlie.model",
                "yinlie": 0.9,
                "shixiao": 0.9,
            },
            "xuhan": {
                "model_dir" : /root/el/xuhan.model",
                "shixiao": 0.9,
            }
        }
        :return:
        """
        # 缺陷类型与置信度
        model_infos = {}
        for model_name, this_info in self.MODEL.items():
            # 模型路径
            model_infos[model_name] = {
                "model_dir": this_info["model_dir"],  # 模型路径
                "defect_name": [],  # 该模型中的缺陷名称
                "threshold": {}  # 置信度
            }
            for defect_name, defect_threshold in this_info["threshold"].items():
                # 缺陷类型与置信度
                model_infos[model_name]["defect_name"].append(defect_name)
                model_infos[model_name]["threshold"][defect_name] = defect_threshold
        return model_infos


config_path = r"config.yml"
cfgs = Config(config_path)

model_infos = cfgs.get_model_info()
print(model_infos)
