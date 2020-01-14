import sys
import logging


def get_logger():
    logger = logging.getLogger('test_and_evaluation')
    logger.setLevel(logging.INFO)

    rf_handler = logging.StreamHandler(sys.stderr)  # 默认是sys.stderr
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

    f_handler = logging.FileHandler('test.log')
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)

    return logger
