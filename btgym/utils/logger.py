import logging
import os
import time

from btgym.utils.path import ROOT_PATH

log_path = os.path.join(ROOT_PATH, '../outputs/logs')

logger = None

def log(text):
    global logger
    if logger is None:
        set_logger_entry('default')
    logger.info(time.strftime('%H:%M:%S', time.localtime()) + ' ' + str(text))

def set_logger_entry(file_path):
    global logger

    file_name = os.path.basename(file_path).split('.')[0]
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    # 创建文件处理程序，将日志写入文件
    os.makedirs(f'{log_path}', exist_ok=True)
    file_handler = logging.FileHandler(f'{log_path}/{file_name}.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建日志格式器并将其添加到处理程序
    formatter = logging.Formatter('%(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理程序添加到日志记录器
    logger.addHandler(file_handler)

    logger.info('\n\n====== {} new run ======'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))


