import logging
import os
import sys
from pathlib import Path

def setup_logger(log_file="app.log", level=logging.INFO):
    """
    设置日志配置，解决乱码问题
    
    参数:
    log_file (str): 日志文件路径
    level (int): 日志级别
    
    返回:
    logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器，使用utf-8编码
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name=None):
    """
    获取一个命名的日志记录器
    
    参数:
    name (str): 日志记录器名称
    
    返回:
    logging.Logger: 命名的日志记录器
    """
    return logging.getLogger(name)
