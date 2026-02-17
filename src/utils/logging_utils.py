"""
日志工具模块

提供统一的日志配置和管理功能。
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    设置并获取logger实例
    
    Args:
        name: logger名称
        level: 日志级别 ["DEBUG", "INFO", "WARNING", "ERROR"]
        log_dir: 日志文件保存目录
        log_to_file: 是否保存到文件
        log_to_console: 是否输出到控制台
        
    Returns:
        logging.Logger: 配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有handler
    logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件输出
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 控制台输出
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
