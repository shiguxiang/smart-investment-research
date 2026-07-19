"""
日志配置模块
基于 loguru，支持控制台 + 文件双输出
"""

import sys
from pathlib import Path
from loguru import logger

from config.settings import settings

# 移除默认 handler
logger.remove()

# 控制台输出 (彩色)
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level=settings.log_level,
    colorize=True,
)

# 文件输出
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
)

# 错误日志单独输出
logger.add(
    LOG_DIR / "error_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="ERROR",
    rotation="10 MB",
    retention="90 days",
    encoding="utf-8",
)


def get_logger(name: str):
    """获取命名 logger"""
    return logger.bind(name=name)
