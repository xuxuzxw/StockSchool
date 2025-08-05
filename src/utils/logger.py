import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志工具
提供配置驱动的日志管理功能
"""


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # 颜色代码
    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",  # 重置
    }

    def format(self, record):
        """方法描述"""
        log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset_color = self.COLORS["RESET"]

        # 格式化消息
        formatted = super().format(record)

        # 只在终端输出时添加颜色
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            return f"{log_color}{formatted}{reset_color}"
        else:
            return formatted


class LoggerManager:
    """日志管理器"""

    _loggers = {}
    _default_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_enabled": True,
        "file_path": "logs/application.log",
        "max_file_size": "10MB",
        "backup_count": 5,
        "console_enabled": True,
        "colored_output": True,
    }

    @classmethod
    def setup_logger(cls, name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
        """
        设置日志器

        Args:
            name: 日志器名称
            config: 日志配置

        Returns:
            配置好的日志器
        """
        if name in cls._loggers:
            return cls._loggers[name]

        # 合并配置
        logger_config = cls._default_config.copy()
        if config:
            logger_config.update(config)

        # 创建日志器
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, logger_config["level"].upper()))

        # 清除现有处理器
        logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(logger_config["format"])
        colored_formatter = ColoredFormatter(logger_config["format"])

        # 控制台处理器
        if logger_config.get("console_enabled", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, logger_config["level"].upper()))

            if logger_config.get("colored_output", True):
                console_handler.setFormatter(colored_formatter)
            else:
                console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

        # 文件处理器
        if logger_config.get("file_enabled", True):
            file_path = Path(logger_config["file_path"])
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 解析文件大小
            max_bytes = cls._parse_file_size(logger_config.get("max_file_size", "10MB"))
            backup_count = logger_config.get("backup_count", 5)

            file_handler = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(getattr(logging, logger_config["level"].upper()))
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

        # 防止重复日志
        logger.propagate = False

        # 缓存日志器
        cls._loggers[name] = logger

        return logger

    @staticmethod
    def _parse_file_size(size_str: str) -> int:
        """解析文件大小字符串"""
        size_str = size_str.upper().strip()

        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            # 默认为字节
            return int(size_str)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取已配置的日志器"""
        if name in cls._loggers:
            return cls._loggers[name]
        else:
            return cls.setup_logger(name)

    @classmethod
    def update_logger_config(cls, name: str, config: Dict[str, Any]):
        """更新日志器配置"""
        if name in cls._loggers:
            # 移除现有日志器
            del cls._loggers[name]

        # 重新创建日志器
        cls.setup_logger(name, config)

    @classmethod
    def set_global_level(cls, level: str):
        """设置全局日志级别"""
        log_level = getattr(logging, level.upper())

        for logger in cls._loggers.values():
            logger.setLevel(log_level)
            for handler in logger.handlers:
                handler.setLevel(log_level)

    @classmethod
    def list_loggers(cls) -> list:
        """列出所有已配置的日志器"""
        return list(cls._loggers.keys())

    @classmethod
    def cleanup_loggers(cls):
        """清理所有日志器"""
        for logger in cls._loggers.values():
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()

        cls._loggers.clear()


class PerformanceLogger:
    """性能日志记录器"""

    def __init__(self, logger: logging.Logger):
        """方法描述"""
        self.start_time = None

    def start(self, operation: str):
        """开始计时"""
        self.operation = operation
        self.start_time = datetime.now()
        self.logger.debug(f"开始执行: {operation}")

    def end(self, success: bool = True, details: str = None):
        """结束计时"""
        if self.start_time is None:
            self.logger.warning("性能计时未开始")
            return

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        status = "成功" if success else "失败"
        message = f"执行完成: {self.operation} - {status} - 耗时: {duration:.3f}秒"

        if details:
            message += f" - {details}"

        if success:
            self.logger.info(message)
        else:
            self.logger.error(message)

        self.start_time = None

    def __enter__(self):
        """方法描述"""

    def __exit__(self, exc_type, exc_val, exc_tb):
        """方法描述"""
        error_details = str(exc_val) if exc_val else None
        self.end(success=success, details=error_details)


# 便捷函数 - 直接映射到LoggerManager的方法
setup_logger = LoggerManager.setup_logger
get_logger = LoggerManager.get_logger


def setup_factor_engine_logger(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """为因子引擎设置专用日志器"""
    default_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        "file_enabled": True,
        "file_path": "logs/factor_engine.log",
        "max_file_size": "50MB",
        "backup_count": 10,
        "console_enabled": True,
        "colored_output": True,
    }

    if config:
        default_config.update(config)

    return setup_logger("FactorEngine", default_config)


def setup_database_logger(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """为数据库操作设置专用日志器"""
    default_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - [DB] - %(message)s",
        "file_enabled": True,
        "file_path": "logs/database.log",
        "max_file_size": "20MB",
        "backup_count": 5,
        "console_enabled": False,  # 数据库日志通常不在控制台显示
        "colored_output": False,
    }

    if config:
        default_config.update(config)

    return setup_logger("Database", default_config)


def create_performance_logger(logger: logging.Logger) -> PerformanceLogger:
    """创建性能日志记录器"""
    return PerformanceLogger(logger)


# 示例用法
if __name__ == "__main__":
    # 设置基本日志器
    logger = setup_logger("test")

    # 测试不同级别的日志
    logger.debug("这是调试信息")
    logger.info("这是信息")
    logger.warning("这是警告")
    logger.error("这是错误")
    logger.critical("这是严重错误")

    # 测试性能日志
    perf_logger = create_performance_logger(logger)

    with perf_logger:
        perf_logger.start("测试操作")
        import time

        time.sleep(1)
        # 自动调用 end()

    # 手动使用性能日志
    perf_logger.start("另一个测试操作")
    time.sleep(0.5)
    perf_logger.end(success=True, details="处理了100条记录")

    print(f"已配置的日志器: {LoggerManager.list_loggers()}")
