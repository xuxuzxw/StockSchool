import functools
import logging
from typing import Any, Callable, Optional, Type

"""
配置管理错误处理装饰器
"""


logger = logging.getLogger(__name__)


def handle_config_errors(
    default_return: Any = None, log_error: bool = True, reraise: bool = False, error_message: Optional[str] = None
):
    """
    配置操作错误处理装饰器

    Args:
        default_return: 错误时的默认返回值
        log_error: 是否记录错误日志
        reraise: 是否重新抛出异常
        error_message: 自定义错误消息
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        def wrapper(*args, **kwargs):
            """方法描述"""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    message = error_message or f"{func.__name__} 执行失败"
                    logger.error(f"{message}: {e}", exc_info=True)

                if reraise:
                    raise

                return default_return

        return wrapper

    return decorator


class ConfigError(Exception):
    """配置相关异常基类"""

    pass


class ConfigValidationError(ConfigError):
    """配置验证异常"""

    pass


class ConfigLoadError(ConfigError):
    """配置加载异常"""

    pass


class ConfigWatchError(ConfigError):
    """配置监控异常"""

    pass
