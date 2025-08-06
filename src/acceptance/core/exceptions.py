"""
自定义异常类和异常处理装饰器
"""

import functools
import logging
from typing import Callable, Any, Type, Union


class AcceptanceTestError(Exception):
    """验收测试基础异常"""
    pass


class ConfigurationError(AcceptanceTestError):
    """配置错误"""
    pass


class InfrastructureError(AcceptanceTestError):
    """基础设施错误"""
    pass


class DatabaseConnectionError(InfrastructureError):
    """数据库连接错误"""
    pass


class RedisConnectionError(InfrastructureError):
    """Redis连接错误"""
    pass


class AIAnalysisError(AcceptanceTestError):
    """AI分析错误"""
    pass


class TestExecutionError(AcceptanceTestError):
    """测试执行错误"""
    pass


def handle_test_exceptions(
    logger: logging.Logger,
    default_return: Any = None,
    reraise: bool = True
):
    """
    测试异常处理装饰器
    
    Args:
        logger: 日志记录器
        default_return: 异常时的默认返回值
        reraise: 是否重新抛出异常
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AcceptanceTestError as e:
                logger.error(f"验收测试错误 in {func.__name__}: {e}")
                if reraise:
                    raise
                return default_return
            except Exception as e:
                logger.error(f"未预期错误 in {func.__name__}: {e}", exc_info=True)
                if reraise:
                    raise AcceptanceTestError(f"测试执行失败: {e}") from e
                return default_return
        
        return wrapper
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    失败重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 重试间隔（秒）
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                        continue
                    raise
            
            # 这行代码理论上不会执行到，但为了类型检查
            raise last_exception
        
        return wrapper
    return decorator