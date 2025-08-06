"""
验收测试装饰器 - 统一异常处理和日志记录
"""

import functools
import time
from typing import Callable, Any, Dict
from ..core.exceptions import AcceptanceTestError


def test_method_wrapper(test_name: str = None, timeout: int = 60):
    """
    测试方法装饰器 - 统一异常处理、超时控制和日志记录
    
    Args:
        test_name: 测试名称，用于日志和异常信息
        timeout: 超时时间（秒）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Dict[str, Any]:
            method_test_name = test_name or func.__name__
            start_time = time.time()
            
            # 记录开始日志
            self.logger.info(f"开始执行测试: {method_test_name}")
            
            try:
                # 执行测试方法
                result = func(self, *args, **kwargs)
                
                # 记录成功日志
                execution_time = time.time() - start_time
                self.logger.info(
                    f"测试 {method_test_name} 执行成功",
                    extra={
                        "test_name": method_test_name,
                        "execution_time": execution_time,
                        "status": "success"
                    }
                )
                
                # 添加执行时间到结果
                if isinstance(result, dict):
                    result["execution_time"] = execution_time
                
                return result
                
            except AcceptanceTestError:
                # 重新抛出验收测试异常
                execution_time = time.time() - start_time
                self.logger.error(
                    f"测试 {method_test_name} 执行失败",
                    extra={
                        "test_name": method_test_name,
                        "execution_time": execution_time,
                        "status": "failed"
                    }
                )
                raise
                
            except Exception as e:
                # 包装其他异常为验收测试异常
                execution_time = time.time() - start_time
                error_message = f"{method_test_name} 测试失败: {str(e)}"
                
                self.logger.error(
                    error_message,
                    extra={
                        "test_name": method_test_name,
                        "execution_time": execution_time,
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    exc_info=True
                )
                
                raise AcceptanceTestError(error_message) from e
                
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器 - 在失败时自动重试
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间倍数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                    
                except AcceptanceTestError as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        self.logger.warning(
                            f"测试 {func.__name__} 第 {attempt + 1} 次尝试失败，{current_delay}秒后重试: {str(e)}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        self.logger.error(f"测试 {func.__name__} 重试 {max_retries} 次后仍然失败")
                        
                except Exception as e:
                    # 对于非验收测试异常，不进行重试
                    raise AcceptanceTestError(f"{func.__name__} 执行异常: {str(e)}") from e
            
            # 抛出最后一次的异常
            raise last_exception
            
        return wrapper
    return decorator


def performance_monitor(threshold_seconds: float = 30.0):
    """
    性能监控装饰器 - 监控方法执行时间
    
    Args:
        threshold_seconds: 性能警告阈值（秒）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(self, *args, **kwargs)
                execution_time = time.time() - start_time
                
                # 记录性能信息
                if execution_time > threshold_seconds:
                    self.logger.warning(
                        f"方法 {func.__name__} 执行时间过长: {execution_time:.2f}秒",
                        extra={
                            "method_name": func.__name__,
                            "execution_time": execution_time,
                            "threshold": threshold_seconds,
                            "performance_warning": True
                        }
                    )
                else:
                    self.logger.debug(
                        f"方法 {func.__name__} 执行完成: {execution_time:.2f}秒",
                        extra={
                            "method_name": func.__name__,
                            "execution_time": execution_time
                        }
                    )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(
                    f"方法 {func.__name__} 执行失败: {execution_time:.2f}秒",
                    extra={
                        "method_name": func.__name__,
                        "execution_time": execution_time,
                        "error": str(e)
                    }
                )
                raise
                
        return wrapper
    return decorator


def validate_prerequisites(*required_attrs):
    """
    前提条件验证装饰器 - 确保必需的属性已初始化
    
    Args:
        required_attrs: 必需的属性名列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # 检查必需属性
            missing_attrs = []
            for attr in required_attrs:
                if not hasattr(self, attr) or getattr(self, attr) is None:
                    missing_attrs.append(attr)
            
            if missing_attrs:
                raise AcceptanceTestError(
                    f"方法 {func.__name__} 执行前提条件不满足，缺少属性: {missing_attrs}"
                )
            
            return func(self, *args, **kwargs)
            
        return wrapper
    return decorator