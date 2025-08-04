#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统装饰器

提供通用的错误处理、重试、缓存等装饰器
"""

import functools
import time
from typing import Any, Callable, Optional, Type, Union
from loguru import logger


def handle_exceptions(
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False,
    exception_types: tuple = (Exception,)
):
    """
    异常处理装饰器
    
    Args:
        default_return: 异常时的默认返回值
        log_error: 是否记录错误日志
        reraise: 是否重新抛出异常
        exception_types: 要捕获的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    logger.error(f"{func.__name__} 执行失败: {e}")
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exception_types: tuple = (Exception,)
):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 退避因子
        exception_types: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception_types as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(f"{func.__name__} 第{attempt + 1}次尝试失败: {e}, {current_delay}秒后重试")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} 重试{max_retries}次后仍然失败: {e}")
            
            raise last_exception
        return wrapper
    return decorator


def measure_execution_time(log_result: bool = True):
    """
    执行时间测量装饰器
    
    Args:
        log_result: 是否记录执行时间
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                if log_result:
                    logger.debug(f"{func.__name__} 执行时间: {execution_time:.3f}秒")
                
                # 可以将执行时间存储到指标系统
                # metrics.record_execution_time(func.__name__, execution_time)
        return wrapper
    return decorator


def cache_result(ttl: int = 300, key_func: Optional[Callable] = None):
    """
    结果缓存装饰器
    
    Args:
        ttl: 缓存生存时间（秒）
        key_func: 自定义缓存键生成函数
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 检查缓存
            if cache_key in cache:
                cached_result, cached_time = cache[cache_key]
                if time.time() - cached_time < ttl:
                    logger.debug(f"{func.__name__} 使用缓存结果")
                    return cached_result
                else:
                    # 缓存过期，删除
                    del cache[cache_key]
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            return result
        
        # 添加清理缓存的方法
        wrapper.clear_cache = lambda: cache.clear()
        
        return wrapper
    return decorator


def validate_parameters(**validators):
    """
    参数验证装饰器
    
    Args:
        **validators: 参数名到验证函数的映射
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证参数
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"参数 {param_name} 验证失败: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 组合装饰器示例
def monitoring_operation(
    max_retries: int = 3,
    cache_ttl: int = 0,
    measure_time: bool = True
):
    """
    监控操作组合装饰器
    
    Args:
        max_retries: 最大重试次数
        cache_ttl: 缓存时间（0表示不缓存）
        measure_time: 是否测量执行时间
    """
    def decorator(func: Callable) -> Callable:
        # 应用装饰器链
        decorated_func = func
        
        if measure_time:
            decorated_func = measure_execution_time()(decorated_func)
        
        if cache_ttl > 0:
            decorated_func = cache_result(ttl=cache_ttl)(decorated_func)
        
        if max_retries > 0:
            decorated_func = retry_on_failure(max_retries=max_retries)(decorated_func)
        
        decorated_func = handle_exceptions(log_error=True)(decorated_func)
        
        return decorated_func
    return decorator