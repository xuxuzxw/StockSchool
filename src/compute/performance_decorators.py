#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化装饰器
提供缓存、计时、内存监控等功能
"""

import time
import hashlib
import functools
import psutil
import os
from typing import Any, Callable, Dict, Optional
from loguru import logger
import pandas as pd


def timing_decorator(func: Callable) -> Callable:
    """
    计时装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(f"{func.__name__} 执行时间: {execution_time:.3f}秒")
            
            # 如果结果是字典且包含多个因子，记录每个因子的平均时间
            if isinstance(result, dict) and len(result) > 1:
                avg_time_per_factor = execution_time / len(result)
                logger.debug(f"平均每个因子计算时间: {avg_time_per_factor:.3f}秒")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} 执行失败，耗时: {execution_time:.3f}秒，错误: {e}")
            raise
    
    return wrapper


def memory_monitor_decorator(func: Callable) -> Callable:
    """
    内存监控装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # 记录开始时的内存使用
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            # 记录结束时的内存使用
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            if memory_increase > 10:  # 超过10MB才记录
                logger.debug(f"{func.__name__} 内存使用增加: {memory_increase:.1f}MB")
            
            return result
            
        except Exception as e:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            logger.error(f"{func.__name__} 执行失败，内存使用增加: {memory_increase:.1f}MB")
            raise
    
    return wrapper


class FactorCalculationCache:
    """因子计算缓存类"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存生存时间（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def _generate_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            func_name: 函数名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存键
        """
        # 对于DataFrame参数，使用其形状和列名生成哈希
        cache_parts = [func_name]
        
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # 使用DataFrame的形状、列名和部分数据生成哈希
                df_info = f"shape:{arg.shape},cols:{list(arg.columns)}"
                if not arg.empty:
                    # 使用前几行和后几行的数据
                    sample_data = pd.concat([arg.head(2), arg.tail(2)])
                    df_info += f",sample:{sample_data.values.tobytes().hex()[:32]}"
                cache_parts.append(df_info)
            else:
                cache_parts.append(str(arg))
        
        for key, value in sorted(kwargs.items()):
            cache_parts.append(f"{key}:{value}")
        
        cache_string = "|".join(cache_parts)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """
        检查缓存条目是否过期
        
        Args:
            cache_entry: 缓存条目
            
        Returns:
            是否过期
        """
        if self.ttl_seconds <= 0:
            return False
        
        return time.time() - cache_entry['timestamp'] > self.ttl_seconds
    
    def _cleanup_expired(self) -> None:
        """清理过期的缓存条目"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def _cleanup_lru(self) -> None:
        """清理最久未使用的缓存条目"""
        if len(self._cache) <= self.max_size:
            return
        
        # 按访问时间排序，删除最旧的条目
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1]['last_access']
        )
        
        items_to_remove = len(self._cache) - self.max_size + 1
        for i in range(items_to_remove):
            key = sorted_items[i][0]
            del self._cache[key]
        
        logger.debug(f"清理了 {items_to_remove} 个LRU缓存条目")
    
    def get(self, func_name: str, *args, **kwargs) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            func_name: 函数名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存的结果，如果不存在或过期则返回None
        """
        cache_key = self._generate_cache_key(func_name, *args, **kwargs)
        
        if cache_key not in self._cache:
            return None
        
        cache_entry = self._cache[cache_key]
        
        if self._is_expired(cache_entry):
            del self._cache[cache_key]
            return None
        
        # 更新访问时间
        cache_entry['last_access'] = time.time()
        cache_entry['hit_count'] += 1
        
        logger.debug(f"缓存命中: {func_name}")
        return cache_entry['result']
    
    def set(self, func_name: str, result: Any, *args, **kwargs) -> None:
        """
        设置缓存值
        
        Args:
            func_name: 函数名
            result: 计算结果
            *args: 位置参数
            **kwargs: 关键字参数
        """
        cache_key = self._generate_cache_key(func_name, *args, **kwargs)
        
        # 清理过期条目
        self._cleanup_expired()
        
        # 清理LRU条目
        self._cleanup_lru()
        
        # 存储新的缓存条目
        self._cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'last_access': time.time(),
            'hit_count': 0
        }
        
        logger.debug(f"缓存设置: {func_name}")
    
    def clear(self) -> None:
        """清空所有缓存"""
        cache_count = len(self._cache)
        self._cache.clear()
        logger.info(f"清空了 {cache_count} 个缓存条目")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        total_hits = sum(entry['hit_count'] for entry in self._cache.values())
        
        return {
            'total_entries': len(self._cache),
            'total_hits': total_hits,
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


# 全局缓存实例
_global_cache = FactorCalculationCache()


def cached_calculation(cache: Optional[FactorCalculationCache] = None, 
                      ttl_seconds: Optional[int] = None):
    """
    缓存计算结果的装饰器
    
    Args:
        cache: 缓存实例，如果为None则使用全局缓存
        ttl_seconds: 缓存生存时间，如果为None则使用缓存实例的默认值
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache
            if cache is None:
                cache = _global_cache
            
            # 尝试从缓存获取结果
            cached_result = cache.get(func.__name__, *args, **kwargs)
            if cached_result is not None:
                return cached_result
            
            # 执行计算
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache.set(func.__name__, result, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator


def performance_optimized(enable_cache: bool = True, 
                         enable_timing: bool = True,
                         enable_memory_monitor: bool = False):
    """
    综合性能优化装饰器
    
    Args:
        enable_cache: 是否启用缓存
        enable_timing: 是否启用计时
        enable_memory_monitor: 是否启用内存监控
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        # 应用装饰器
        optimized_func = func
        
        if enable_cache:
            optimized_func = cached_calculation()(optimized_func)
        
        if enable_timing:
            optimized_func = timing_decorator(optimized_func)
        
        if enable_memory_monitor:
            optimized_func = memory_monitor_decorator(optimized_func)
        
        return optimized_func
    
    return decorator