#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API缓存机制
提供Redis缓存和内存缓存支持
"""

import json
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta
import logging

import redis
from src.utils.config_loader import config

logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.redis_client = self._init_redis()
        self.default_ttl = config.get('cache.default_ttl', 300)  # 5分钟
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """初始化Redis连接"""
        try:
            redis_url = config.get('redis.url', 'redis://localhost:6379/0')
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()  # 测试连接
            return client
        except Exception as e:
            logger.warning(f"Redis连接失败，将使用内存缓存: {e}")
            return None
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        # 创建参数的哈希值
        params_str = json.dumps({
            'args': args,
            'kwargs': sorted(kwargs.items())
        }, sort_keys=True, default=str)
        
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"{prefix}:{params_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            if self.redis_client:
                ttl = ttl or self.default_ttl
                serialized_value = json.dumps(value, default=str)
                return self.redis_client.setex(key, ttl, serialized_value)
            return False
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            return False
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False


# 全局缓存管理器实例
cache_manager = CacheManager()


def cache_response(
    prefix: str = "api",
    ttl: Optional[int] = None,
    skip_cache: Callable = None
):
    """
    API响应缓存装饰器
    
    Args:
        prefix: 缓存键前缀
        ttl: 缓存过期时间（秒）
        skip_cache: 跳过缓存的条件函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 检查是否跳过缓存
            if skip_cache and skip_cache(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # 生成缓存键
            cache_key = cache_manager._generate_cache_key(
                f"{prefix}:{func.__name__}", *args, **kwargs
            )
            
            # 尝试从缓存获取
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            
            # 只缓存成功的响应
            if hasattr(result, 'success') and result.success:
                cache_manager.set(cache_key, result.dict(), ttl)
                logger.debug(f"结果已缓存: {cache_key}")
            
            return result
        return wrapper
    return decorator


def invalidate_cache_pattern(pattern: str):
    """
    根据模式删除缓存
    
    Args:
        pattern: 缓存键模式，支持通配符
    """
    try:
        if cache_manager.redis_client:
            keys = cache_manager.redis_client.keys(pattern)
            if keys:
                cache_manager.redis_client.delete(*keys)
                logger.info(f"已删除 {len(keys)} 个缓存项，模式: {pattern}")
    except Exception as e:
        logger.error(f"删除缓存模式失败: {e}")


# 缓存失效策略
class CacheInvalidationStrategy:
    """缓存失效策略"""
    
    @staticmethod
    def on_factor_calculation_complete(task_id: str):
        """因子计算完成后的缓存失效"""
        # 删除因子查询相关的缓存
        invalidate_cache_pattern("api:get_factors:*")
        invalidate_cache_pattern("api:get_factor_metadata:*")
        logger.info(f"任务 {task_id} 完成，已清理相关缓存")
    
    @staticmethod
    def on_factor_standardization_complete():
        """因子标准化完成后的缓存失效"""
        # 删除标准化因子相关的缓存
        invalidate_cache_pattern("api:get_factors:*standardized*")
        logger.info("因子标准化完成，已清理相关缓存")