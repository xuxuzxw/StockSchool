import hashlib
import json
import pickle
import threading
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import redis
from loguru import logger
from redis.exceptions import ConnectionError, TimeoutError

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子数据缓存机制
实现Redis缓存和内存缓存的因子数据管理
"""


from .factor_models import FactorResult, FactorType, FactorValue


class CacheConfig:
    """缓存配置"""

    def __init__(self):
        """方法描述"""
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.redis_db = 0
        self.redis_password = None
        self.redis_socket_timeout = 5
        self.redis_socket_connect_timeout = 5

        # 缓存过期时间（秒）
        self.default_ttl = 3600  # 1小时
        self.factor_data_ttl = 7200  # 2小时
        self.calculation_result_ttl = 1800  # 30分钟
        self.metadata_ttl = 86400  # 24小时

        # 内存缓存配置
        self.memory_cache_size = 1000  # 最大缓存项数
        self.memory_cache_ttl = 1800  # 30分钟

        # 缓存键前缀
        self.key_prefix = "stockschool:factor:"

        # 序列化配置
        self.use_compression = True
        self.compression_threshold = 1024  # 超过1KB的数据进行压缩


class MemoryCache:
    """内存缓存实现"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 1800):
        """初始化内存缓存"""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.access_times = {}
        self.expire_times = {}
        self.lock = threading.RLock()

    def _generate_key(self, key: str) -> str:
        """生成缓存键"""
        return hashlib.md5(key.encode()).hexdigest()

    def _is_expired(self, key: str) -> bool:
        """检查缓存项是否过期"""
        if key not in self.expire_times:
            return True
        return time.time() > self.expire_times[key]

    def _evict_expired(self):
        """清理过期缓存项"""
        current_time = time.time()
        expired_keys = [key for key, expire_time in self.expire_times.items() if current_time > expire_time]

        for key in expired_keys:
            self._remove_key(key)

    def _evict_lru(self):
        """LRU淘汰策略"""
        while len(self.cache) >= self.max_size:
            # 移除最久未访问的项
            oldest_key = next(iter(self.cache))
            self._remove_key(oldest_key)

    def _remove_key(self, key: str):
        """移除缓存项"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expire_times.pop(key, None)

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            cache_key = self._generate_key(key)

            # 检查是否存在且未过期
            if cache_key not in self.cache or self._is_expired(cache_key):
                return None

            # 更新访问时间
            self.access_times[cache_key] = time.time()

            # 移动到末尾（LRU）
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value

            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        with self.lock:
            try:
                cache_key = self._generate_key(key)

                # 清理过期项
                self._evict_expired()

                # LRU淘汰
                self._evict_lru()

                # 设置缓存
                self.cache[cache_key] = value
                self.access_times[cache_key] = time.time()

                # 设置过期时间
                expire_ttl = ttl or self.default_ttl
                self.expire_times[cache_key] = time.time() + expire_ttl

                return True

            except Exception as e:
                logger.error(f"内存缓存设置失败: {e}")
                return False

    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            cache_key = self._generate_key(key)
            if cache_key in self.cache:
                self._remove_key(cache_key)
                return True
            return False

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.expire_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            current_time = time.time()
            expired_count = sum(1 for expire_time in self.expire_times.values() if current_time > expire_time)

            return {
                "total_items": len(self.cache),
                "expired_items": expired_count,
                "valid_items": len(self.cache) - expired_count,
                "max_size": self.max_size,
                "usage_ratio": len(self.cache) / self.max_size,
            }


class RedisCache:
    """Redis缓存实现"""

    def __init__(self, config: CacheConfig):
        """初始化Redis缓存"""
        self.config = config
        self.redis_client = None
        self._connect()

    def _connect(self):
        """连接Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                socket_timeout=self.config.redis_socket_timeout,
                socket_connect_timeout=self.config.redis_socket_connect_timeout,
                decode_responses=False,  # 保持二进制模式以支持pickle
            )

            # 测试连接
            self.redis_client.ping()
            logger.info("Redis连接成功")

        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            self.redis_client = None

    def _generate_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.config.key_prefix}{key}"

    def _serialize_value(self, value: Any) -> bytes:
        """序列化值"""
        try:
            # 使用pickle序列化
            serialized = pickle.dumps(value)

            # 如果启用压缩且数据大于阈值
            if self.config.use_compression and len(serialized) > self.config.compression_threshold:
                import gzip

                serialized = gzip.compress(serialized)
                # 添加压缩标记
                serialized = b"GZIP:" + serialized

            return serialized

        except Exception as e:
            logger.error(f"序列化失败: {e}")
            raise

    def _deserialize_value(self, data: bytes) -> Any:
        """反序列化值"""
        try:
            # 检查是否压缩
            if data.startswith(b"GZIP:"):
                import gzip

                data = gzip.decompress(data[5:])  # 移除'GZIP:'前缀

            return pickle.loads(data)

        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        if not self.redis_client:
            return None

        try:
            redis_key = self._generate_key(key)
            data = self.redis_client.get(redis_key)

            if data is None:
                return None

            return self._deserialize_value(data)

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis连接异常: {e}")
            self._connect()  # 尝试重连
            return None
        except Exception as e:
            logger.error(f"Redis获取缓存失败: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        if not self.redis_client:
            return False

        try:
            redis_key = self._generate_key(key)
            serialized_value = self._serialize_value(value)

            expire_ttl = ttl or self.config.default_ttl

            result = self.redis_client.setex(redis_key, expire_ttl, serialized_value)
            return bool(result)

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis连接异常: {e}")
            self._connect()  # 尝试重连
            return False
        except Exception as e:
            logger.error(f"Redis设置缓存失败: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存项"""
        if not self.redis_client:
            return False

        try:
            redis_key = self._generate_key(key)
            result = self.redis_client.delete(redis_key)
            return bool(result)

        except Exception as e:
            logger.error(f"Redis删除缓存失败: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查缓存项是否存在"""
        if not self.redis_client:
            return False

        try:
            redis_key = self._generate_key(key)
            return bool(self.redis_client.exists(redis_key))

        except Exception as e:
            logger.error(f"Redis检查存在性失败: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """获取缓存项的剩余生存时间"""
        if not self.redis_client:
            return -1

        try:
            redis_key = self._generate_key(key)
            return self.redis_client.ttl(redis_key)

        except Exception as e:
            logger.error(f"Redis获取TTL失败: {e}")
            return -1

    def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存项"""
        if not self.redis_client:
            return 0

        try:
            redis_pattern = self._generate_key(pattern)
            keys = self.redis_client.keys(redis_pattern)

            if keys:
                return self.redis_client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Redis批量删除失败: {e}")
            return 0


class FactorCache:
    """因子数据缓存管理器"""

    def __init__(self, config: Optional[CacheConfig] = None):
        """初始化因子缓存"""
        self.config = config or Cacheget_config()

        # 初始化缓存层
        self.memory_cache = MemoryCache(
            max_size=self.config.memory_cache_size, default_ttl=self.config.memory_cache_ttl
        )

        self.redis_cache = RedisCache(self.config)

        # 缓存统计
        self.stats = {"memory_hits": 0, "redis_hits": 0, "misses": 0, "sets": 0, "deletes": 0}

    def _generate_factor_key(
        self, ts_code: str, factor_name: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> str:
        """生成因子数据缓存键"""
        key_parts = ["factor_data", ts_code, factor_name]

        if start_date:
            key_parts.append(start_date.isoformat())
        if end_date:
            key_parts.append(end_date.isoformat())

        return ":".join(key_parts)

    def _generate_result_key(self, ts_code: str, factor_type: str, calculation_params: Dict[str, Any]) -> str:
        """生成计算结果缓存键"""
        # 创建参数的哈希值
        params_str = json.dumps(calculation_params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        return f"result:{factor_type}:{ts_code}:{params_hash}"

    def get_factor_data(
        self, ts_code: str, factor_name: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> Optional[List[FactorValue]]:
        """获取因子数据（已重构为使用统一服务）"""
        try:
            from src.data.factor_data_service import DataSource, FactorDataService, FactorQuery

            # 创建统一查询
            query = FactorQuery(
                factor_names=[factor_name],
                ts_codes=[ts_code],
                start_date=start_date,
                end_date=end_date,
                data_source=DataSource.CACHE,
            )

            # 使用统一服务获取数据
            service = FactorDataService()
            result = service.get_factor_data(query)

            if result.data.empty:
                return None

            # 转换为FactorValue格式
            factor_values = []
            for _, row in result.data.iterrows():
                factor_values.append(
                    FactorValue(
                        ts_code=row.get("ts_code", ts_code),
                        trade_date=row.get("trade_date"),
                        factor_name=factor_name,
                        value=row.get("value"),
                        quality_flag=row.get("quality_flag", 1),
                    )
                )

            return factor_values

        except ImportError:
            # 如果新服务不可用，回退到原实现
            logger.warning("使用旧版缓存实现，建议升级到统一服务")
            return self._legacy_get_factor_data(ts_code, factor_name, start_date, end_date)
        except Exception as e:
            logger.error(f"获取因子数据失败: {e}")
            return None

    def _legacy_get_factor_data(
        self, ts_code: str, factor_name: str, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> Optional[List[FactorValue]]:
        """旧版实现，用于向后兼容"""
        cache_key = self._generate_factor_key(ts_code, factor_name, start_date, end_date)

        # 先尝试内存缓存
        data = self.memory_cache.get(cache_key)
        if data is not None:
            self.stats["memory_hits"] += 1
            return data

        # 再尝试Redis缓存
        data = self.redis_cache.get(cache_key)
        if data is not None:
            self.stats["redis_hits"] += 1
            # 回写到内存缓存
            self.memory_cache.set(cache_key, data, self.config.memory_cache_ttl)
            return data

        self.stats["misses"] += 1
        return None

    def set_factor_data(
        self,
        ts_code: str,
        factor_name: str,
        factor_values: List[FactorValue],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置因子数据"""
        cache_key = self._generate_factor_key(ts_code, factor_name, start_date, end_date)
        cache_ttl = ttl or self.config.factor_data_ttl

        # 同时设置内存缓存和Redis缓存
        memory_success = self.memory_cache.set(cache_key, factor_values, cache_ttl)
        redis_success = self.redis_cache.set(cache_key, factor_values, cache_ttl)

        if memory_success or redis_success:
            self.stats["sets"] += 1
            return True

        return False

    def get_calculation_result(
        self, ts_code: str, factor_type: str, calculation_params: Dict[str, Any]
    ) -> Optional[FactorResult]:
        """获取计算结果"""
        cache_key = self._generate_result_key(ts_code, factor_type, calculation_params)

        # 先尝试内存缓存
        result = self.memory_cache.get(cache_key)
        if result is not None:
            self.stats["memory_hits"] += 1
            return result

        # 再尝试Redis缓存
        result = self.redis_cache.get(cache_key)
        if result is not None:
            self.stats["redis_hits"] += 1
            # 回写到内存缓存
            self.memory_cache.set(cache_key, result, self.config.memory_cache_ttl)
            return result

        self.stats["misses"] += 1
        return None

    def set_calculation_result(
        self,
        ts_code: str,
        factor_type: str,
        calculation_params: Dict[str, Any],
        result: FactorResult,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置计算结果"""
        cache_key = self._generate_result_key(ts_code, factor_type, calculation_params)
        cache_ttl = ttl or self.config.calculation_result_ttl

        # 同时设置内存缓存和Redis缓存
        memory_success = self.memory_cache.set(cache_key, result, cache_ttl)
        redis_success = self.redis_cache.set(cache_key, result, cache_ttl)

        if memory_success or redis_success:
            self.stats["sets"] += 1
            return True

        return False

    def invalidate_factor(self, ts_code: str, factor_name: str):
        """使因子缓存失效"""
        pattern = f"factor_data:{ts_code}:{factor_name}:*"

        # 清除Redis缓存
        deleted_count = self.redis_cache.clear_pattern(pattern)

        # 内存缓存无法按模式删除，只能清空相关项
        # 这里简化处理，实际可以改进

        self.stats["deletes"] += deleted_count
        logger.info(f"使因子 {factor_name} 的缓存失效，删除了 {deleted_count} 个缓存项")

    def invalidate_stock(self, ts_code: str):
        """使股票相关的所有缓存失效"""
        patterns = [f"factor_data:{ts_code}:*", f"result:*:{ts_code}:*"]

        total_deleted = 0
        for pattern in patterns:
            deleted_count = self.redis_cache.clear_pattern(pattern)
            total_deleted += deleted_count

        self.stats["deletes"] += total_deleted
        logger.info(f"使股票 {ts_code} 的缓存失效，删除了 {total_deleted} 个缓存项")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        memory_stats = self.memory_cache.get_stats()

        total_requests = self.stats["memory_hits"] + self.stats["redis_hits"] + self.stats["misses"]

        hit_rate = 0
        if total_requests > 0:
            hit_rate = ((self.stats["memory_hits"] + self.stats["redis_hits"]) / total_requests) * 100

        return {
            "requests": {
                "total": total_requests,
                "memory_hits": self.stats["memory_hits"],
                "redis_hits": self.stats["redis_hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
            },
            "operations": {"sets": self.stats["sets"], "deletes": self.stats["deletes"]},
            "memory_cache": memory_stats,
            "redis_available": self.redis_cache.redis_client is not None,
        }

    def clear_all_cache(self):
        """清空所有缓存"""
        # 清空内存缓存
        self.memory_cache.clear()

        # 清空Redis缓存
        self.redis_cache.clear_pattern("*")

        logger.info("已清空所有缓存")


def cache_factor_data(cache_ttl: int = 3600):
    """因子数据缓存装饰器"""

    def decorator(func):
        """方法描述"""

        def wrapper(self, ts_code: str, *args, **kwargs):
            """方法描述"""
            if not hasattr(self, "cache") or self.cache is None:
                return func(self, ts_code, *args, **kwargs)

            # 生成缓存键
            func_name = func.__name__
            cache_key = f"{func_name}:{ts_code}:{hash(str(args) + str(kwargs))}"

            # 尝试从缓存获取
            cached_result = self.cache.memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 执行原函数
            result = func(self, ts_code, *args, **kwargs)

            # 缓存结果
            if result is not None:
                self.cache.memory_cache.set(cache_key, result, cache_ttl)

            return result

        return wrapper

    return decorator


def cache_calculation_result(cache_ttl: int = 1800):
    """计算结果缓存装饰器"""

    def decorator(func):
        """方法描述"""

        def wrapper(self, ts_code: str, *args, **kwargs):
            """方法描述"""
            if not hasattr(self, "cache") or self.cache is None:
                return func(self, ts_code, *args, **kwargs)

            # 生成缓存参数
            calculation_params = {"args": args, "kwargs": kwargs, "function": func.__name__}

            # 尝试从缓存获取
            cached_result = self.cache.get_calculation_result(ts_code, "calculation", calculation_params)
            if cached_result is not None:
                return cached_result

            # 执行原函数
            result = func(self, ts_code, *args, **kwargs)

            # 缓存结果
            if result is not None:
                self.cache.set_calculation_result(ts_code, "calculation", calculation_params, result, cache_ttl)

            return result

        return wrapper

    return decorator
