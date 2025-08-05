import functools
import threading
import time
from typing import Any, Callable, Dict

"""
配置管理性能优化
"""


class ConfigCache:
    """配置缓存"""

    def __init__(self, ttl: int = 300):
        """方法描述"""
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        """获取缓存值"""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self._cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()


def cached_config(ttl: int = 300):
    """配置缓存装饰器"""
    cache = ConfigCache(ttl)

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        def wrapper(*args, **kwargs):
            """方法描述"""
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result)

            return result

        # 添加清理缓存的方法
        wrapper.clear_cache = cache.clear

        return wrapper

    return decorator
