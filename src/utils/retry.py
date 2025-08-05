import time
from functools import wraps
from typing import Any, Callable, Optional

from ..config.unified_config import config


def idempotent_retry(max_retries=None):
    """方法描述"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            """方法描述"""
            if max_retries is None:
                retries = config.get('utils_params.max_retries', 3)
            else:
                retries = max_retries
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i+1} failed for {func.__name__}: {e}")
                    if i == retries - 1:
                        raise e
                    sleep_time = config.get('utils_params.retry_sleep_base', 2) ** i
                    time.sleep(sleep_time)
        return wrapper
    return decorator