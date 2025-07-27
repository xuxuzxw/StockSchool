import time
from functools import wraps

def idempotent_retry(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i+1} failed for {func.__name__}: {e}")
                    if i == max_retries - 1:
                        raise e
                    time.sleep(2 ** i)
        return wrapper
    return decorator