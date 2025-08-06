"""
性能优化工具
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict
from functools import wraps


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
    
    def time_function(self, func_name: str = None):
        """函数执行时间装饰器"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self.record_metric(name, execution_time)
            
            return wrapper
        return decorator
    
    def record_metric(self, name: str, value: float):
        """记录性能指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取性能摘要"""
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                'count': len(values),
                'total': sum(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
        return summary


class ParallelExecutor:
    """并行执行器"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
    
    def execute_parallel(
        self, 
        tasks: List[Callable], 
        timeout: float = None
    ) -> List[Any]:
        """并行执行任务列表"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(task): task for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        return results
    
    async def execute_async(
        self, 
        async_tasks: List[Callable], 
        timeout: float = None
    ) -> List[Any]:
        """异步执行任务列表"""
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task() for task in async_tasks], return_exceptions=True),
                timeout=timeout
            )
            return results
        except asyncio.TimeoutError:
            raise TimeoutError(f"异步任务执行超时: {timeout}秒")


class ResourcePool:
    """资源池管理器"""
    
    def __init__(self, resource_factory: Callable, max_size: int = 10):
        self.resource_factory = resource_factory
        self.max_size = max_size
        self._pool = []
        self._in_use = set()
    
    def acquire(self):
        """获取资源"""
        if self._pool:
            resource = self._pool.pop()
        else:
            resource = self.resource_factory()
        
        self._in_use.add(resource)
        return resource
    
    def release(self, resource):
        """释放资源"""
        if resource in self._in_use:
            self._in_use.remove(resource)
            if len(self._pool) < self.max_size:
                self._pool.append(resource)
    
    def __enter__(self):
        return self.acquire()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 注意：这里需要传入resource，实际使用时需要改进
        pass


def cache_result(ttl: int = 300):
    """结果缓存装饰器"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            current_time = time.time()
            
            # 检查缓存
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl:
                    return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            
            return result
        
        return wrapper
    return decorator