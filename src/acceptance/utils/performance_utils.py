"""
性能优化工具
"""
import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "peak_memory_mb": self.peak_memory_mb
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """开始性能监控"""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
        self.monitoring = True
        
        # 启动内存监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """停止监控并返回性能指标"""
        if not self.monitoring:
            raise ValueError("监控未启动")
        
        self.monitoring = False
        
        execution_time = time.time() - self.start_time
        current_memory = self._get_memory_usage()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=current_memory - self.start_memory,
            cpu_usage_percent=cpu_usage,
            peak_memory_mb=self.peak_memory
        )
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _monitor_memory(self):
        """监控内存使用峰值"""
        while self.monitoring:
            current_memory = self._get_memory_usage()
            self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(0.1)


@contextmanager
def performance_monitor():
    """性能监控上下文管理器"""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        metrics = monitor.stop_monitoring()
        print(f"性能指标: {metrics.to_dict()}")


class LazyInitializer:
    """延迟初始化工具"""
    
    def __init__(self, init_func: Callable, *args, **kwargs):
        self.init_func = init_func
        self.args = args
        self.kwargs = kwargs
        self._instance = None
        self._initialized = False
        self._lock = threading.Lock()
    
    def get_instance(self):
        """获取实例（延迟初始化）"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._instance = self.init_func(*self.args, **self.kwargs)
                    self._initialized = True
        
        return self._instance
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized


class BatchProcessor:
    """批处理工具"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_in_batches(self, items: list, process_func: Callable, 
                          **kwargs) -> list:
        """批量处理数据"""
        results = []
        
        # 分批处理
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = self._process_batch(batch, process_func, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def process_in_parallel(self, items: list, process_func: Callable, 
                           **kwargs) -> list:
        """并行处理数据"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_func, item, **kwargs): item 
                for item in items
            }
            
            # 收集结果
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"处理项目 {item} 时出错: {e}")
        
        return results
    
    def _process_batch(self, batch: list, process_func: Callable, 
                      **kwargs) -> list:
        """处理单个批次"""
        return [process_func(item, **kwargs) for item in batch]


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if time.time() - entry['timestamp'] > self.ttl_seconds:
                del self._cache[key]
                return None
            
            return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            # 如果缓存已满，删除最旧的条目
            if len(self._cache) >= self.max_size:
                oldest_key = min(
                    self._cache.keys(), 
                    key=lambda k: self._cache[k]['timestamp']
                )
                del self._cache[oldest_key]
            
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


def cached_method(ttl_seconds: int = 3600):
    """方法缓存装饰器"""
    cache = CacheManager(ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


# 使用示例
if __name__ == "__main__":
    # 性能监控示例
    with performance_monitor() as monitor:
        # 模拟一些计算
        time.sleep(1)
        data = [i for i in range(1000000)]
    
    # 批处理示例
    processor = BatchProcessor(batch_size=1000, max_workers=4)
    
    def square(x):
        return x * x
    
    numbers = list(range(10000))
    results = processor.process_in_parallel(numbers, square)
    
    # 缓存示例
    @cached_method(ttl_seconds=300)
    def expensive_calculation(n):
        time.sleep(0.1)  # 模拟耗时计算
        return n * n
    
    # 第一次调用会执行计算
    result1 = expensive_calculation(100)
    
    # 第二次调用会从缓存返回
    result2 = expensive_calculation(100)