#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试优化策略
"""

import threading
from typing import Dict, Any, Optional, List
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

logger = logging.getLogger(__name__)

class TestDataCache:
    """测试数据缓存"""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """存储缓存数据"""
        with self._lock:
            # 如果缓存已满，删除最久未使用的项
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'keys': list(self.cache.keys())
            }

class EnginePool:
    """引擎对象池"""
    
    def __init__(self, engine_factory: callable, pool_size: int = 5):
        self.engine_factory = engine_factory
        self.pool_size = pool_size
        self.engines: List[Any] = []
        self.available_engines: List[Any] = []
        self._lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化对象池"""
        for _ in range(self.pool_size):
            engine = self.engine_factory()
            self.engines.append(engine)
            self.available_engines.append(engine)
    
    def acquire(self) -> Any:
        """获取引擎实例"""
        with self._lock:
            if self.available_engines:
                return self.available_engines.pop()
            else:
                # 如果池中没有可用实例，创建新的
                logger.warning("引擎池已满，创建新实例")
                return self.engine_factory()
    
    def release(self, engine: Any):
        """释放引擎实例"""
        with self._lock:
            if len(self.available_engines) < self.pool_size:
                self.available_engines.append(engine)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息"""
        with self._lock:
            return {
                'total_engines': len(self.engines),
                'available_engines': len(self.available_engines),
                'pool_size': self.pool_size
            }

def cached_test_data(cache_key_func: callable = None):
    """测试数据缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_data = test_data_cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"使用缓存数据: {cache_key}")
                return cached_data
            
            # 生成新数据并缓存
            result = func(*args, **kwargs)
            test_data_cache.put(cache_key, result)
            logger.debug(f"缓存新数据: {cache_key}")
            
            return result
        return wrapper
    return decorator

class BatchProcessor:
    """批处理器 - 优化大量数据处理"""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_in_batches(self, items: List[Any], process_func: callable) -> List[Any]:
        """批量处理项目"""
        results = []
        
        # 分批处理
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = self._process_batch(batch, process_func)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[Any], process_func: callable) -> List[Any]:
        """处理单个批次"""
        if len(batch) == 1:
            # 单个项目直接处理
            return [process_func(batch[0])]
        
        # 多个项目并发处理
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
            future_to_item = {executor.submit(process_func, item): item for item in batch}
            results = []
            
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"批处理项目失败: {e}")
                    results.append(None)
            
            return results

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.optimization_strategies = {
            'cache_test_data': self._enable_test_data_caching,
            'pool_engines': self._enable_engine_pooling,
            'batch_processing': self._enable_batch_processing,
            'parallel_execution': self._enable_parallel_execution
        }
    
    def apply_optimizations(self, strategies: List[str], test_config: Dict[str, Any]):
        """应用优化策略"""
        for strategy in strategies:
            if strategy in self.optimization_strategies:
                self.optimization_strategies[strategy](test_config)
                logger.info(f"已应用优化策略: {strategy}")
            else:
                logger.warning(f"未知的优化策略: {strategy}")
    
    def _enable_test_data_caching(self, config: Dict[str, Any]):
        """启用测试数据缓存"""
        config['use_data_cache'] = True
        config['cache_size'] = config.get('cache_size', 50)
    
    def _enable_engine_pooling(self, config: Dict[str, Any]):
        """启用引擎池化"""
        config['use_engine_pool'] = True
        config['pool_size'] = config.get('pool_size', 5)
    
    def _enable_batch_processing(self, config: Dict[str, Any]):
        """启用批处理"""
        config['use_batch_processing'] = True
        config['batch_size'] = config.get('batch_size', 10)
    
    def _enable_parallel_execution(self, config: Dict[str, Any]):
        """启用并行执行"""
        config['use_parallel_execution'] = True
        config['max_workers'] = config.get('max_workers', 4)

# 全局实例
test_data_cache = TestDataCache()
performance_optimizer = PerformanceOptimizer()

# 优化后的测试数据生成函数示例
@cached_test_data(lambda stock_count, days: f"stock_data_{stock_count}_{days}")
def generate_optimized_test_data(stock_count: int, days: int) -> Dict[str, Any]:
    """优化的测试数据生成"""
    from tests.utils.test_data_generator import TestDataGenerator
    from datetime import date, timedelta
    
    generator = TestDataGenerator()
    
    # 生成股票列表
    stock_codes = [f"TEST{i:06d}.SZ" for i in range(stock_count)]
    
    # 生成日期范围
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    # 生成数据
    data = {}
    for ts_code in stock_codes:
        data[ts_code] = generator.generate_stock_daily_data(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
    
    return {
        'stock_data': data,
        'stock_codes': stock_codes,
        'date_range': (start_date, end_date),
        'generated_at': time.time()
    }

class OptimizedPerformanceTest:
    """优化的性能测试基类"""
    
    def __init__(self, engine_factory: callable):
        self.engine_factory = engine_factory
        self.engine_pool = EnginePool(engine_factory, pool_size=3)
        self.batch_processor = BatchProcessor(batch_size=5, max_workers=2)
    
    def run_optimized_test(self, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """运行优化的性能测试"""
        # 应用优化策略
        optimization_strategies = test_params.get('optimizations', ['cache_test_data'])
        optimized_config = test_params.copy()
        performance_optimizer.apply_optimizations(optimization_strategies, optimized_config)
        
        # 获取引擎实例
        engine = self.engine_pool.acquire()
        
        try:
            # 执行测试
            if optimized_config.get('use_batch_processing', False):
                return self._run_batch_test(engine, optimized_config)
            else:
                return self._run_single_test(engine, optimized_config)
        finally:
            # 释放引擎实例
            self.engine_pool.release(engine)
    
    def _run_batch_test(self, engine: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行批处理测试"""
        stock_codes = config.get('stock_codes', ['000001.SZ'])
        
        def process_stock(ts_code):
            return engine.calculate_sma([ts_code], 
                                      config.get('start_date'),
                                      config.get('end_date'),
                                      config.get('window', 5))
        
        results = self.batch_processor.process_in_batches(stock_codes, process_stock)
        
        return {
            'results': results,
            'processed_stocks': len(stock_codes),
            'successful_results': len([r for r in results if r is not None])
        }
    
    def _run_single_test(self, engine: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个测试"""
        result = engine.calculate_sma(
            config.get('stock_codes', ['000001.SZ']),
            config.get('start_date'),
            config.get('end_date'),
            config.get('window', 5)
        )
        
        return {
            'result': result,
            'data_count': len(result) if not result.empty else 0
        }