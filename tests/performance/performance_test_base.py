import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import psutil

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试基础设施
提供统一的性能测试框架和工具
"""


logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    data_count: int
    success: bool
    error_message: Optional[str] = None

    @property
    def throughput(self) -> float:
        """计算吞吐量（记录/秒）"""
        return self.data_count / self.execution_time if self.execution_time > 0 else 0

@dataclass
class PerformanceThresholds:
    """性能阈值配置"""
    max_execution_time: float
    max_memory_usage_mb: float
    min_throughput: float
    max_avg_time_per_item: float

class PerformanceTestConfig:
    """性能测试配置管理"""

    DEFAULT_THRESHOLDS = {
        'single_factor': PerformanceThresholds(
            max_execution_time=5.0,
            max_memory_usage_mb=200.0,
            min_throughput=100.0,
            max_avg_time_per_item=1.0
        ),
        'multi_factor': PerformanceThresholds(
            max_execution_time=30.0,
            max_memory_usage_mb=500.0,
            min_throughput=50.0,
            max_avg_time_per_item=5.0
        ),
        'concurrent': PerformanceThresholds(
            max_execution_time=20.0,
            max_memory_usage_mb=300.0,
            min_throughput=200.0,
            max_avg_time_per_item=2.0
        )
    }

    @classmethod
    def get_threshold(cls, test_type: str) -> PerformanceThresholds:
        """获取指定测试类型的性能阈值"""
        return cls.DEFAULT_THRESHOLDS.get(test_type, cls.DEFAULT_THRESHOLDS['single_factor'])

class PerformanceMonitor:
    """性能监控装饰器"""

    @staticmethod
    @contextmanager
    def monitor_performance(test_name: str):
        """性能监控上下文管理器"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        start_time = time.time()
        peak_memory = initial_memory

        try:
            yield lambda: process.memory_info().rss / 1024 / 1024
        finally:
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, final_memory)

            execution_time = end_time - start_time
            memory_usage = final_memory - initial_memory

            logger.info(f"{test_name} 性能指标: "
                       f"执行时间={execution_time:.2f}s, "
                       f"内存使用={memory_usage:.1f}MB, "
                       f"峰值内存={peak_memory:.1f}MB")

def performance_test(test_type: str = 'single_factor'):
    """性能测试装饰器"""
    def decorator(func):
        """方法描述"""
            test_name = func.__name__
            thresholds = PerformanceTestConfig.get_threshold(test_type)

            with PerformanceMonitor.monitor_performance(test_name):
                start_time = time.time()
                initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

                try:
                    result = func(*args, **kwargs)
                    success = True
                    error_message = None
                except Exception as e:
                    result = None
                    success = False
                    error_message = str(e)
                    logger.error(f"{test_name} 执行失败: {e}")
                    raise
                finally:
                    execution_time = time.time() - start_time
                    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    memory_usage = final_memory - initial_memory

                    # 性能断言
                    if success:
                        assert execution_time <= thresholds.max_execution_time, \
                            f"执行时间超限: {execution_time:.2f}s > {thresholds.max_execution_time}s"

                        assert memory_usage <= thresholds.max_memory_usage_mb, \
                            f"内存使用超限: {memory_usage:.1f}MB > {thresholds.max_memory_usage_mb}MB"

                return result
        return wrapper
    return decorator

class BasePerformanceTest(ABC):
    """性能测试基类 - 模板方法模式"""

    def __init__(self, engine, data_generator):
        """方法描述"""
        self.data_generator = data_generator
        self.metrics_history: List[PerformanceMetrics] = []

    def run_performance_test(self, test_name: str, test_params: Dict[str, Any]) -> PerformanceMetrics:
        """模板方法：执行性能测试的标准流程"""
        logger.info(f"开始性能测试: {test_name}")

        # 1. 准备测试数据
        test_data = self.prepare_test_data(test_params)

        # 2. 执行性能测试
        with PerformanceMonitor.monitor_performance(test_name) as get_memory:
            start_time = time.time()
            initial_memory = get_memory()

            try:
                result = self.execute_test(test_data, test_params)
                success = True
                error_message = None
            except Exception as e:
                result = None
                success = False
                error_message = str(e)
                logger.error(f"测试执行失败: {e}")
                raise
            finally:
                execution_time = time.time() - start_time
                final_memory = get_memory()
                peak_memory = max(initial_memory, final_memory)
                memory_usage = final_memory - initial_memory

        # 3. 创建性能指标
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=peak_memory,
            data_count=len(result) if result is not None else 0,
            success=success,
            error_message=error_message
        )

        # 4. 验证性能指标
        self.validate_performance(metrics, test_params)

        # 5. 记录历史
        self.metrics_history.append(metrics)

        return metrics

    @abstractmethod
    def prepare_test_data(self, test_params: Dict[str, Any]) -> Any:
        """准备测试数据 - 子类实现"""
        pass

    @abstractmethod
    def execute_test(self, test_data: Any, test_params: Dict[str, Any]) -> Any:
        """执行具体测试 - 子类实现"""
        pass

    def validate_performance(self, metrics: PerformanceMetrics, test_params: Dict[str, Any]):
        """验证性能指标"""
        test_type = test_params.get('test_type', 'single_factor')
        thresholds = PerformanceTestConfig.get_threshold(test_type)

        if not metrics.success:
            return

        assert metrics.execution_time <= thresholds.max_execution_time, \
            f"执行时间超限: {metrics.execution_time:.2f}s > {thresholds.max_execution_time}s"

        assert metrics.memory_usage_mb <= thresholds.max_memory_usage_mb, \
            f"内存使用超限: {metrics.memory_usage_mb:.1f}MB > {thresholds.max_memory_usage_mb}MB"

        if metrics.throughput > 0:
            assert metrics.throughput >= thresholds.min_throughput, \
                f"吞吐量过低: {metrics.throughput:.2f} < {thresholds.min_throughput}"