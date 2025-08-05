from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行计算策略模式实现
"""

from .factor_models import FactorResult


class ParallelCalculationStrategy(ABC):
    """并行计算策略抽象基类"""

    @abstractmethod
    def execute(self, tasks: List[Any], max_workers: int, **kwargs) -> List[FactorResult]:
        """执行并行计算策略"""
        pass

    @abstractmethod
    def get_optimal_workers(self, task_count: int) -> int:
        """获取最优工作进程数"""
        pass


class BatchCalculationStrategy(ParallelCalculationStrategy):
    """批量计算策略"""

    def __init__(self, batch_size: int = None):
        """方法描述"""

    def execute(self, tasks: List[Any], max_workers: int, **kwargs) -> List[FactorResult]:
        """执行批量并行计算"""
        from .parallel_workers import BatchWorkerExecutor

        executor = BatchWorkerExecutor(max_workers)
        return executor.execute_batch(tasks, self.batch_size, **kwargs)

    def get_optimal_workers(self, task_count: int) -> int:
        """计算批量模式的最优工作进程数"""
        import os

        cpu_count = os.cpu_count() or 1
        return min(max(1, cpu_count), task_count // 10, 8)


class IndividualCalculationStrategy(ParallelCalculationStrategy):
    """单个任务计算策略"""

    def execute(self, tasks: List[Any], max_workers: int, **kwargs) -> List[FactorResult]:
        """执行单个任务并行计算"""
        from .parallel_workers import IndividualWorkerExecutor

        executor = IndividualWorkerExecutor(max_workers)
        return executor.execute_individual(tasks, **kwargs)

    def get_optimal_workers(self, task_count: int) -> int:
        """计算单任务模式的最优工作进程数"""
        import os

        cpu_count = os.cpu_count() or 1
        return min(max(1, cpu_count * 2), task_count, 16)


class StrategyFactory:
    """策略工厂"""

    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> ParallelCalculationStrategy:
        """创建计算策略"""
        strategies = {"batch": BatchCalculationStrategy, "individual": IndividualCalculationStrategy}

        strategy_class = strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"不支持的策略类型: {strategy_type}")

        return strategy_class(**kwargs)
