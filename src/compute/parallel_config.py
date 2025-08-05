import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行计算配置管理
"""



@dataclass
class ResourceThresholds:
    """资源阈值配置"""
    min_memory_mb: float = 1000.0
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    disk_warning_percent: float = 90.0


@dataclass
class WorkerConfig:
    """工作进程配置"""
    max_workers: Optional[int] = None
    default_timeout_seconds: int = 300
    individual_timeout_seconds: int = 60
    batch_timeout_seconds: int = 600

    def __post_init__(self):
        """方法描述"""
            self.max_workers = min(32, (os.cpu_count() or 1) + 4)


@dataclass
class BatchConfig:
    """批处理配置"""
    default_batch_size: int = 50
    max_batch_size: int = 200
    min_batch_size: int = 1
    memory_per_stock_mb: float = 10.0

    def calculate_optimal_batch_size(self, total_stocks: int,
                                   available_memory_mb: float) -> int:
        """计算最优批次大小"""
        memory_based_size = int(available_memory_mb / self.memory_per_stock_mb)
        cpu_based_size = total_stocks // (os.cpu_count() or 1)

        optimal_size = min(
            memory_based_size,
            cpu_based_size,
            self.max_batch_size
        )

        return max(self.min_batch_size, optimal_size)


@dataclass
class PerformanceConfig:
    """性能配置"""
    enable_gc_optimization: bool = True
    gc_threshold_mb: float = 500.0
    enable_memory_monitoring: bool = True
    monitoring_interval_seconds: int = 30

    # 复杂度因子
    complexity_factors: Dict[str, float] = field(default_factory=lambda: {
        'low': 2.0,
        'medium': 1.5,
        'high': 1.0,
        'very_high': 0.5
    })


@dataclass
class ParallelCalculationConfig:
    """并行计算总配置"""
    resource_thresholds: ResourceThresholds = field(default_factory=ResourceThresholds)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)

    # 数据库配置
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_recycle: int = 3600

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ParallelCalculationConfig':
        """从字典创建配置"""
        return cls(**config_dict)

    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证资源阈值
            assert 0 < self.resource_thresholds.min_memory_mb < 10000
            assert 0 < self.resource_thresholds.max_cpu_percent <= 100

            # 验证工作进程配置
            assert self.worker_config.max_workers > 0
            assert self.worker_config.default_timeout_seconds > 0

            # 验证批处理配置
            assert self.batch_config.min_batch_size <= self.batch_config.max_batch_size
            assert self.batch_config.memory_per_stock_mb > 0

            return True

        except AssertionError as e:
            logger.error(f"配置验证失败: {e}")
            return False


# 默认配置实例
DEFAULT_PARALLEL_CONFIG = ParallelCalculationget_config()