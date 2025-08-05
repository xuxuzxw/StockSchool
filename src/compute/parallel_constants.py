from enum import Enum

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行计算常量定义
"""


class TimeoutConstants:
    """超时常量"""

    DEFAULT_WORKER_TIMEOUT = 300  # 5分钟
    INDIVIDUAL_TASK_TIMEOUT = 60  # 1分钟
    BATCH_TASK_TIMEOUT = 600  # 10分钟
    RESOURCE_CHECK_TIMEOUT = 5  # 5秒


class ResourceConstants:
    """资源常量"""

    MIN_MEMORY_MB = 1000  # 最小内存要求 1GB
    MAX_CPU_PERCENT = 80  # 最大CPU使用率
    MAX_MEMORY_PERCENT = 85  # 最大内存使用率
    MEMORY_PER_STOCK_MB = 10  # 每个股票预估内存使用

    # 垃圾回收阈值
    GC_THRESHOLD_MB = 500  # 内存使用超过500MB时触发GC
    GC_INTERVAL_SECONDS = 30  # GC检查间隔


class WorkerConstants:
    """工作进程常量"""

    DEFAULT_MAX_WORKERS = 8  # 默认最大工作进程数
    MIN_WORKERS = 1  # 最小工作进程数
    MAX_WORKERS_LIMIT = 32  # 工作进程数上限

    # 批处理常量
    DEFAULT_BATCH_SIZE = 50  # 默认批次大小
    MIN_BATCH_SIZE = 1  # 最小批次大小
    MAX_BATCH_SIZE = 200  # 最大批次大小


class ComplexityLevel(Enum):
    """任务复杂度级别"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CalculationMode(Enum):
    """计算模式"""

    BATCH = "batch"
    INDIVIDUAL = "individual"
    AUTO = "auto"  # 自动选择模式


# 复杂度因子映射
COMPLEXITY_FACTORS = {
    ComplexityLevel.LOW: 2.0,
    ComplexityLevel.MEDIUM: 1.5,
    ComplexityLevel.HIGH: 1.0,
    ComplexityLevel.VERY_HIGH: 0.5,
}

# 因子类型复杂度映射
FACTOR_TYPE_COMPLEXITY = {
    "technical": ComplexityLevel.LOW,
    "fundamental": ComplexityLevel.MEDIUM,
    "sentiment": ComplexityLevel.MEDIUM,
}
