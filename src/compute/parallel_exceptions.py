from typing import Any, List, Optional

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行计算异常定义
"""


class ParallelCalculationError(Exception):
    """并行计算基础异常"""

    def __init__(self, message: str, context: Optional[dict] = None):
        """方法描述"""
        super().__init__(message)


class ResourceInsufficientError(ParallelCalculationError):
    """资源不足异常"""

    def __init__(self, resource_type: str, required: float, available: float):
        """方法描述"""
        self.required = required
        self.available = available

        message = f"{resource_type}资源不足: 需要{required}, 可用{available}"
        super().__init__(message, {"resource_type": resource_type, "required": required, "available": available})


class WorkerExecutionError(ParallelCalculationError):
    """工作进程执行异常"""

    def __init__(self, worker_id: str, task_info: dict, original_error: Exception):
        """方法描述"""
        self.task_info = task_info
        self.original_error = original_error

        message = f"工作进程 {worker_id} 执行失败: {original_error}"
        super().__init__(
            message, {"worker_id": worker_id, "task_info": task_info, "original_error": str(original_error)}
        )


class TaskTimeoutError(ParallelCalculationError):
    """任务超时异常"""

    def __init__(self, task_id: str, timeout_seconds: int):
        """方法描述"""
        self.timeout_seconds = timeout_seconds

        message = f"任务 {task_id} 执行超时: {timeout_seconds}秒"
        super().__init__(message, {"task_id": task_id, "timeout_seconds": timeout_seconds})


class BatchProcessingError(ParallelCalculationError):
    """批处理异常"""

    def __init__(self, batch_id: str, failed_items: List[str], total_items: int, error_details: dict):
        self.batch_id = batch_id
        self.failed_items = failed_items
        self.total_items = total_items
        self.error_details = error_details

        message = f"批次 {batch_id} 处理失败: " f"{len(failed_items)}/{total_items} 项失败"
        super().__init__(
            message,
            {
                "batch_id": batch_id,
                "failed_items": failed_items,
                "total_items": total_items,
                "error_details": error_details,
            },
        )


class ConfigurationError(ParallelCalculationError):
    """配置错误异常"""

    def __init__(self, config_key: str, config_value: Any, reason: str):
        """方法描述"""
        self.config_value = config_value
        self.reason = reason

        message = f"配置错误 {config_key}={config_value}: {reason}"
        super().__init__(message, {"config_key": config_key, "config_value": config_value, "reason": reason})


class EngineCreationError(ParallelCalculationError):
    """引擎创建异常"""

    def __init__(self, factor_type: str, database_url: str, original_error: Exception):
        """方法描述"""
        self.database_url = database_url
        self.original_error = original_error

        message = f"创建 {factor_type} 引擎失败: {original_error}"
        super().__init__(
            message, {"factor_type": factor_type, "database_url": database_url, "original_error": str(original_error)}
        )
