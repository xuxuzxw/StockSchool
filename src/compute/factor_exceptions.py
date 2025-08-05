from typing import Any, Optional

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算相关异常定义
提供层次化的异常处理机制
"""


class FactorCalculationError(Exception):
    """因子计算基础异常"""

    def __init__(self, message: str, factor_name: Optional[str] = None, original_error: Optional[Exception] = None):
        """
        初始化异常

        Args:
            message: 错误信息
            factor_name: 相关的因子名称
            original_error: 原始异常
        """
        self.factor_name = factor_name
        self.original_error = original_error

        if factor_name:
            message = f"[{factor_name}] {message}"

        super().__init__(message)


class InvalidFactorParameterError(FactorCalculationError):
    """无效因子参数异常"""

    def __init__(self, parameter_name: str, parameter_value: Any, message: str = "", factor_name: Optional[str] = None):
        """
        初始化参数异常

        Args:
            parameter_name: 参数名称
            parameter_value: 参数值
            message: 错误信息
            factor_name: 因子名称
        """
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value

        if not message:
            message = f"参数 {parameter_name} 的值 {parameter_value} 无效"

        super().__init__(message, factor_name)


class DataValidationError(FactorCalculationError):
    """数据验证异常"""

    pass


class InsufficientDataError(DataValidationError):
    """数据不足异常"""

    def __init__(self, required_length: int, actual_length: int, factor_name: Optional[str] = None):
        """
        初始化数据不足异常

        Args:
            required_length: 需要的数据长度
            actual_length: 实际数据长度
            factor_name: 因子名称
        """
        self.required_length = required_length
        self.actual_length = actual_length

        message = f"数据长度不足，需要 {required_length} 条，实际 {actual_length} 条"
        super().__init__(message, factor_name)


class MissingDataColumnError(DataValidationError):
    """缺少数据列异常"""

    def __init__(self, missing_columns: list, factor_name: Optional[str] = None):
        """
        初始化缺少列异常

        Args:
            missing_columns: 缺少的列名列表
            factor_name: 因子名称
        """
        self.missing_columns = missing_columns

        message = f"缺少必要的数据列: {missing_columns}"
        super().__init__(message, factor_name)


class CalculationTimeoutError(FactorCalculationError):
    """计算超时异常"""

    def __init__(self, timeout_seconds: float, factor_name: Optional[str] = None):
        """
        初始化超时异常

        Args:
            timeout_seconds: 超时时间（秒）
            factor_name: 因子名称
        """
        self.timeout_seconds = timeout_seconds

        message = f"计算超时，超过 {timeout_seconds} 秒"
        super().__init__(message, factor_name)


class FactorNotFoundError(FactorCalculationError):
    """因子未找到异常"""

    def __init__(self, factor_name: str):
        """
        初始化因子未找到异常

        Args:
            factor_name: 因子名称
        """
        message = f"未找到因子: {factor_name}"
        super().__init__(message, factor_name)


class ConfigurationError(FactorCalculationError):
    """配置错误异常"""

    def __init__(self, config_key: str, message: str = ""):
        """
        初始化配置错误异常

        Args:
            config_key: 配置键名
            message: 错误信息
        """
        self.config_key = config_key

        if not message:
            message = f"配置项 {config_key} 错误"

        super().__init__(message)
