import functools
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据验证装饰器
提供统一的数据验证机制
"""


from .factor_exceptions import DataValidationError, InsufficientDataError, MissingDataColumnError


def validate_dataframe_input(
    required_columns: Optional[List[str]] = None, min_length: Optional[int] = None, allow_empty: bool = False
):
    """
    DataFrame输入验证装饰器

    Args:
        required_columns: 必需的列名列表
        min_length: 最小数据长度
        allow_empty: 是否允许空DataFrame
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        def wrapper(*args, **kwargs):
            """方法描述"""
            dataframes = []
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    dataframes.append(arg)

            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    dataframes.append(value)

            # 验证每个DataFrame
            for df in dataframes:
                _validate_single_dataframe(df, required_columns, min_length, allow_empty, func.__name__)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _validate_single_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]],
    min_length: Optional[int],
    allow_empty: bool,
    func_name: str,
):
    """验证单个DataFrame"""

    # 检查是否为空
    if df.empty and not allow_empty:
        raise DataValidationError(f"{func_name}: DataFrame不能为空")

    # 检查必需列
    if required_columns and not df.empty:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise MissingDataColumnError(missing_columns, func_name)

    # 检查最小长度
    if min_length and len(df) < min_length:
        raise InsufficientDataError(min_length, len(df), func_name)


def validate_factor_return_data(
    factor_col: str = "factor_value", return_col: str = "return_rate", min_samples: int = 10
):
    """
    因子和收益率数据验证装饰器

    Args:
        factor_col: 因子值列名
        return_col: 收益率列名
        min_samples: 最小样本数
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        def wrapper(*args, **kwargs):
            """方法描述"""
            if len(args) >= 2:
                factor_data, return_data = args[0], args[1]

                # 验证因子数据
                if factor_data.empty:
                    logger.warning(f"{func.__name__}: 因子数据为空")
                    return pd.DataFrame() if "DataFrame" in str(func.__annotations__.get("return", "")) else {}

                # 验证收益率数据
                if return_data.empty:
                    logger.warning(f"{func.__name__}: 收益率数据为空")
                    return pd.DataFrame() if "DataFrame" in str(func.__annotations__.get("return", "")) else {}

                # 检查必需列
                if factor_col not in factor_data.columns:
                    raise MissingDataColumnError([factor_col], func.__name__)

                if return_col not in return_data.columns:
                    raise MissingDataColumnError([return_col], func.__name__)

                # 合并数据检查样本数
                merged = pd.merge(factor_data, return_data, on=["ts_code", "trade_date"], how="inner")

                valid_samples = merged.dropna(subset=[factor_col, return_col])
                if len(valid_samples) < min_samples:
                    raise InsufficientDataError(min_samples, len(valid_samples), func.__name__)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def handle_analysis_errors(default_return_value=None):
    """
    分析错误处理装饰器

    Args:
        default_return_value: 发生错误时的默认返回值
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        def wrapper(*args, **kwargs):
            """方法描述"""
            try:
                return func(*args, **kwargs)

            except DataValidationError as e:
                logger.error(f"{func.__name__} 数据验证失败: {e}")
                return default_return_value or pd.DataFrame()

            except InsufficientDataError as e:
                logger.warning(f"{func.__name__} 数据不足: {e}")
                return default_return_value or pd.DataFrame()

            except MissingDataColumnError as e:
                logger.error(f"{func.__name__} 缺少必要列: {e}")
                return default_return_value or pd.DataFrame()

            except Exception as e:
                logger.error(f"{func.__name__} 执行失败: {e}", exc_info=True)
                return default_return_value or pd.DataFrame()

        return wrapper

    return decorator


def log_analysis_performance(include_memory: bool = False):
    """
    分析性能日志装饰器

    Args:
        include_memory: 是否包含内存使用信息
    """

    def decorator(func: Callable) -> Callable:
        """方法描述"""

        def wrapper(*args, **kwargs):
            """方法描述"""
            import time

            start_time = time.time()

            if include_memory:
                import os
                import psutil

                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB

            try:
                result = func(*args, **kwargs)

                execution_time = time.time() - start_time

                log_msg = f"{func.__name__} 执行完成，耗时: {execution_time:.3f}秒"

                if include_memory:
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = memory_after - memory_before
                    log_msg += f"，内存增长: {memory_increase:.1f}MB"

                logger.info(log_msg)

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} 执行失败，耗时: {execution_time:.3f}秒，错误: {e}")
                raise

        return wrapper

    return decorator
