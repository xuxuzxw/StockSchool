from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽象因子计算器
定义因子计算的通用模板和接口
"""


from .data_validation_mixin import DataValidationError, DataValidationMixin
from .factor_calculation_config import TechnicalFactorCalculationConfig


class AbstractFactorCalculator(ABC, DataValidationMixin):
    """
    抽象因子计算器基类

    使用模板方法模式定义计算流程：
    1. 预处理和验证
    2. 执行计算
    3. 后处理和验证
    4. 返回结果
    """

    def __init__(self, config: Optional[TechnicalFactorCalculationConfig] = None):
        """
        初始化计算器

        Args:
            config: 计算配置，如果为None则使用默认配置
        """
        self.config = config or TechnicalFactorCalculationConfig()
        self._calculation_cache: Dict[str, Any] = {}

    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        模板方法：执行因子计算的完整流程

        Args:
            data: 输入数据
            **kwargs: 额外参数

        Returns:
            因子计算结果字典

        Raises:
            DataValidationError: 数据验证失败
            FactorCalculationError: 计算过程出错
        """
        start_time = datetime.now()

        try:
            # 1. 预处理和验证
            self._preprocess_and_validate(data, **kwargs)

            # 2. 执行具体计算
            results = self._execute_calculation(data, **kwargs)

            # 3. 后处理和验证
            results = self._postprocess_and_validate(results, data, **kwargs)

            # 4. 记录计算信息
            execution_time = (datetime.now() - start_time).total_seconds()
            self._log_calculation_info(results, execution_time)

            return results

        except Exception as e:
            logger.error(f"{self.__class__.__name__} 计算失败: {e}")
            raise

    def _preprocess_and_validate(self, data: pd.DataFrame, **kwargs) -> None:
        """
        预处理和验证数据

        Args:
            data: 输入数据
            **kwargs: 额外参数
        """
        # 基本数据验证
        required_columns = self._get_required_columns()
        self.validate_required_columns(data, required_columns)

        # 数据长度验证
        min_length = self._get_minimum_data_length()
        if min_length > 0:
            self.validate_data_length(data, min_length, self.__class__.__name__)

        # 数值列验证
        for col in required_columns:
            self.validate_numeric_column(data, col)

        # 记录数据质量信息
        if self.config.enable_data_quality_check:
            self.log_data_quality_info(data, self.__class__.__name__)

        # 子类特定的验证
        self._validate_specific_requirements(data, **kwargs)

    def _postprocess_and_validate(
        self, results: Dict[str, pd.Series], data: pd.DataFrame, **kwargs
    ) -> Dict[str, pd.Series]:
        """
        后处理和验证结果

        Args:
            results: 计算结果
            data: 原始数据
            **kwargs: 额外参数

        Returns:
            处理后的结果
        """
        # 验证结果完整性
        self._validate_calculation_results(results, data)

        # 处理异常值
        results = self._handle_outliers(results)

        # 子类特定的后处理
        results = self._postprocess_specific(results, data, **kwargs)

        return results

    @abstractmethod
    def _execute_calculation(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        执行具体的因子计算（子类实现）

        Args:
            data: 输入数据
            **kwargs: 额外参数

        Returns:
            计算结果字典
        """
        pass

    @abstractmethod
    def _get_required_columns(self) -> List[str]:
        """
        获取计算所需的数据列（子类实现）

        Returns:
            必需的列名列表
        """
        pass

    def _get_minimum_data_length(self) -> int:
        """
        获取最小数据长度要求

        Returns:
            最小数据长度，0表示无要求
        """
        return 0

    def _validate_specific_requirements(self, data: pd.DataFrame, **kwargs) -> None:
        """
        验证子类特定的要求（子类可重写）

        Args:
            data: 输入数据
            **kwargs: 额外参数
        """
        pass

    def _postprocess_specific(
        self, results: Dict[str, pd.Series], data: pd.DataFrame, **kwargs
    ) -> Dict[str, pd.Series]:
        """
        子类特定的后处理（子类可重写）

        Args:
            results: 计算结果
            data: 原始数据
            **kwargs: 额外参数

        Returns:
            处理后的结果
        """
        return results

    def _validate_calculation_results(self, results: Dict[str, pd.Series], data: pd.DataFrame) -> None:
        """
        验证计算结果的有效性

        Args:
            results: 计算结果
            data: 原始数据
        """
        if not results:
            raise DataValidationError("计算结果为空")

        data_length = len(data)
        for factor_name, factor_series in results.items():
            if len(factor_series) != data_length:
                raise DataValidationError(
                    f"因子 {factor_name} 的结果长度 {len(factor_series)} " f"与输入数据长度 {data_length} 不匹配"
                )

    def _handle_outliers(self, results: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        处理异常值（使用3σ原则）

        Args:
            results: 计算结果

        Returns:
            处理后的结果
        """
        processed_results = {}

        for factor_name, factor_series in results.items():
            # 计算3σ边界
            mean_val = factor_series.mean()
            std_val = factor_series.std()

            if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val

                # 标记异常值但不直接删除，而是记录日志
                outliers = (factor_series < lower_bound) | (factor_series > upper_bound)
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    logger.warning(f"因子 {factor_name} 发现 {outlier_count} 个异常值")

            processed_results[factor_name] = factor_series

        return processed_results

    def _log_calculation_info(self, results: Dict[str, pd.Series], execution_time: float) -> None:
        """
        记录计算信息

        Args:
            results: 计算结果
            execution_time: 执行时间（秒）
        """
        factor_count = len(results)
        total_values = sum(len(series) for series in results.values())

        logger.info(f"{self.__class__.__name__} 计算完成:")
        logger.info(f"  因子数量: {factor_count}")
        logger.info(f"  总数据点: {total_values}")
        logger.info(f"  执行时间: {execution_time:.3f}秒")

    def clear_cache(self) -> None:
        """清空计算缓存"""
        self._calculation_cache.clear()
        logger.debug(f"{self.__class__.__name__} 缓存已清空")
