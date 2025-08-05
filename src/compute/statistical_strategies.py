from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计计算策略模式实现
提供可扩展的统计分析方法
"""


class StatisticalStrategy(ABC):
    """统计计算策略基类"""

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """执行统计计算"""
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        pass


class ICCalculationStrategy(StatisticalStrategy):
    """IC计算策略"""

    def __init__(self, min_samples: int = 10):
        """方法描述"""
        self.min_samples = min_samples

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证IC计算所需数据"""
        required_cols = ["factor_value", "return_rate"]
        return (
            not data.empty
            and all(col in data.columns for col in required_cols)
            and len(data.dropna(subset=required_cols)) >= self.min_samples
        )

    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """计算IC值"""
        if not self.validate_data(data):
            return {}

        valid_data = data.dropna(subset=["factor_value", "return_rate"])

        # Pearson相关系数
        ic_pearson, p_value_pearson = stats.pearsonr(valid_data["factor_value"], valid_data["return_rate"])

        # Spearman相关系数
        ic_spearman, p_value_spearman = stats.spearmanr(valid_data["factor_value"], valid_data["return_rate"])

        return {
            "ic_pearson": ic_pearson,
            "ic_spearman": ic_spearman,
            "p_value_pearson": p_value_pearson,
            "p_value_spearman": p_value_spearman,
            "sample_count": len(valid_data),
            "factor_mean": valid_data["factor_value"].mean(),
            "factor_std": valid_data["factor_value"].std(),
            "return_mean": valid_data["return_rate"].mean(),
            "return_std": valid_data["return_rate"].std(),
        }


class LayeredBacktestStrategy(StatisticalStrategy):
    """分层回测策略"""

    def __init__(self, n_layers: int = 10, min_stocks_per_layer: int = 2):
        """方法描述"""
        self.n_layers = n_layers
        self.min_stocks_per_layer = min_stocks_per_layer

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证分层回测所需数据"""
        required_cols = ["factor_value", "return_rate"]
        return (
            not data.empty
            and all(col in data.columns for col in required_cols)
            and len(data.dropna(subset=required_cols)) >= self.n_layers * self.min_stocks_per_layer
        )

    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """执行分层回测"""
        if not self.validate_data(data):
            return {}

        valid_data = data.dropna(subset=["factor_value", "return_rate"])

        # 按因子值分层
        try:
            valid_data["factor_quantile"] = pd.qcut(
                valid_data["factor_value"], q=self.n_layers, labels=range(1, self.n_layers + 1), duplicates="drop"
            )
        except ValueError as e:
            logger.warning(f"分层失败: {e}")
            return {}

        # 计算各层统计信息
        layer_stats = (
            valid_data.groupby("factor_quantile")["return_rate"].agg(["mean", "std", "count", "median"]).reset_index()
        )

        return {
            "layer_statistics": layer_stats.to_dict("records"),
            "total_samples": len(valid_data),
            "layers_count": len(layer_stats),
        }


class StatisticalAnalyzer:
    """统计分析器 - 使用策略模式"""

    def __init__(self):
        """方法描述"""
        self.strategies = {}

    def register_strategy(self, name: str, strategy: StatisticalStrategy):
        """注册统计策略"""
        self.strategies[name] = strategy

    def analyze(self, strategy_name: str, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """执行指定策略的分析"""
        if strategy_name not in self.strategies:
            raise ValueError(f"未知的策略: {strategy_name}")

        strategy = self.strategies[strategy_name]
        return strategy.calculate(data, **kwargs)
