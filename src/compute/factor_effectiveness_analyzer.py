import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sqlalchemy import text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子有效性检验分析器
实现因子IC、IR、分层回测、衰减分析等有效性检验功能

重构后的版本：
- 使用模板方法模式标准化分析流程
- 应用策略模式实现可扩展的统计计算
- 集中配置管理和常量定义
- 统一的数据验证和错误处理
- 性能优化和内存管理
"""


from .effectiveness_analysis_template import EffectivenessAnalysisTemplate
from .effectiveness_config import (
    DEFAULT_EFFECTIVENESS_CONFIG,
    AnalysisType,
    EffectivenessAnalysisConfig,
    StatisticalConstants,
    get_analysis_config,
)
from .factor_exceptions import DataValidationError, InsufficientDataError
from .factor_models import FactorCategory, FactorType
from .performance_decorators import cached_calculation, performance_optimized
from .statistical_strategies import ICCalculationStrategy, LayeredBacktestStrategy, StatisticalAnalyzer
from .validation_decorators import (
    handle_analysis_errors,
    log_analysis_performance,
    validate_dataframe_input,
    validate_factor_return_data,
)


class ReturnPeriod(Enum):
    """收益率周期枚举"""

    DAILY = "1d"
    WEEKLY = "5d"
    MONTHLY = "20d"
    QUARTERLY = "60d"


class FactorEffectivenessAnalyzer:
    """
    因子有效性分析器

    重构后的版本，采用组合模式和策略模式：
    - 使用配置类管理参数
    - 使用策略模式处理不同分析类型
    - 统一的错误处理和日志记录
    - 支持并行计算和缓存
    """

    def __init__(self, engine, config: Optional[EffectivenessAnalysisConfig] = None):
        """
        初始化因子有效性分析器

        Args:
            engine: 数据库引擎
            config: 分析配置，如果为None则使用默认配置
        """
        self.engine = engine
        self.config = config or DEFAULT_EFFECTIVENESS_CONFIG

        # 初始化统计分析器
        self.statistical_analyzer = StatisticalAnalyzer()
        self._register_strategies()

        # 初始化缓存
        if self.config.enable_caching:
            from .performance_decorators import _global_cache

            self.cache = _global_cache
        else:
            self.cache = None

    def _register_strategies(self):
        """注册统计分析策略"""
        # IC计算策略
        ic_strategy = ICCalculationStrategy(min_samples=self.config.ic_config.min_samples)
        self.statistical_analyzer.register_strategy("ic_calculation", ic_strategy)

        # 分层回测策略
        layered_strategy = LayeredBacktestStrategy(
            n_layers=self.config.layered_config.n_layers,
            min_stocks_per_layer=self.config.layered_config.min_stocks_per_layer,
        )
        self.statistical_analyzer.register_strategy("layered_backtest", layered_strategy)

    @cached_calculation(ttl_seconds=3600)
    @log_analysis_performance(include_memory=True)
    def get_return_data(
        self, ts_codes: List[str], start_date: date, end_date: date, return_period: ReturnPeriod = ReturnPeriod.DAILY
    ) -> pd.DataFrame:
        """
        获取股票收益率数据

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            return_period: 收益率周期

        Returns:
            包含收益率的DataFrame
        """
        try:
            # 安全的参数化查询 - 避免SQL注入
            if not ts_codes:
                return pd.DataFrame()

            # 使用参数化查询替代字符串拼接
            ts_codes_param = tuple(ts_codes)

            # 根据周期确定计算方法
            if return_period == ReturnPeriod.DAILY:
                period_days = 1
            elif return_period == ReturnPeriod.WEEKLY:
                period_days = 5
            elif return_period == ReturnPeriod.MONTHLY:
                period_days = 20
            elif return_period == ReturnPeriod.QUARTERLY:
                period_days = 60
            else:
                period_days = 1

            query = text(
                f"""
                WITH price_data AS (
                    SELECT ts_code, trade_date, close,
                           LAG(close, {period_days}) OVER (
                               PARTITION BY ts_code ORDER BY trade_date
                           ) as prev_close
                    FROM stock_daily
                    WHERE ts_code IN ({placeholders})
                    AND trade_date BETWEEN :start_date AND :end_date
                    ORDER BY ts_code, trade_date
                )
                SELECT ts_code, trade_date, close, prev_close,
                       CASE
                           WHEN prev_close IS NOT NULL AND prev_close > 0
                           THEN (close - prev_close) / prev_close
                           ELSE NULL
                       END as return_rate
                FROM price_data
                WHERE prev_close IS NOT NULL
            """
            )

            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(query, {"start_date": start_date, "end_date": end_date})

                return_data = pd.DataFrame(result.fetchall(), columns=result.keys())

            if not return_data.empty:
                return_data["trade_date"] = pd.to_datetime(return_data["trade_date"])
                return_data["return_rate"] = pd.to_numeric(return_data["return_rate"], errors="coerce")

            return return_data

        except Exception as e:
            logger.error(f"获取收益率数据失败: {e}")
            return pd.DataFrame()

    def calculate_ic(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
        factor_col: str = "factor_value",
        return_col: str = "return_rate",
        date_col: str = "trade_date",
        ts_code_col: str = "ts_code",
    ) -> pd.DataFrame:
        """
        计算信息系数(IC)

        Args:
            factor_data: 因子数据
            return_data: 收益率数据
            factor_col: 因子值列名
            return_col: 收益率列名
            date_col: 日期列名
            ts_code_col: 股票代码列名

        Returns:
            包含IC值的DataFrame
        """
        if factor_data.empty or return_data.empty:
            return pd.DataFrame()

        try:
            # 合并因子数据和收益率数据
            merged_data = pd.merge(factor_data, return_data, on=[ts_code_col, date_col], how="inner")

            if merged_data.empty:
                logger.warning("因子数据和收益率数据无法匹配")
                return pd.DataFrame()

            # 按日期分组计算IC
            ic_results = []

            for trade_date, group in merged_data.groupby(date_col):
                # 过滤有效数据
                valid_data = group.dropna(subset=[factor_col, return_col])

                if len(valid_data) < 10:  # 至少需要10个有效样本
                    continue

                # 计算Pearson相关系数作为IC
                ic_pearson, p_value_pearson = stats.pearsonr(valid_data[factor_col], valid_data[return_col])

                # 计算Spearman相关系数作为Rank IC
                ic_spearman, p_value_spearman = stats.spearmanr(valid_data[factor_col], valid_data[return_col])

                ic_results.append(
                    {
                        "trade_date": trade_date,
                        "ic_pearson": ic_pearson,
                        "ic_spearman": ic_spearman,
                        "p_value_pearson": p_value_pearson,
                        "p_value_spearman": p_value_spearman,
                        "sample_count": len(valid_data),
                        "factor_mean": valid_data[factor_col].mean(),
                        "factor_std": valid_data[factor_col].std(),
                        "return_mean": valid_data[return_col].mean(),
                        "return_std": valid_data[return_col].std(),
                    }
                )

            ic_df = pd.DataFrame(ic_results)

            if not ic_df.empty:
                ic_df["trade_date"] = pd.to_datetime(ic_df["trade_date"])
                ic_df = ic_df.sort_values("trade_date")

            return ic_df

        except Exception as e:
            logger.error(f"计算IC失败: {e}")
            return pd.DataFrame()

    def calculate_ic_statistics(self, ic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算IC统计指标

        Args:
            ic_data: IC数据

        Returns:
            IC统计指标字典
        """
        if ic_data.empty:
            return {}

        try:
            stats_result = {}

            # Pearson IC统计
            if "ic_pearson" in ic_data.columns:
                ic_pearson = ic_data["ic_pearson"].dropna()
                if not ic_pearson.empty:
                    stats_result["ic_pearson"] = {
                        "mean": ic_pearson.mean(),
                        "std": ic_pearson.std(),
                        "abs_mean": ic_pearson.abs().mean(),
                        "positive_ratio": (ic_pearson > 0).mean(),
                        "significant_ratio": (ic_data["p_value_pearson"] < 0.05).mean(),
                        "max": ic_pearson.max(),
                        "min": ic_pearson.min(),
                        "skewness": ic_pearson.skew(),
                        "kurtosis": ic_pearson.kurtosis(),
                    }

            # Spearman IC统计
            if "ic_spearman" in ic_data.columns:
                ic_spearman = ic_data["ic_spearman"].dropna()
                if not ic_spearman.empty:
                    stats_result["ic_spearman"] = {
                        "mean": ic_spearman.mean(),
                        "std": ic_spearman.std(),
                        "abs_mean": ic_spearman.abs().mean(),
                        "positive_ratio": (ic_spearman > 0).mean(),
                        "significant_ratio": (ic_data["p_value_spearman"] < 0.05).mean(),
                        "max": ic_spearman.max(),
                        "min": ic_spearman.min(),
                        "skewness": ic_spearman.skew(),
                        "kurtosis": ic_spearman.kurtosis(),
                    }

            # 整体统计
            stats_result["overall"] = {
                "total_periods": len(ic_data),
                "avg_sample_count": ic_data["sample_count"].mean(),
                "data_coverage": ic_data["sample_count"].sum() / len(ic_data) if len(ic_data) > 0 else 0,
            }

            return stats_result

        except Exception as e:
            logger.error(f"计算IC统计指标失败: {e}")
            return {}

    def calculate_ir(self, ic_data: pd.DataFrame, ic_col: str = "ic_pearson", window: int = 20) -> pd.DataFrame:
        """
        计算信息比率(IR)

        Args:
            ic_data: IC数据
            ic_col: IC列名
            window: 滚动窗口期

        Returns:
            包含IR值的DataFrame
        """
        if ic_data.empty or ic_col not in ic_data.columns:
            return pd.DataFrame()

        try:
            ir_data = ic_data.copy()

            # 计算滚动IC均值和标准差
            ic_series = ir_data[ic_col]

            # 滚动均值
            ir_data["ic_mean"] = ic_series.rolling(window=window).mean()

            # 滚动标准差
            ir_data["ic_std"] = ic_series.rolling(window=window).std()

            # 计算IR = IC均值 / IC标准差
            ir_data["ir"] = ir_data["ic_mean"] / ir_data["ic_std"]

            # 计算累积IC
            ir_data["ic_cumsum"] = ic_series.cumsum()

            # 计算IC的t统计量
            ir_data["ic_t_stat"] = ir_data["ic_mean"] / (ir_data["ic_std"] / np.sqrt(window))

            # 计算IR的置信区间（95%）
            t_critical = stats.t.ppf(0.975, window - 1)  # 95%置信区间
            ir_data["ir_ci_lower"] = ir_data["ir"] - t_critical * (1 / np.sqrt(window))
            ir_data["ir_ci_upper"] = ir_data["ir"] + t_critical * (1 / np.sqrt(window))

            return ir_data

        except Exception as e:
            logger.error(f"计算IR失败: {e}")
            return pd.DataFrame()

    def calculate_ir_statistics(self, ir_data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算IR统计指标

        Args:
            ir_data: IR数据

        Returns:
            IR统计指标字典
        """
        if ir_data.empty or "ir" not in ir_data.columns:
            return {}

        try:
            ir_series = ir_data["ir"].dropna()

            if ir_series.empty:
                return {}

            stats_result = {
                "ir_mean": ir_series.mean(),
                "ir_std": ir_series.std(),
                "ir_max": ir_series.max(),
                "ir_min": ir_series.min(),
                "ir_positive_ratio": (ir_series > 0).mean(),
                "ir_significant_ratio": (ir_series.abs() > 1.96).mean(),  # 95%显著性
                "ir_stability": 1 - (ir_series.std() / ir_series.abs().mean()) if ir_series.abs().mean() > 0 else 0,
            }

            # 最终IR值（最新的IR）
            if not ir_series.empty:
                stats_result["final_ir"] = ir_series.iloc[-1]

            # IC累积统计
            if "ic_cumsum" in ir_data.columns:
                ic_cumsum = ir_data["ic_cumsum"].dropna()
                if not ic_cumsum.empty:
                    stats_result["ic_cumsum_final"] = ic_cumsum.iloc[-1]
                    stats_result["ic_cumsum_max"] = ic_cumsum.max()
                    stats_result["ic_cumsum_min"] = ic_cumsum.min()

            return stats_result

        except Exception as e:
            logger.error(f"计算IR统计指标失败: {e}")
            return {}

    def factor_layered_backtest(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
        n_layers: int = 10,
        factor_col: str = "factor_value",
        return_col: str = "return_rate",
        date_col: str = "trade_date",
        ts_code_col: str = "ts_code",
    ) -> pd.DataFrame:
        """
        因子分层回测

        Args:
            factor_data: 因子数据
            return_data: 收益率数据
            n_layers: 分层数量
            factor_col: 因子值列名
            return_col: 收益率列名
            date_col: 日期列名
            ts_code_col: 股票代码列名

        Returns:
            分层回测结果
        """
        if factor_data.empty or return_data.empty:
            return pd.DataFrame()

        try:
            # 合并数据
            merged_data = pd.merge(factor_data, return_data, on=[ts_code_col, date_col], how="inner")

            if merged_data.empty:
                return pd.DataFrame()

            # 按日期分组进行分层
            layered_results = []

            for trade_date, group in merged_data.groupby(date_col):
                # 过滤有效数据
                valid_data = group.dropna(subset=[factor_col, return_col])

                if len(valid_data) < n_layers * 2:  # 每层至少需要2个股票
                    continue

                # 按因子值分层
                valid_data["factor_quantile"] = pd.qcut(
                    valid_data[factor_col], q=n_layers, labels=range(1, n_layers + 1), duplicates="drop"
                )

                # 计算各层收益率
                layer_stats = (
                    valid_data.groupby("factor_quantile")[return_col]
                    .agg(["mean", "std", "count", "median"])
                    .reset_index()
                )

                layer_stats["trade_date"] = trade_date
                layered_results.append(layer_stats)

            if not layered_results:
                return pd.DataFrame()

            # 合并所有日期的结果
            layered_df = pd.concat(layered_results, ignore_index=True)
            layered_df["trade_date"] = pd.to_datetime(layered_df["trade_date"])

            return layered_df

        except Exception as e:
            logger.error(f"因子分层回测失败: {e}")
            return pd.DataFrame()

    def analyze_layered_performance(self, layered_data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析分层回测表现

        Args:
            layered_data: 分层回测数据

        Returns:
            分层表现分析结果
        """
        if layered_data.empty:
            return {}

        try:
            analysis_result = {}

            # 按分层计算平均表现
            layer_performance = (
                layered_data.groupby("factor_quantile").agg({"mean": ["mean", "std"], "count": "mean"}).round(6)
            )

            layer_performance.columns = ["avg_return", "return_volatility", "avg_stock_count"]
            layer_performance = layer_performance.reset_index()

            analysis_result["layer_performance"] = layer_performance.to_dict("records")

            # 计算多空收益（最高层 - 最低层）
            if len(layer_performance) >= 2:
                top_layer = layer_performance.iloc[-1]
                bottom_layer = layer_performance.iloc[0]

                long_short_return = top_layer["avg_return"] - bottom_layer["avg_return"]
                long_short_volatility = np.sqrt(
                    top_layer["return_volatility"] ** 2 + bottom_layer["return_volatility"] ** 2
                )

                analysis_result["long_short"] = {
                    "return": long_short_return,
                    "volatility": long_short_volatility,
                    "sharpe_ratio": long_short_return / long_short_volatility if long_short_volatility > 0 else 0,
                    "top_layer_return": top_layer["avg_return"],
                    "bottom_layer_return": bottom_layer["avg_return"],
                }

            # 计算单调性指标
            returns_by_layer = layer_performance["avg_return"].values
            if len(returns_by_layer) > 2:
                # 计算Spearman相关系数检验单调性
                layer_ranks = np.arange(1, len(returns_by_layer) + 1)
                monotonicity, p_value = stats.spearmanr(layer_ranks, returns_by_layer)

                analysis_result["monotonicity"] = {
                    "correlation": monotonicity,
                    "p_value": p_value,
                    "is_significant": p_value < 0.05,
                    "is_monotonic": monotonicity > 0.5 and p_value < 0.05,
                }

            # 计算分层效果的统计显著性
            if len(layered_data) > 1:
                # 使用方差分析检验各层收益率差异的显著性
                layer_groups = [group["mean"].values for _, group in layered_data.groupby("factor_quantile")]

                if len(layer_groups) > 1 and all(len(group) > 1 for group in layer_groups):
                    f_stat, p_value_anova = stats.f_oneway(*layer_groups)

                    analysis_result["significance_test"] = {
                        "f_statistic": f_stat,
                        "p_value": p_value_anova,
                        "is_significant": p_value_anova < 0.05,
                    }

            return analysis_result

        except Exception as e:
            logger.error(f"分析分层表现失败: {e}")
            return {}

    def calculate_factor_decay(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
        max_periods: int = 20,
        factor_col: str = "factor_value",
        return_col: str = "return_rate",
        date_col: str = "trade_date",
        ts_code_col: str = "ts_code",
    ) -> pd.DataFrame:
        """
        计算因子衰减分析

        Args:
            factor_data: 因子数据
            return_data: 收益率数据
            max_periods: 最大预测期数
            factor_col: 因子值列名
            return_col: 收益率列名
            date_col: 日期列名
            ts_code_col: 股票代码列名

        Returns:
            因子衰减分析结果
        """
        if factor_data.empty or return_data.empty:
            return pd.DataFrame()

        try:
            decay_results = []

            # 为每个预测期计算IC
            for period in range(1, max_periods + 1):
                # 创建滞后的收益率数据
                return_shifted = return_data.copy()
                return_shifted["trade_date"] = pd.to_datetime(return_shifted["trade_date"])
                return_shifted = return_shifted.sort_values([ts_code_col, "trade_date"])

                # 按股票分组，将收益率向前移动period期
                shifted_returns = []
                for ts_code, group in return_shifted.groupby(ts_code_col):
                    group_shifted = group.copy()
                    group_shifted["trade_date"] = group_shifted["trade_date"] - pd.Timedelta(days=period)
                    shifted_returns.append(group_shifted)

                if shifted_returns:
                    return_shifted_all = pd.concat(shifted_returns, ignore_index=True)

                    # 计算该期的IC
                    ic_period = self.calculate_ic(
                        factor_data, return_shifted_all, factor_col, return_col, date_col, ts_code_col
                    )

                    if not ic_period.empty:
                        # 计算该期的平均IC
                        avg_ic_pearson = ic_period["ic_pearson"].mean()
                        avg_ic_spearman = ic_period["ic_spearman"].mean()
                        ic_std_pearson = ic_period["ic_pearson"].std()
                        ic_std_spearman = ic_period["ic_spearman"].std()

                        decay_results.append(
                            {
                                "period": period,
                                "avg_ic_pearson": avg_ic_pearson,
                                "avg_ic_spearman": avg_ic_spearman,
                                "ic_std_pearson": ic_std_pearson,
                                "ic_std_spearman": ic_std_spearman,
                                "abs_avg_ic_pearson": abs(avg_ic_pearson),
                                "abs_avg_ic_spearman": abs(avg_ic_spearman),
                                "sample_periods": len(ic_period),
                            }
                        )

            decay_df = pd.DataFrame(decay_results)

            return decay_df

        except Exception as e:
            logger.error(f"计算因子衰减失败: {e}")
            return pd.DataFrame()

    def analyze_factor_decay(self, decay_data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析因子衰减特征

        Args:
            decay_data: 因子衰减数据

        Returns:
            因子衰减分析结果
        """
        if decay_data.empty:
            return {}

        try:
            analysis_result = {}

            # 找到最优持有期（绝对IC最大的期数）
            if "abs_avg_ic_pearson" in decay_data.columns:
                optimal_period_pearson = decay_data.loc[decay_data["abs_avg_ic_pearson"].idxmax(), "period"]
                max_ic_pearson = decay_data["abs_avg_ic_pearson"].max()

                analysis_result["optimal_period_pearson"] = {
                    "period": optimal_period_pearson,
                    "max_abs_ic": max_ic_pearson,
                }

            if "abs_avg_ic_spearman" in decay_data.columns:
                optimal_period_spearman = decay_data.loc[decay_data["abs_avg_ic_spearman"].idxmax(), "period"]
                max_ic_spearman = decay_data["abs_avg_ic_spearman"].max()

                analysis_result["optimal_period_spearman"] = {
                    "period": optimal_period_spearman,
                    "max_abs_ic": max_ic_spearman,
                }

            # 计算衰减速度（IC衰减到初始值一半所需的期数）
            if len(decay_data) > 1:
                initial_ic = decay_data.iloc[0]["abs_avg_ic_pearson"]
                half_ic = initial_ic / 2

                # 找到第一个低于一半IC的期数
                half_life_periods = decay_data[decay_data["abs_avg_ic_pearson"] <= half_ic]

                if not half_life_periods.empty:
                    half_life = half_life_periods.iloc[0]["period"]
                    analysis_result["half_life"] = half_life
                else:
                    analysis_result["half_life"] = len(decay_data)  # 未衰减到一半

            # 衰减模式分类
            if len(decay_data) >= 3:
                ic_values = decay_data["abs_avg_ic_pearson"].values
                periods = decay_data["period"].values

                # 计算衰减趋势的相关系数
                decay_corr, p_value = stats.pearsonr(periods, ic_values)

                if p_value < 0.05:
                    if decay_corr < -0.5:
                        decay_pattern = "快速衰减"
                    elif decay_corr < -0.2:
                        decay_pattern = "缓慢衰减"
                    elif decay_corr > 0.2:
                        decay_pattern = "增强型"
                    else:
                        decay_pattern = "稳定型"
                else:
                    decay_pattern = "无明显趋势"

                analysis_result["decay_pattern"] = {
                    "pattern": decay_pattern,
                    "correlation": decay_corr,
                    "p_value": p_value,
                }

            # 计算衰减稳定性
            if "ic_std_pearson" in decay_data.columns:
                avg_ic_std = decay_data["ic_std_pearson"].mean()
                analysis_result["decay_stability"] = {
                    "avg_ic_volatility": avg_ic_std,
                    "stability_score": 1 / (1 + avg_ic_std) if avg_ic_std > 0 else 1,
                }

            return analysis_result

        except Exception as e:
            logger.error(f"分析因子衰减特征失败: {e}")
            return {}


class FactorCorrelationAnalyzer:
    """因子相关性分析器"""

    def __init__(self, engine):
        """初始化因子相关性分析器"""
        self.engine = engine

    def calculate_factor_correlation_matrix(
        self, factor_data: pd.DataFrame, factor_cols: List[str], method: str = "pearson"
    ) -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            factor_data: 因子数据
            factor_cols: 因子列名列表
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')

        Returns:
            因子相关性矩阵
        """
        if factor_data.empty or not factor_cols:
            return pd.DataFrame()

        try:
            # 检查因子列是否存在
            available_cols = [col for col in factor_cols if col in factor_data.columns]
            if len(available_cols) < 2:
                logger.warning("可用因子列少于2个，无法计算相关性矩阵")
                return pd.DataFrame()

            # 计算相关性矩阵
            correlation_matrix = factor_data[available_cols].corr(method=method)

            return correlation_matrix

        except Exception as e:
            logger.error(f"计算因子相关性矩阵失败: {e}")
            return pd.DataFrame()

    def analyze_correlation_changes(
        self,
        factor_data: pd.DataFrame,
        factor_cols: List[str],
        date_col: str = "trade_date",
        window: int = 60,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        分析因子相关性的时间变化

        Args:
            factor_data: 因子数据
            factor_cols: 因子列名列表
            date_col: 日期列名
            window: 滚动窗口期
            method: 相关性计算方法

        Returns:
            相关性时间变化数据
        """
        if factor_data.empty or len(factor_cols) < 2:
            return pd.DataFrame()

        try:
            # 按日期排序
            factor_data_sorted = factor_data.sort_values(date_col)

            # 滚动计算相关性
            correlation_changes = []

            for i in range(window, len(factor_data_sorted)):
                window_data = factor_data_sorted.iloc[i - window : i]

                # 计算该窗口的相关性矩阵
                corr_matrix = self.calculate_factor_correlation_matrix(window_data, factor_cols, method)

                if not corr_matrix.empty:
                    # 提取上三角矩阵的相关系数
                    for j, factor1 in enumerate(factor_cols):
                        for k, factor2 in enumerate(factor_cols):
                            if j < k and factor1 in corr_matrix.index and factor2 in corr_matrix.columns:
                                correlation_changes.append(
                                    {
                                        "trade_date": factor_data_sorted.iloc[i - 1][date_col],
                                        "factor1": factor1,
                                        "factor2": factor2,
                                        "correlation": corr_matrix.loc[factor1, factor2],
                                        "window_end_idx": i,
                                    }
                                )

            correlation_df = pd.DataFrame(correlation_changes)

            if not correlation_df.empty:
                correlation_df["trade_date"] = pd.to_datetime(correlation_df["trade_date"])

            return correlation_df

        except Exception as e:
            logger.error(f"分析因子相关性时间变化失败: {e}")
            return pd.DataFrame()

    def identify_high_correlation_factors(
        self, correlation_matrix: pd.DataFrame, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        识别高相关因子对

        Args:
            correlation_matrix: 相关性矩阵
            threshold: 高相关阈值

        Returns:
            高相关因子对列表
        """
        if correlation_matrix.empty:
            return []

        try:
            high_corr_pairs = []

            # 遍历上三角矩阵
            for i, factor1 in enumerate(correlation_matrix.index):
                for j, factor2 in enumerate(correlation_matrix.columns):
                    if i < j:  # 只考虑上三角
                        corr_value = correlation_matrix.iloc[i, j]

                        if abs(corr_value) >= threshold:
                            high_corr_pairs.append((factor1, factor2, corr_value))

            # 按相关系数绝对值降序排序
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            return high_corr_pairs

        except Exception as e:
            logger.error(f"识别高相关因子失败: {e}")
            return []

    def calculate_factor_independence(self, factor_data: pd.DataFrame, factor_cols: List[str]) -> Dict[str, float]:
        """
        计算因子独立性评估

        Args:
            factor_data: 因子数据
            factor_cols: 因子列名列表

        Returns:
            因子独立性评分字典
        """
        if factor_data.empty or len(factor_cols) < 2:
            return {}

        try:
            # 计算相关性矩阵
            corr_matrix = self.calculate_factor_correlation_matrix(factor_data, factor_cols)

            if corr_matrix.empty:
                return {}

            independence_scores = {}

            for factor in factor_cols:
                if factor in corr_matrix.index:
                    # 计算该因子与其他因子的平均绝对相关系数
                    other_factors = [f for f in factor_cols if f != factor and f in corr_matrix.columns]

                    if other_factors:
                        correlations = [abs(corr_matrix.loc[factor, other]) for other in other_factors]
                        avg_abs_corr = np.mean(correlations)

                        # 独立性评分 = 1 - 平均绝对相关系数
                        independence_scores[factor] = 1 - avg_abs_corr
                    else:
                        independence_scores[factor] = 1.0

            return independence_scores

        except Exception as e:
            logger.error(f"计算因子独立性失败: {e}")
            return {}


class FactorEffectivenessReporter:
    """因子有效性报告生成器"""

    def __init__(self, engine):
        """初始化因子有效性报告生成器"""
        self.engine = engine
        self.analyzer = FactorEffectivenessAnalyzer(engine)
        self.correlation_analyzer = FactorCorrelationAnalyzer(engine)

    def generate_comprehensive_report(
        self, factor_data: pd.DataFrame, return_data: pd.DataFrame, factor_name: str, factor_col: str = "factor_value"
    ) -> Dict[str, Any]:
        """
        生成因子有效性综合报告

        Args:
            factor_data: 因子数据
            return_data: 收益率数据
            factor_name: 因子名称
            factor_col: 因子值列名

        Returns:
            综合报告字典
        """
        report = {
            "factor_name": factor_name,
            "analysis_date": datetime.now().isoformat(),
            "data_summary": {},
            "ic_analysis": {},
            "ir_analysis": {},
            "layered_analysis": {},
            "decay_analysis": {},
            "overall_rating": {},
        }

        try:
            # 数据概览
            report["data_summary"] = self._generate_data_summary(factor_data, return_data)

            # IC分析
            ic_data = self.analyzer.calculate_ic(factor_data, return_data, factor_col)
            if not ic_data.empty:
                report["ic_analysis"] = {
                    "ic_statistics": self.analyzer.calculate_ic_statistics(ic_data),
                    "ic_time_series": ic_data.to_dict("records"),
                }

            # IR分析
            if not ic_data.empty:
                ir_data = self.analyzer.calculate_ir(ic_data)
                if not ir_data.empty:
                    report["ir_analysis"] = {
                        "ir_statistics": self.analyzer.calculate_ir_statistics(ir_data),
                        "ir_time_series": ir_data[["trade_date", "ir", "ic_mean", "ic_std"]].to_dict("records"),
                    }

            # 分层回测分析
            layered_data = self.analyzer.factor_layered_backtest(factor_data, return_data)
            if not layered_data.empty:
                report["layered_analysis"] = self.analyzer.analyze_layered_performance(layered_data)

            # 衰减分析
            decay_data = self.analyzer.calculate_factor_decay(factor_data, return_data)
            if not decay_data.empty:
                report["decay_analysis"] = self.analyzer.analyze_factor_decay(decay_data)

            # 综合评级
            report["overall_rating"] = self._calculate_overall_rating(report)

            return report

        except Exception as e:
            logger.error(f"生成因子有效性报告失败: {e}")
            report["error"] = str(e)
            return report

    def _generate_data_summary(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> Dict[str, Any]:
        """生成数据概览"""
        summary = {}

        try:
            # 因子数据统计
            if not factor_data.empty:
                summary["factor_data"] = {
                    "total_records": len(factor_data),
                    "unique_stocks": factor_data["ts_code"].nunique() if "ts_code" in factor_data.columns else 0,
                    "date_range": {
                        "start": (
                            factor_data["trade_date"].min().isoformat() if "trade_date" in factor_data.columns else None
                        ),
                        "end": (
                            factor_data["trade_date"].max().isoformat() if "trade_date" in factor_data.columns else None
                        ),
                    },
                    "missing_ratio": factor_data.isnull().sum().sum() / (len(factor_data) * len(factor_data.columns)),
                }

            # 收益率数据统计
            if not return_data.empty:
                summary["return_data"] = {
                    "total_records": len(return_data),
                    "unique_stocks": return_data["ts_code"].nunique() if "ts_code" in return_data.columns else 0,
                    "avg_return": return_data["return_rate"].mean() if "return_rate" in return_data.columns else 0,
                    "return_volatility": (
                        return_data["return_rate"].std() if "return_rate" in return_data.columns else 0
                    ),
                }

            # 数据匹配度
            if not factor_data.empty and not return_data.empty:
                merged_data = pd.merge(factor_data, return_data, on=["ts_code", "trade_date"], how="inner")

                summary["data_coverage"] = {
                    "matched_records": len(merged_data),
                    "coverage_ratio": len(merged_data) / max(len(factor_data), len(return_data)),
                    "effective_stocks": merged_data["ts_code"].nunique(),
                    "effective_periods": merged_data["trade_date"].nunique(),
                }

        except Exception as e:
            logger.error(f"生成数据概览失败: {e}")
            summary["error"] = str(e)

        return summary

    def _calculate_overall_rating(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """计算因子综合评级"""
        rating = {"score": 0, "grade": "F", "components": {}, "strengths": [], "weaknesses": []}

        try:
            total_score = 0
            component_count = 0

            # IC评分 (30%)
            ic_stats = report.get("ic_analysis", {}).get("ic_statistics", {})
            if ic_stats:
                ic_pearson = ic_stats.get("ic_pearson", {})
                if ic_pearson:
                    ic_mean = abs(ic_pearson.get("mean", 0))
                    ic_positive_ratio = ic_pearson.get("positive_ratio", 0.5)
                    ic_significant_ratio = ic_pearson.get("significant_ratio", 0)

                    ic_score = min(100, (ic_mean * 1000 + ic_positive_ratio * 20 + ic_significant_ratio * 30))
                    rating["components"]["ic_score"] = ic_score
                    total_score += ic_score * 0.3
                    component_count += 1

                    if ic_mean > 0.05:
                        rating["strengths"].append(f"IC均值较高 ({ic_mean:.4f})")
                    if ic_positive_ratio > 0.6:
                        rating["strengths"].append(f"IC正比例较高 ({ic_positive_ratio:.2%})")
                    if ic_mean < 0.02:
                        rating["weaknesses"].append(f"IC均值较低 ({ic_mean:.4f})")

            # IR评分 (25%)
            ir_stats = report.get("ir_analysis", {}).get("ir_statistics", {})
            if ir_stats:
                final_ir = ir_stats.get("final_ir", 0)
                ir_positive_ratio = ir_stats.get("ir_positive_ratio", 0.5)

                ir_score = min(100, (abs(final_ir) * 20 + ir_positive_ratio * 50))
                rating["components"]["ir_score"] = ir_score
                total_score += ir_score * 0.25
                component_count += 1

                if abs(final_ir) > 1:
                    rating["strengths"].append(f"IR值较高 ({final_ir:.2f})")
                if abs(final_ir) < 0.5:
                    rating["weaknesses"].append(f"IR值较低 ({final_ir:.2f})")

            # 分层回测评分 (25%)
            layered_stats = report.get("layered_analysis", {})
            if layered_stats:
                long_short = layered_stats.get("long_short", {})
                monotonicity = layered_stats.get("monotonicity", {})

                layered_score = 0
                if long_short:
                    sharpe_ratio = long_short.get("sharpe_ratio", 0)
                    layered_score += min(50, abs(sharpe_ratio) * 25)

                if monotonicity:
                    if monotonicity.get("is_monotonic", False):
                        layered_score += 50
                    else:
                        layered_score += monotonicity.get("correlation", 0) * 25

                rating["components"]["layered_score"] = layered_score
                total_score += layered_score * 0.25
                component_count += 1

                if long_short.get("sharpe_ratio", 0) > 1:
                    rating["strengths"].append("多空组合夏普比率较高")
                if monotonicity.get("is_monotonic", False):
                    rating["strengths"].append("因子具有良好的单调性")

            # 衰减分析评分 (20%)
            decay_stats = report.get("decay_analysis", {})
            if decay_stats:
                optimal_period = decay_stats.get("optimal_period_pearson", {})
                decay_pattern = decay_stats.get("decay_pattern", {})

                decay_score = 0
                if optimal_period:
                    # 最优持有期在1-5天得满分，超过10天扣分
                    period = optimal_period.get("period", 10)
                    if period <= 5:
                        decay_score += 50
                    elif period <= 10:
                        decay_score += 30
                    else:
                        decay_score += 10

                if decay_pattern:
                    pattern = decay_pattern.get("pattern", "")
                    if pattern in ["稳定型", "增强型"]:
                        decay_score += 50
                    elif pattern == "缓慢衰减":
                        decay_score += 30
                    else:
                        decay_score += 10

                rating["components"]["decay_score"] = decay_score
                total_score += decay_score * 0.2
                component_count += 1

            # 计算最终评分
            if component_count > 0:
                final_score = total_score / component_count * (component_count / 4)  # 标准化到4个组件
                rating["score"] = round(final_score, 2)

                # 评级等级
                if final_score >= 80:
                    rating["grade"] = "A"
                elif final_score >= 70:
                    rating["grade"] = "B"
                elif final_score >= 60:
                    rating["grade"] = "C"
                elif final_score >= 50:
                    rating["grade"] = "D"
                else:
                    rating["grade"] = "F"

        except Exception as e:
            logger.error(f"计算综合评级失败: {e}")
            rating["error"] = str(e)

        return rating

    def rank_factors(self, factor_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对多个因子进行排名

        Args:
            factor_reports: 因子报告列表

        Returns:
            排名后的因子列表
        """
        try:
            # 提取评分信息
            factor_scores = []
            for report in factor_reports:
                overall_rating = report.get("overall_rating", {})
                score = overall_rating.get("score", 0)

                factor_scores.append(
                    {
                        "factor_name": report.get("factor_name", ""),
                        "score": score,
                        "grade": overall_rating.get("grade", "F"),
                        "report": report,
                    }
                )

            # 按评分降序排序
            factor_scores.sort(key=lambda x: x["score"], reverse=True)

            # 添加排名
            for i, factor_info in enumerate(factor_scores):
                factor_info["rank"] = i + 1

            return factor_scores

        except Exception as e:
            logger.error(f"因子排名失败: {e}")
            return factor_reports

    def generate_factor_recommendation(self, ranked_factors: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
        """
        生成因子推荐

        Args:
            ranked_factors: 排名后的因子列表
            top_n: 推荐因子数量

        Returns:
            因子推荐结果
        """
        try:
            recommendation = {"recommended_factors": [], "summary": {}, "usage_suggestions": []}

            # 选择前N个因子
            top_factors = ranked_factors[:top_n]

            for factor_info in top_factors:
                factor_summary = {
                    "factor_name": factor_info["factor_name"],
                    "rank": factor_info["rank"],
                    "score": factor_info["score"],
                    "grade": factor_info["grade"],
                    "key_strengths": factor_info["report"].get("overall_rating", {}).get("strengths", [])[:3],
                }
                recommendation["recommended_factors"].append(factor_summary)

            # 生成总结
            if top_factors:
                avg_score = np.mean([f["score"] for f in top_factors])
                grade_distribution = {}
                for factor in top_factors:
                    grade = factor["grade"]
                    grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

                recommendation["summary"] = {
                    "total_factors": len(top_factors),
                    "average_score": round(avg_score, 2),
                    "grade_distribution": grade_distribution,
                    "best_factor": top_factors[0]["factor_name"] if top_factors else None,
                }

            # 使用建议
            if avg_score >= 70:
                recommendation["usage_suggestions"].append("推荐因子质量较高，可直接用于投资决策")
            elif avg_score >= 50:
                recommendation["usage_suggestions"].append("推荐因子质量中等，建议结合其他因子使用")
            else:
                recommendation["usage_suggestions"].append("推荐因子质量较低，建议谨慎使用或进一步优化")

            if len(top_factors) > 5:
                recommendation["usage_suggestions"].append("可考虑因子合成以提高稳定性")

            return recommendation

        except Exception as e:
            logger.error(f"生成因子推荐失败: {e}")
            return {"error": str(e)}
