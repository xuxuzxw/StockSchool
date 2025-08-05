import warnings
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子标准化和评分机制
实现各种因子标准化算法和评分功能
"""


from .factor_models import FactorCategory, FactorType


class StandardizationMethod(Enum):
    """标准化方法枚举"""

    ZSCORE = "zscore"  # Z-score标准化
    QUANTILE = "quantile"  # 分位数标准化
    RANK = "rank"  # 排名标准化
    MINMAX = "minmax"  # 最小最大值标准化
    ROBUST = "robust"  # 鲁棒标准化（基于中位数和MAD）


class OutlierTreatment(Enum):
    """异常值处理方法枚举"""

    NONE = "none"  # 不处理
    CLIP = "clip"  # 截尾处理
    WINSORIZE = "winsorize"  # 缩尾处理
    REMOVE = "remove"  # 移除异常值


class FactorStandardizer:
    """因子标准化器"""

    def __init__(self, engine=None):
        """初始化因子标准化器"""
        self.engine = engine

    def zscore_standardize(
        self,
        factor_data: pd.Series,
        outlier_treatment: OutlierTreatment = OutlierTreatment.CLIP,
        clip_quantiles: Tuple[float, float] = (0.01, 0.99),
    ) -> pd.Series:
        """
        Z-score标准化

        Args:
            factor_data: 因子数据
            outlier_treatment: 异常值处理方法
            clip_quantiles: 截尾分位数

        Returns:
            标准化后的因子数据
        """
        if factor_data.empty or factor_data.isna().all():
            return factor_data

        # 处理异常值
        processed_data = self._handle_outliers(factor_data, outlier_treatment, clip_quantiles)

        # Z-score标准化
        mean_val = processed_data.mean()
        std_val = processed_data.std()

        if std_val == 0 or pd.isna(std_val):
            logger.warning("因子数据标准差为0或NaN，返回原数据")
            return processed_data

        standardized = (processed_data - mean_val) / std_val

        return standardized

    def quantile_standardize(self, factor_data: pd.Series, n_quantiles: int = 100) -> pd.Series:
        """
        分位数标准化（转换为均匀分布）

        Args:
            factor_data: 因子数据
            n_quantiles: 分位数数量

        Returns:
            标准化后的因子数据
        """
        if factor_data.empty or factor_data.isna().all():
            return factor_data

        # 计算分位数
        valid_data = factor_data.dropna()
        if len(valid_data) < 2:
            return factor_data

        # 使用分位数变换
        quantiles = np.linspace(0, 1, n_quantiles + 1)
        quantile_values = valid_data.quantile(quantiles)

        # 映射到[0, 1]区间
        standardized = factor_data.copy()
        for i, val in enumerate(factor_data):
            if pd.notna(val):
                # 找到对应的分位数位置
                rank_position = (valid_data <= val).sum() / len(valid_data)
                standardized.iloc[i] = rank_position

        return standardized

    def rank_standardize(self, factor_data: pd.Series, ascending: bool = True) -> pd.Series:
        """
        排名标准化

        Args:
            factor_data: 因子数据
            ascending: 是否升序排名

        Returns:
            标准化后的因子数据（排名）
        """
        if factor_data.empty or factor_data.isna().all():
            return factor_data

        # 计算排名
        ranks = factor_data.rank(ascending=ascending, na_option="keep")

        # 标准化到[0, 1]区间
        valid_ranks = ranks.dropna()
        if len(valid_ranks) > 1:
            min_rank = valid_ranks.min()
            max_rank = valid_ranks.max()
            if max_rank > min_rank:
                standardized = (ranks - min_rank) / (max_rank - min_rank)
            else:
                standardized = pd.Series(0.5, index=ranks.index)
        else:
            standardized = ranks

        return standardized

    def minmax_standardize(self, factor_data: pd.Series, feature_range: Tuple[float, float] = (0, 1)) -> pd.Series:
        """
        最小最大值标准化

        Args:
            factor_data: 因子数据
            feature_range: 目标范围

        Returns:
            标准化后的因子数据
        """
        if factor_data.empty or factor_data.isna().all():
            return factor_data

        min_val = factor_data.min()
        max_val = factor_data.max()

        if max_val == min_val:
            # 所有值相同，返回中间值
            mid_val = (feature_range[0] + feature_range[1]) / 2
            return pd.Series(mid_val, index=factor_data.index)

        # 标准化到指定范围
        standardized = (factor_data - min_val) / (max_val - min_val)
        standardized = standardized * (feature_range[1] - feature_range[0]) + feature_range[0]

        return standardized

    def robust_standardize(self, factor_data: pd.Series) -> pd.Series:
        """
        鲁棒标准化（基于中位数和MAD）

        Args:
            factor_data: 因子数据

        Returns:
            标准化后的因子数据
        """
        if factor_data.empty or factor_data.isna().all():
            return factor_data

        # 计算中位数和MAD（中位数绝对偏差）
        median_val = factor_data.median()
        mad_val = (factor_data - median_val).abs().median()

        if mad_val == 0 or pd.isna(mad_val):
            logger.warning("MAD为0或NaN，使用标准差替代")
            mad_val = factor_data.std()
            if mad_val == 0 or pd.isna(mad_val):
                return factor_data

        # 鲁棒标准化
        standardized = (factor_data - median_val) / mad_val

        return standardized

    def _handle_outliers(
        self, factor_data: pd.Series, treatment: OutlierTreatment, clip_quantiles: Tuple[float, float]
    ) -> pd.Series:
        """
        处理异常值

        Args:
            factor_data: 因子数据
            treatment: 处理方法
            clip_quantiles: 截尾分位数

        Returns:
            处理后的因子数据
        """
        if treatment == OutlierTreatment.NONE:
            return factor_data

        if factor_data.empty or factor_data.isna().all():
            return factor_data

        if treatment == OutlierTreatment.CLIP:
            # 截尾处理
            lower_bound = factor_data.quantile(clip_quantiles[0])
            upper_bound = factor_data.quantile(clip_quantiles[1])
            return factor_data.clip(lower_bound, upper_bound)

        elif treatment == OutlierTreatment.WINSORIZE:
            # 缩尾处理（替换为边界值）
            lower_bound = factor_data.quantile(clip_quantiles[0])
            upper_bound = factor_data.quantile(clip_quantiles[1])
            processed = factor_data.copy()
            processed[processed < lower_bound] = lower_bound
            processed[processed > upper_bound] = upper_bound
            return processed

        elif treatment == OutlierTreatment.REMOVE:
            # 移除异常值（设为NaN）
            lower_bound = factor_data.quantile(clip_quantiles[0])
            upper_bound = factor_data.quantile(clip_quantiles[1])
            processed = factor_data.copy()
            processed[(processed < lower_bound) | (processed > upper_bound)] = np.nan
            return processed

        return factor_data

    def standardize(
        self, factor_data: pd.Series, method: StandardizationMethod = StandardizationMethod.ZSCORE, **kwargs
    ) -> pd.Series:
        """
        统一的标准化接口

        Args:
            factor_data: 因子数据
            method: 标准化方法
            **kwargs: 其他参数

        Returns:
            标准化后的因子数据
        """
        try:
            if method == StandardizationMethod.ZSCORE:
                return self.zscore_standardize(factor_data, **kwargs)
            elif method == StandardizationMethod.QUANTILE:
                return self.quantile_standardize(factor_data, **kwargs)
            elif method == StandardizationMethod.RANK:
                return self.rank_standardize(factor_data, **kwargs)
            elif method == StandardizationMethod.MINMAX:
                return self.minmax_standardize(factor_data, **kwargs)
            elif method == StandardizationMethod.ROBUST:
                return self.robust_standardize(factor_data, **kwargs)
            else:
                logger.error(f"不支持的标准化方法: {method}")
                return factor_data

        except Exception as e:
            logger.error(f"因子标准化失败: {e}")
            return factor_data


class IndustryFactorScorer:
    """行业内因子评分器"""

    def __init__(self, engine):
        """初始化行业内因子评分器"""
        self.engine = engine
        self.standardizer = FactorStandardizer()

    def get_industry_mapping(self, ts_codes: List[str]) -> pd.DataFrame:
        """
        获取股票行业映射

        Args:
            ts_codes: 股票代码列表

        Returns:
            包含股票代码和行业信息的DataFrame
        """
        try:
            # 构建查询SQL
            placeholders = ",".join([f"'{code}'" for code in ts_codes])
            query = text(
                f"""
                SELECT ts_code, industry, area
                FROM stock_basic
                WHERE ts_code IN ({placeholders})
                AND list_status = 'L'
            """
            )

            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(query)
                industry_data = pd.DataFrame(result.fetchall(), columns=result.keys())

            return industry_data

        except Exception as e:
            logger.error(f"获取行业映射失败: {e}")
            return pd.DataFrame()

    def industry_standardize(
        self,
        factor_data: pd.DataFrame,
        ts_code_col: str = "ts_code",
        factor_col: str = "factor_value",
        method: StandardizationMethod = StandardizationMethod.ZSCORE,
    ) -> pd.DataFrame:
        """
        按行业进行因子标准化

        Args:
            factor_data: 因子数据，包含股票代码和因子值
            ts_code_col: 股票代码列名
            factor_col: 因子值列名
            method: 标准化方法

        Returns:
            标准化后的因子数据
        """
        if factor_data.empty:
            return factor_data

        # 获取行业映射
        ts_codes = factor_data[ts_code_col].unique().tolist()
        industry_mapping = self.get_industry_mapping(ts_codes)

        if industry_mapping.empty:
            logger.warning("无法获取行业信息，使用全市场标准化")
            factor_data["standardized_value"] = self.standardizer.standardize(factor_data[factor_col], method=method)
            return factor_data

        # 合并行业信息
        merged_data = factor_data.merge(industry_mapping, on=ts_code_col, how="left")

        # 按行业分组标准化
        standardized_values = []

        for industry, group in merged_data.groupby("industry"):
            if len(group) < 2:
                # 行业内股票数量太少，使用原值
                group_standardized = group[factor_col]
            else:
                # 行业内标准化
                group_standardized = self.standardizer.standardize(group[factor_col], method=method)

            standardized_values.append(
                pd.DataFrame(
                    {
                        ts_code_col: group[ts_code_col],
                        "industry": group["industry"],
                        "standardized_value": group_standardized,
                    }
                )
            )

        # 合并结果
        result = pd.concat(standardized_values, ignore_index=True)

        # 合并回原数据
        final_result = factor_data.merge(result[[ts_code_col, "standardized_value"]], on=ts_code_col, how="left")

        return final_result

    def industry_rank(
        self,
        factor_data: pd.DataFrame,
        ts_code_col: str = "ts_code",
        factor_col: str = "factor_value",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        计算行业内排名

        Args:
            factor_data: 因子数据
            ts_code_col: 股票代码列名
            factor_col: 因子值列名
            ascending: 是否升序排名

        Returns:
            包含行业内排名的数据
        """
        if factor_data.empty:
            return factor_data

        # 获取行业映射
        ts_codes = factor_data[ts_code_col].unique().tolist()
        industry_mapping = self.get_industry_mapping(ts_codes)

        if industry_mapping.empty:
            logger.warning("无法获取行业信息，使用全市场排名")
            factor_data["industry_rank"] = factor_data[factor_col].rank(ascending=ascending, na_option="keep")
            factor_data["industry_rank_pct"] = factor_data["industry_rank"] / len(factor_data)
            return factor_data

        # 合并行业信息
        merged_data = factor_data.merge(industry_mapping, on=ts_code_col, how="left")

        # 按行业分组排名
        ranking_results = []

        for industry, group in merged_data.groupby("industry"):
            group_copy = group.copy()

            # 行业内排名
            group_copy["industry_rank"] = group[factor_col].rank(ascending=ascending, na_option="keep")

            # 行业内排名百分比
            valid_ranks = group_copy["industry_rank"].dropna()
            if len(valid_ranks) > 0:
                group_copy["industry_rank_pct"] = group_copy["industry_rank"] / len(valid_ranks)
            else:
                group_copy["industry_rank_pct"] = np.nan

            # 行业内股票数量
            group_copy["industry_stock_count"] = len(group)

            ranking_results.append(group_copy)

        # 合并结果
        result = pd.concat(ranking_results, ignore_index=True)

        # 选择需要的列
        result_cols = [ts_code_col, "industry", "industry_rank", "industry_rank_pct", "industry_stock_count"]

        # 合并回原数据
        final_result = factor_data.merge(result[result_cols], on=ts_code_col, how="left")

        return final_result

    def industry_neutralize(
        self, factor_data: pd.DataFrame, ts_code_col: str = "ts_code", factor_col: str = "factor_value"
    ) -> pd.DataFrame:
        """
        行业中性化处理

        Args:
            factor_data: 因子数据
            ts_code_col: 股票代码列名
            factor_col: 因子值列名

        Returns:
            行业中性化后的因子数据
        """
        if factor_data.empty:
            return factor_data

        # 获取行业映射
        ts_codes = factor_data[ts_code_col].unique().tolist()
        industry_mapping = self.get_industry_mapping(ts_codes)

        if industry_mapping.empty:
            logger.warning("无法获取行业信息，跳过行业中性化")
            factor_data["neutralized_value"] = factor_data[factor_col]
            return factor_data

        # 合并行业信息
        merged_data = factor_data.merge(industry_mapping, on=ts_code_col, how="left")

        # 计算行业均值
        industry_means = merged_data.groupby("industry")[factor_col].mean()

        # 行业中性化：因子值 - 行业均值
        merged_data["neutralized_value"] = merged_data.apply(
            lambda row: (
                row[factor_col] - industry_means.get(row["industry"], 0) if pd.notna(row[factor_col]) else np.nan
            ),
            axis=1,
        )

        # 合并回原数据
        final_result = factor_data.merge(merged_data[[ts_code_col, "neutralized_value"]], on=ts_code_col, how="left")

        return final_result


class FactorQuantileCalculator:
    """因子分位数计算器"""

    def __init__(self, engine):
        """初始化因子分位数计算器"""
        self.engine = engine

    def calculate_market_quantiles(self, factor_data: pd.Series, n_quantiles: int = 10) -> pd.DataFrame:
        """
        计算全市场因子分位数

        Args:
            factor_data: 因子数据
            n_quantiles: 分位数数量

        Returns:
            包含分位数信息的DataFrame
        """
        if factor_data.empty or factor_data.isna().all():
            return pd.DataFrame()

        # 计算分位数边界
        quantile_boundaries = []
        for i in range(n_quantiles + 1):
            q = i / n_quantiles
            boundary = factor_data.quantile(q)
            quantile_boundaries.append(boundary)

        # 分配分位数
        quantile_labels = pd.cut(
            factor_data,
            bins=quantile_boundaries,
            labels=range(1, n_quantiles + 1),
            include_lowest=True,
            duplicates="drop",
        )

        # 创建结果DataFrame
        result = pd.DataFrame(
            {
                "factor_value": factor_data,
                "quantile": quantile_labels,
                "quantile_rank": quantile_labels.astype(float) / n_quantiles,
            }
        )

        # 添加分位数统计信息
        quantile_stats = []
        for q in range(1, n_quantiles + 1):
            mask = result["quantile"] == q
            if mask.sum() > 0:
                stats = {
                    "quantile": q,
                    "count": mask.sum(),
                    "min_value": result.loc[mask, "factor_value"].min(),
                    "max_value": result.loc[mask, "factor_value"].max(),
                    "mean_value": result.loc[mask, "factor_value"].mean(),
                    "std_value": result.loc[mask, "factor_value"].std(),
                }
                quantile_stats.append(stats)

        result.quantile_stats = pd.DataFrame(quantile_stats)

        return result

    def calculate_industry_quantiles(
        self,
        factor_data: pd.DataFrame,
        ts_code_col: str = "ts_code",
        factor_col: str = "factor_value",
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        计算行业内因子分位数

        Args:
            factor_data: 因子数据
            ts_code_col: 股票代码列名
            factor_col: 因子值列名
            n_quantiles: 分位数数量

        Returns:
            包含行业内分位数信息的DataFrame
        """
        if factor_data.empty:
            return factor_data

        # 获取行业映射
        industry_scorer = IndustryFactorScorer(self.engine)
        ts_codes = factor_data[ts_code_col].unique().tolist()
        industry_mapping = industry_scorer.get_industry_mapping(ts_codes)

        if industry_mapping.empty:
            logger.warning("无法获取行业信息，使用全市场分位数")
            market_quantiles = self.calculate_market_quantiles(factor_data[factor_col], n_quantiles)
            factor_data["industry_quantile"] = market_quantiles["quantile"]
            factor_data["industry_quantile_rank"] = market_quantiles["quantile_rank"]
            return factor_data

        # 合并行业信息
        merged_data = factor_data.merge(industry_mapping, on=ts_code_col, how="left")

        # 按行业分组计算分位数
        quantile_results = []

        for industry, group in merged_data.groupby("industry"):
            if len(group) < n_quantiles:
                # 行业内股票数量太少，使用简单排名
                group_copy = group.copy()
                group_copy["industry_quantile"] = group[factor_col].rank(method="first", na_option="keep")
                group_copy["industry_quantile_rank"] = group_copy["industry_quantile"] / len(group)
            else:
                # 计算行业内分位数
                industry_quantiles = self.calculate_market_quantiles(group[factor_col], n_quantiles)

                group_copy = group.copy()
                group_copy["industry_quantile"] = industry_quantiles["quantile"]
                group_copy["industry_quantile_rank"] = industry_quantiles["quantile_rank"]

            quantile_results.append(group_copy)

        # 合并结果
        result = pd.concat(quantile_results, ignore_index=True)

        # 合并回原数据
        final_result = factor_data.merge(
            result[[ts_code_col, "industry_quantile", "industry_quantile_rank"]], on=ts_code_col, how="left"
        )

        return final_result

    def update_dynamic_quantiles(
        self, factor_data: pd.Series, historical_quantiles: Optional[pd.DataFrame] = None, decay_factor: float = 0.95
    ) -> pd.DataFrame:
        """
        动态更新分位数

        Args:
            factor_data: 新的因子数据
            historical_quantiles: 历史分位数信息
            decay_factor: 衰减因子

        Returns:
            更新后的分位数信息
        """
        if factor_data.empty:
            return pd.DataFrame()

        # 计算当前分位数
        current_quantiles = self.calculate_market_quantiles(factor_data)

        if historical_quantiles is None or historical_quantiles.empty:
            return current_quantiles

        # 动态更新分位数边界
        updated_stats = []

        for _, current_stat in current_quantiles.quantile_stats.iterrows():
            quantile = current_stat["quantile"]

            # 查找历史对应分位数
            hist_stat = historical_quantiles.quantile_stats[historical_quantiles.quantile_stats["quantile"] == quantile]

            if not hist_stat.empty:
                hist_stat = hist_stat.iloc[0]

                # 指数加权更新
                updated_stat = {
                    "quantile": quantile,
                    "count": current_stat["count"],
                    "min_value": (
                        decay_factor * hist_stat["min_value"] + (1 - decay_factor) * current_stat["min_value"]
                    ),
                    "max_value": (
                        decay_factor * hist_stat["max_value"] + (1 - decay_factor) * current_stat["max_value"]
                    ),
                    "mean_value": (
                        decay_factor * hist_stat["mean_value"] + (1 - decay_factor) * current_stat["mean_value"]
                    ),
                    "std_value": (
                        decay_factor * hist_stat["std_value"] + (1 - decay_factor) * current_stat["std_value"]
                    ),
                }
            else:
                updated_stat = current_stat.to_dict()

            updated_stats.append(updated_stat)

        # 更新结果
        result = current_quantiles.copy()
        result.quantile_stats = pd.DataFrame(updated_stats)

        return result


class FactorComposer:
    """因子合成器"""

    def __init__(self):
        """初始化因子合成器"""
        pass

    def equal_weight_compose(self, factor_data: pd.DataFrame, factor_cols: List[str]) -> pd.Series:
        """
        等权重因子合成

        Args:
            factor_data: 包含多个因子的数据
            factor_cols: 要合成的因子列名

        Returns:
            合成后的因子值
        """
        if factor_data.empty or not factor_cols:
            return pd.Series(dtype=float)

        # 检查因子列是否存在
        available_cols = [col for col in factor_cols if col in factor_data.columns]
        if not available_cols:
            logger.warning(f"指定的因子列不存在: {factor_cols}")
            return pd.Series(dtype=float, index=factor_data.index)

        # 等权重合成
        composite_factor = factor_data[available_cols].mean(axis=1)

        return composite_factor

    def weighted_compose(self, factor_data: pd.DataFrame, factor_weights: Dict[str, float]) -> pd.Series:
        """
        加权因子合成

        Args:
            factor_data: 包含多个因子的数据
            factor_weights: 因子权重字典

        Returns:
            合成后的因子值
        """
        if factor_data.empty or not factor_weights:
            return pd.Series(dtype=float)

        # 检查因子列和权重
        available_factors = {}
        total_weight = 0

        for factor_name, weight in factor_weights.items():
            if factor_name in factor_data.columns:
                available_factors[factor_name] = weight
                total_weight += abs(weight)

        if not available_factors:
            logger.warning("没有可用的因子进行合成")
            return pd.Series(dtype=float, index=factor_data.index)

        # 权重归一化
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in available_factors.items()}
        else:
            normalized_weights = available_factors

        # 加权合成
        composite_factor = pd.Series(0.0, index=factor_data.index)

        for factor_name, weight in normalized_weights.items():
            composite_factor += factor_data[factor_name] * weight

        return composite_factor

    def ic_weighted_compose(
        self, factor_data: pd.DataFrame, factor_cols: List[str], ic_values: Dict[str, float]
    ) -> pd.Series:
        """
        基于IC值的加权因子合成

        Args:
            factor_data: 包含多个因子的数据
            factor_cols: 要合成的因子列名
            ic_values: 各因子的IC值

        Returns:
            合成后的因子值
        """
        if factor_data.empty or not factor_cols or not ic_values:
            return pd.Series(dtype=float)

        # 构建权重字典（使用IC绝对值作为权重）
        factor_weights = {}
        for factor_name in factor_cols:
            if factor_name in ic_values:
                # 使用IC绝对值作为权重，IC越大权重越大
                factor_weights[factor_name] = abs(ic_values[factor_name])

        return self.weighted_compose(factor_data, factor_weights)

    def pca_compose(self, factor_data: pd.DataFrame, factor_cols: List[str], n_components: int = 1) -> pd.Series:
        """
        基于PCA的因子合成

        Args:
            factor_data: 包含多个因子的数据
            factor_cols: 要合成的因子列名
            n_components: 主成分数量

        Returns:
            合成后的因子值
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("需要安装scikit-learn库进行PCA分析")
            return self.equal_weight_compose(factor_data, factor_cols)

        if factor_data.empty or not factor_cols:
            return pd.Series(dtype=float)

        # 检查因子列是否存在
        available_cols = [col for col in factor_cols if col in factor_data.columns]
        if len(available_cols) < 2:
            logger.warning("PCA需要至少2个因子，使用等权重合成")
            return self.equal_weight_compose(factor_data, available_cols)

        # 准备数据
        factor_matrix = factor_data[available_cols].dropna()
        if len(factor_matrix) < 2:
            logger.warning("有效数据点太少，无法进行PCA")
            return self.equal_weight_compose(factor_data, available_cols)

        # 标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(factor_matrix)

        # PCA分析
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)

        # 使用第一主成分作为合成因子
        composite_values = pca_result[:, 0]

        # 创建结果Series
        composite_factor = pd.Series(index=factor_matrix.index, data=composite_values).reindex(factor_data.index)

        return composite_factor

    def validate_composition(
        self, original_factors: pd.DataFrame, composite_factor: pd.Series, factor_cols: List[str]
    ) -> Dict[str, float]:
        """
        验证因子合成的有效性

        Args:
            original_factors: 原始因子数据
            composite_factor: 合成因子
            factor_cols: 原始因子列名

        Returns:
            验证结果字典
        """
        validation_results = {}

        try:
            # 计算合成因子与原始因子的相关性
            correlations = {}
            for factor_col in factor_cols:
                if factor_col in original_factors.columns:
                    corr = composite_factor.corr(original_factors[factor_col])
                    correlations[factor_col] = corr if pd.notna(corr) else 0.0

            validation_results["correlations"] = correlations
            validation_results["avg_correlation"] = np.mean(list(correlations.values()))

            # 计算信息保留度（方差解释比例）
            if len(factor_cols) > 1:
                available_cols = [col for col in factor_cols if col in original_factors.columns]
                if available_cols:
                    factor_matrix = original_factors[available_cols].dropna()
                    if not factor_matrix.empty:
                        # 计算原始因子的总方差
                        total_variance = factor_matrix.var().sum()

                        # 计算合成因子的方差
                        composite_variance = composite_factor.var()

                        # 信息保留度
                        if total_variance > 0:
                            information_retention = composite_variance / total_variance
                            validation_results["information_retention"] = information_retention

            # 计算合成因子的统计特性
            validation_results["composite_stats"] = {
                "mean": composite_factor.mean(),
                "std": composite_factor.std(),
                "skewness": composite_factor.skew(),
                "kurtosis": composite_factor.kurtosis(),
                "valid_count": composite_factor.count(),
                "missing_count": composite_factor.isna().sum(),
            }

        except Exception as e:
            logger.error(f"因子合成验证失败: {e}")
            validation_results["error"] = str(e)

        return validation_results


class FactorScoreManager:
    """因子评分历史管理器"""

    def __init__(self, engine):
        """初始化因子评分历史管理器"""
        self.engine = engine

    def save_factor_scores(
        self, scores_data: pd.DataFrame, score_date: date, factor_name: str, score_type: str = "standardized"
    ) -> bool:
        """
        保存因子评分到数据库

        Args:
            scores_data: 评分数据
            score_date: 评分日期
            factor_name: 因子名称
            score_type: 评分类型

        Returns:
            是否保存成功
        """
        try:
            # 准备保存的数据
            save_data = scores_data.copy()
            save_data["score_date"] = score_date
            save_data["factor_name"] = factor_name
            save_data["score_type"] = score_type
            save_data["created_time"] = datetime.now()

            # 保存到数据库
            save_data.to_sql("factor_scores_history", self.engine, if_exists="append", index=False, method="multi")

            logger.info(f"成功保存因子评分: {factor_name}, 日期: {score_date}")
            return True

        except Exception as e:
            logger.error(f"保存因子评分失败: {e}")
            return False

    def load_factor_scores(
        self,
        factor_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        score_type: str = "standardized",
    ) -> pd.DataFrame:
        """
        加载历史因子评分

        Args:
            factor_name: 因子名称
            start_date: 开始日期
            end_date: 结束日期
            score_type: 评分类型

        Returns:
            历史评分数据
        """
        try:
            # 构建查询SQL
            query = text(
                """
                SELECT * FROM factor_scores_history
                WHERE factor_name = :factor_name
                AND score_type = :score_type
                AND (:start_date IS NULL OR score_date >= :start_date)
                AND (:end_date IS NULL OR score_date <= :end_date)
                ORDER BY score_date, ts_code
            """
            )

            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(
                    query,
                    {
                        "factor_name": factor_name,
                        "score_type": score_type,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                )

                scores_data = pd.DataFrame(result.fetchall(), columns=result.keys())

            if not scores_data.empty:
                scores_data["score_date"] = pd.to_datetime(scores_data["score_date"])

            return scores_data

        except Exception as e:
            logger.error(f"加载因子评分失败: {e}")
            return pd.DataFrame()

    def analyze_score_stability(self, factor_name: str, lookback_days: int = 60) -> Dict[str, Any]:
        """
        分析因子评分的稳定性

        Args:
            factor_name: 因子名称
            lookback_days: 回看天数

        Returns:
            稳定性分析结果
        """
        try:
            # 加载历史评分数据
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days)

            scores_data = self.load_factor_scores(factor_name, start_date, end_date)

            if scores_data.empty:
                return {"error": "无历史评分数据"}

            # 计算稳定性指标
            stability_metrics = {}

            # 按股票分组分析
            stock_stability = []

            for ts_code, group in scores_data.groupby("ts_code"):
                if len(group) < 2:
                    continue

                # 计算评分的标准差和变异系数
                score_std = group["standardized_value"].std()
                score_mean = group["standardized_value"].mean()

                if abs(score_mean) > 1e-6:
                    cv = score_std / abs(score_mean)
                else:
                    cv = np.inf

                stock_stability.append(
                    {
                        "ts_code": ts_code,
                        "score_std": score_std,
                        "score_mean": score_mean,
                        "coefficient_of_variation": cv,
                        "data_points": len(group),
                    }
                )

            if stock_stability:
                stability_df = pd.DataFrame(stock_stability)

                stability_metrics["overall_stability"] = {
                    "avg_std": stability_df["score_std"].mean(),
                    "avg_cv": stability_df["coefficient_of_variation"].mean(),
                    "stable_stocks_ratio": (stability_df["coefficient_of_variation"] < 0.5).mean(),
                    "total_stocks": len(stability_df),
                }

                # 时间序列稳定性
                daily_stats = (
                    scores_data.groupby("score_date").agg({"standardized_value": ["mean", "std", "count"]}).round(4)
                )

                stability_metrics["temporal_stability"] = {
                    "daily_mean_std": daily_stats[("standardized_value", "std")].std(),
                    "daily_count_std": daily_stats[("standardized_value", "count")].std(),
                    "data_coverage": len(daily_stats) / lookback_days,
                }

            return stability_metrics

        except Exception as e:
            logger.error(f"分析因子评分稳定性失败: {e}")
            return {"error": str(e)}
