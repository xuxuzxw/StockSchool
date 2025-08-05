import os
import sys
import warnings
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.compute.factor_composer import FactorComposer
from src.compute.factor_correlation_analyzer import FactorCorrelationAnalyzer
from src.compute.factor_effectiveness_analyzer import FactorEffectivenessAnalyzer
from src.compute.factor_standardizer import FactorStandardizer

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子分析单元测试
测试因子标准化、有效性分析、相关性分析等功能
"""


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


# 忽略pandas警告
warnings.filterwarnings("ignore", category=FutureWarning)


class TestAnalysisDataGenerator:
    """分析测试数据生成器"""

    @staticmethod
    def generate_factor_data(n_stocks=100, n_days=252, n_factors=5, seed=42):
        """生成因子测试数据"""
        np.random.seed(seed)

        # 生成股票代码
        stock_codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n_stocks + 1)]

        # 生成日期
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

        # 生成因子数据
        data_list = []

        for stock in stock_codes:
            for date in dates:
                row_data = {"ts_code": stock, "factor_date": date}

                # 生成相关的因子数据
                base_factor = np.random.normal(0, 1)

                for i in range(n_factors):
                    # 添加一些相关性
                    correlation = 0.3 if i > 0 else 0
                    factor_value = base_factor * correlation + np.random.normal(0, 1) * (1 - correlation)
                    row_data[f"factor_{i+1}"] = factor_value

                data_list.append(row_data)

        return pd.DataFrame(data_list)

    @staticmethod
    def generate_return_data(n_stocks=100, n_days=252, seed=42):
        """生成收益率数据"""
        np.random.seed(seed)

        stock_codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n_stocks + 1)]
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

        data_list = []

        for stock in stock_codes:
            # 生成相关的收益率序列
            returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率

            for i, date in enumerate(dates):
                data_list.append(
                    {
                        "ts_code": stock,
                        "trade_date": date,
                        "return_1d": returns[i],
                        "return_5d": np.mean(returns[max(0, i - 4) : i + 1]) if i >= 4 else np.nan,
                        "return_20d": np.mean(returns[max(0, i - 19) : i + 1]) if i >= 19 else np.nan,
                    }
                )

        return pd.DataFrame(data_list)

    @staticmethod
    def generate_industry_data(n_stocks=100):
        """生成行业数据"""
        np.random.seed(42)

        stock_codes = [f"{str(i).zfill(6)}.SZ" for i in range(1, n_stocks + 1)]
        industries = ["银行", "地产", "科技", "医药", "消费", "制造", "能源", "材料"]

        data_list = []
        for stock in stock_codes:
            industry = np.random.choice(industries)
            data_list.append({"ts_code": stock, "industry": industry})

        return pd.DataFrame(data_list)


class TestFactorStandardizer:
    """因子标准化器测试"""

    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()

    @pytest.fixture
    def standardizer(self, mock_engine):
        """创建标准化器实例"""
        return FactorStandardizer(mock_engine)

    @pytest.fixture
    def factor_data(self):
        """因子数据"""
        return TestAnalysisDataGenerator.generate_factor_data(50, 100, 3)

    def test_zscore_standardization(self, standardizer, factor_data):
        """测试Z-score标准化"""
        # 选择一个因子进行标准化
        factor_values = factor_data["factor_1"].values

        standardized = standardizer.zscore_standardize(factor_values)

        # 验证标准化结果
        assert len(standardized) == len(factor_values), "标准化后长度应保持一致"

        # 验证均值接近0，标准差接近1
        valid_values = standardized[~np.isnan(standardized)]
        assert abs(np.mean(valid_values)) < 1e-10, "标准化后均值应接近0"
        assert abs(np.std(valid_values, ddof=1) - 1) < 1e-10, "标准化后标准差应接近1"

    def test_quantile_standardization(self, standardizer, factor_data):
        """测试分位数标准化"""
        factor_values = factor_data["factor_1"].values

        standardized = standardizer.quantile_standardize(factor_values)

        # 验证标准化结果
        assert len(standardized) == len(factor_values), "标准化后长度应保持一致"

        # 验证分位数标准化的特性
        valid_values = standardized[~np.isnan(standardized)]
        assert all(0 <= val <= 1 for val in valid_values), "分位数标准化结果应在0-1之间"

    def test_rank_standardization(self, standardizer, factor_data):
        """测试排名标准化"""
        factor_values = factor_data["factor_1"].values

        standardized = standardizer.rank_standardize(factor_values)

        # 验证标准化结果
        assert len(standardized) == len(factor_values), "标准化后长度应保持一致"

        # 验证排名标准化的特性
        valid_values = standardized[~np.isnan(standardized)]
        assert all(0 <= val <= 1 for val in valid_values), "排名标准化结果应在0-1之间"

    def test_outlier_handling(self, standardizer):
        """测试异常值处理"""
        # 创建包含异常值的数据
        data = np.array([1, 2, 3, 4, 5, 100, -100, 6, 7, 8])

        # 测试截尾处理
        clipped = standardizer.clip_outliers(data, method="clip", threshold=2)
        assert not any(np.abs(stats.zscore(clipped)) > 2), "截尾后不应有超过2σ的异常值"

        # 测试缩尾处理
        winsorized = standardizer.clip_outliers(data, method="winsorize", threshold=2)
        assert len(winsorized) == len(data), "缩尾后长度应保持一致"

        # 测试移除处理
        removed = standardizer.clip_outliers(data, method="remove", threshold=2)
        assert len(removed) < len(data), "移除异常值后长度应减少"

    def test_industry_neutral_standardization(self, standardizer, factor_data):
        """测试行业中性化标准化"""
        # 添加行业信息
        industry_data = TestAnalysisDataGenerator.generate_industry_data(50)

        # 合并数据
        merged_data = factor_data.merge(industry_data, on="ts_code", how="left")

        # 按行业进行标准化
        standardized = standardizer.industry_neutral_standardize(
            merged_data, factor_column="factor_1", industry_column="industry"
        )

        # 验证行业中性化效果
        assert len(standardized) == len(merged_data), "标准化后长度应保持一致"

        # 验证每个行业内的标准化效果
        for industry in merged_data["industry"].unique():
            industry_mask = merged_data["industry"] == industry
            industry_values = standardized[industry_mask]
            valid_values = industry_values[~np.isnan(industry_values)]

            if len(valid_values) > 1:
                # 行业内均值应接近0
                assert abs(np.mean(valid_values)) < 0.1, f"行业 {industry} 内均值应接近0"


class TestFactorEffectivenessAnalyzer:
    """因子有效性分析器测试"""

    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()

    @pytest.fixture
    def analyzer(self, mock_engine):
        """创建分析器实例"""
        return FactorEffectivenessAnalyzer(mock_engine)

    @pytest.fixture
    def factor_data(self):
        """因子数据"""
        return TestAnalysisDataGenerator.generate_factor_data(50, 100, 3)

    @pytest.fixture
    def return_data(self):
        """收益率数据"""
        return TestAnalysisDataGenerator.generate_return_data(50, 100)

    def test_calculate_ic(self, analyzer, factor_data, return_data):
        """测试IC计算"""
        # 合并因子和收益率数据
        merged_data = factor_data.merge(
            return_data, left_on=["ts_code", "factor_date"], right_on=["ts_code", "trade_date"], how="inner"
        )

        # 计算IC
        ic_results = analyzer.calculate_ic(
            merged_data, factor_columns=["factor_1", "factor_2"], return_columns=["return_1d", "return_5d"]
        )

        # 验证IC结果
        assert isinstance(ic_results, dict), "IC结果应为字典"
        assert "factor_1" in ic_results, "应包含factor_1的IC结果"

        # 验证IC值的合理性
        for factor_name, ic_data in ic_results.items():
            assert "ic_mean" in ic_data, "应包含IC均值"
            assert "ic_std" in ic_data, "应包含IC标准差"
            assert "ic_ir" in ic_data, "应包含IR值"

            # IC值应在合理范围内
            assert -1 <= ic_data["ic_mean"] <= 1, "IC均值应在-1到1之间"

    def test_calculate_ir(self, analyzer, factor_data, return_data):
        """测试IR计算"""
        merged_data = factor_data.merge(
            return_data, left_on=["ts_code", "factor_date"], right_on=["ts_code", "trade_date"], how="inner"
        )

        # 计算IR
        ir_results = analyzer.calculate_ir(
            merged_data, factor_columns=["factor_1"], return_column="return_1d", window=20
        )

        # 验证IR结果
        assert isinstance(ir_results, dict), "IR结果应为字典"
        assert "factor_1" in ir_results, "应包含factor_1的IR结果"

        ir_data = ir_results["factor_1"]
        assert "ir_mean" in ir_data, "应包含IR均值"
        assert "ir_std" in ir_data, "应包含IR标准差"

    def test_layered_backtest(self, analyzer, factor_data, return_data):
        """测试分层回测"""
        merged_data = factor_data.merge(
            return_data, left_on=["ts_code", "factor_date"], right_on=["ts_code", "trade_date"], how="inner"
        )

        # 进行分层回测
        backtest_results = analyzer.layered_backtest(
            merged_data, factor_column="factor_1", return_column="return_1d", layers=5
        )

        # 验证回测结果
        assert isinstance(backtest_results, dict), "回测结果应为字典"
        assert "layer_returns" in backtest_results, "应包含分层收益率"
        assert "long_short_return" in backtest_results, "应包含多空收益率"

        layer_returns = backtest_results["layer_returns"]
        assert len(layer_returns) == 5, "应有5层收益率"

    def test_factor_decay_analysis(self, analyzer, factor_data, return_data):
        """测试因子衰减分析"""
        merged_data = factor_data.merge(
            return_data, left_on=["ts_code", "factor_date"], right_on=["ts_code", "trade_date"], how="inner"
        )

        # 进行衰减分析
        decay_results = analyzer.factor_decay_analysis(
            merged_data, factor_column="factor_1", return_columns=["return_1d", "return_5d", "return_20d"]
        )

        # 验证衰减分析结果
        assert isinstance(decay_results, dict), "衰减分析结果应为字典"
        assert "decay_pattern" in decay_results, "应包含衰减模式"
        assert "half_life" in decay_results, "应包含半衰期"
        assert "optimal_holding_period" in decay_results, "应包含最优持有期"


class TestFactorCorrelationAnalyzer:
    """因子相关性分析器测试"""

    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()

    @pytest.fixture
    def analyzer(self, mock_engine):
        """创建分析器实例"""
        return FactorCorrelationAnalyzer(mock_engine)

    @pytest.fixture
    def factor_data(self):
        """因子数据"""
        return TestAnalysisDataGenerator.generate_factor_data(50, 100, 5)

    def test_calculate_correlation_matrix(self, analyzer, factor_data):
        """测试相关性矩阵计算"""
        factor_columns = ["factor_1", "factor_2", "factor_3", "factor_4", "factor_5"]

        # 计算相关性矩阵
        corr_matrix = analyzer.calculate_correlation_matrix(
            factor_data, factor_columns=factor_columns, method="pearson"
        )

        # 验证相关性矩阵
        assert isinstance(corr_matrix, pd.DataFrame), "相关性矩阵应为DataFrame"
        assert corr_matrix.shape == (5, 5), "相关性矩阵应为5x5"

        # 验证对角线为1
        for i in range(5):
            assert abs(corr_matrix.iloc[i, i] - 1) < 1e-10, "对角线元素应为1"

        # 验证矩阵对称性
        for i in range(5):
            for j in range(5):
                assert abs(corr_matrix.iloc[i, j] - corr_matrix.iloc[j, i]) < 1e-10, "相关性矩阵应对称"

    def test_rolling_correlation(self, analyzer, factor_data):
        """测试滚动相关性计算"""
        # 计算滚动相关性
        rolling_corr = analyzer.calculate_rolling_correlation(
            factor_data, factor1="factor_1", factor2="factor_2", window=20
        )

        # 验证滚动相关性结果
        assert isinstance(rolling_corr, pd.Series), "滚动相关性应为Series"
        assert len(rolling_corr) == len(factor_data["factor_date"].unique()), "长度应与日期数一致"

        # 验证相关性值的范围
        valid_corr = rolling_corr.dropna()
        assert all(-1 <= val <= 1 for val in valid_corr), "相关性值应在-1到1之间"

    def test_identify_high_correlation_factors(self, analyzer, factor_data):
        """测试高相关因子识别"""
        factor_columns = ["factor_1", "factor_2", "factor_3", "factor_4", "factor_5"]

        # 识别高相关因子
        high_corr_pairs = analyzer.identify_high_correlation_factors(
            factor_data, factor_columns=factor_columns, threshold=0.3
        )

        # 验证高相关因子识别结果
        assert isinstance(high_corr_pairs, list), "高相关因子对应为列表"

        # 验证每个因子对的相关性确实超过阈值
        for pair in high_corr_pairs:
            factor1, factor2, correlation = pair
            assert abs(correlation) >= 0.3, f"因子对 {factor1}-{factor2} 相关性应超过阈值"

    def test_factor_independence_score(self, analyzer, factor_data):
        """测试因子独立性评分"""
        factor_columns = ["factor_1", "factor_2", "factor_3"]

        # 计算独立性评分
        independence_scores = analyzer.calculate_independence_score(factor_data, factor_columns=factor_columns)

        # 验证独立性评分
        assert isinstance(independence_scores, dict), "独立性评分应为字典"

        for factor_name, score in independence_scores.items():
            assert 0 <= score <= 1, f"因子 {factor_name} 独立性评分应在0-1之间"


class TestFactorComposer:
    """因子合成器测试"""

    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()

    @pytest.fixture
    def composer(self, mock_engine):
        """创建合成器实例"""
        return FactorComposer(mock_engine)

    @pytest.fixture
    def factor_data(self):
        """因子数据"""
        return TestAnalysisDataGenerator.generate_factor_data(50, 100, 4)

    def test_equal_weight_composition(self, composer, factor_data):
        """测试等权重合成"""
        factor_columns = ["factor_1", "factor_2", "factor_3"]

        # 等权重合成
        composite_factor = composer.equal_weight_compose(factor_data, factor_columns=factor_columns)

        # 验证合成结果
        assert isinstance(composite_factor, pd.Series), "合成因子应为Series"
        assert len(composite_factor) == len(factor_data), "合成因子长度应与原数据一致"

        # 验证合成逻辑
        valid_indices = ~composite_factor.isna()
        if valid_indices.sum() > 0:
            # 合成因子应该是各因子的平均值
            manual_composite = factor_data[factor_columns].mean(axis=1)
            np.testing.assert_array_almost_equal(
                composite_factor[valid_indices], manual_composite[valid_indices], decimal=10
            )

    def test_weighted_composition(self, composer, factor_data):
        """测试加权合成"""
        factor_columns = ["factor_1", "factor_2", "factor_3"]
        weights = [0.5, 0.3, 0.2]

        # 加权合成
        composite_factor = composer.weighted_compose(factor_data, factor_columns=factor_columns, weights=weights)

        # 验证合成结果
        assert isinstance(composite_factor, pd.Series), "合成因子应为Series"
        assert len(composite_factor) == len(factor_data), "合成因子长度应与原数据一致"

        # 验证权重和为1
        assert abs(sum(weights) - 1.0) < 1e-10, "权重和应为1"

    def test_ic_weighted_composition(self, composer, factor_data):
        """测试IC加权合成"""
        factor_columns = ["factor_1", "factor_2", "factor_3"]
        return_data = TestAnalysisDataGenerator.generate_return_data(50, 100)

        # 合并数据
        merged_data = factor_data.merge(
            return_data, left_on=["ts_code", "factor_date"], right_on=["ts_code", "trade_date"], how="inner"
        )

        # IC加权合成
        composite_factor = composer.ic_weighted_compose(
            merged_data, factor_columns=factor_columns, return_column="return_1d"
        )

        # 验证合成结果
        assert isinstance(composite_factor, pd.Series), "合成因子应为Series"
        assert len(composite_factor) <= len(merged_data), "合成因子长度应不超过原数据"

    def test_pca_composition(self, composer, factor_data):
        """测试PCA合成"""
        factor_columns = ["factor_1", "factor_2", "factor_3", "factor_4"]

        # PCA合成
        composite_factor, explained_variance = composer.pca_compose(
            factor_data, factor_columns=factor_columns, n_components=1
        )

        # 验证合成结果
        assert isinstance(composite_factor, pd.Series), "合成因子应为Series"
        assert isinstance(explained_variance, float), "解释方差应为浮点数"
        assert 0 <= explained_variance <= 1, "解释方差应在0-1之间"

    def test_composition_effectiveness_validation(self, composer, factor_data):
        """测试合成有效性验证"""
        factor_columns = ["factor_1", "factor_2", "factor_3"]
        return_data = TestAnalysisDataGenerator.generate_return_data(50, 100)

        # 合并数据
        merged_data = factor_data.merge(
            return_data, left_on=["ts_code", "factor_date"], right_on=["ts_code", "trade_date"], how="inner"
        )

        # 等权重合成
        composite_factor = composer.equal_weight_compose(merged_data, factor_columns=factor_columns)

        # 验证合成有效性
        effectiveness = composer.validate_composition_effectiveness(
            merged_data, composite_factor=composite_factor, original_factors=factor_columns, return_column="return_1d"
        )

        # 验证有效性结果
        assert isinstance(effectiveness, dict), "有效性结果应为字典"
        assert "composite_ic" in effectiveness, "应包含合成因子IC"
        assert "original_ic_mean" in effectiveness, "应包含原始因子平均IC"
        assert "improvement_ratio" in effectiveness, "应包含改进比率"


class TestIntegrationAndEdgeCases:
    """集成测试和边界条件测试"""

    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_data = pd.DataFrame()

        standardizer = FactorStandardizer(Mock())

        # 空数据应该被正确处理
        result = standardizer.zscore_standardize(np.array([]))
        assert len(result) == 0, "空数据标准化结果应为空"

    def test_single_value_data(self):
        """测试单值数据处理"""
        single_value_data = np.array([5.0])

        standardizer = FactorStandardizer(Mock())

        # 单值数据标准化应该返回NaN或0
        result = standardizer.zscore_standardize(single_value_data)
        assert len(result) == 1, "单值数据标准化结果长度应为1"
        assert np.isnan(result[0]) or result[0] == 0, "单值数据标准化结果应为NaN或0"

    def test_all_same_values(self):
        """测试所有值相同的情况"""
        same_values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        standardizer = FactorStandardizer(Mock())

        # 所有值相同时标准化应该返回0或NaN
        result = standardizer.zscore_standardize(same_values)
        assert all(np.isnan(val) or val == 0 for val in result), "相同值标准化结果应为NaN或0"

    def test_missing_values_handling(self):
        """测试缺失值处理"""
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        standardizer = FactorStandardizer(Mock())

        # 包含NaN的数据应该被正确处理
        result = standardizer.zscore_standardize(data_with_nan)
        assert len(result) == len(data_with_nan), "结果长度应与输入一致"
        assert np.isnan(result[2]), "原始NaN位置应保持NaN"

        # 非NaN值应该被正确标准化
        valid_mask = ~np.isnan(data_with_nan)
        valid_result = result[valid_mask]
        assert abs(np.mean(valid_result)) < 1e-10, "有效值的均值应接近0"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
