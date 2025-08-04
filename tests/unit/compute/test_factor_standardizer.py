#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子标准化和评分机制测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.compute.factor_standardizer import (
    FactorStandardizer, IndustryFactorScorer, FactorQuantileCalculator,
    FactorComposer, FactorScoreManager, StandardizationMethod, OutlierTreatment
)


class TestFactorStandardizer:
    """因子标准化器测试"""
    
    @pytest.fixture
    def sample_factor_data(self):
        """创建测试因子数据"""
        np.random.seed(42)
        return pd.Series(np.random.normal(0, 1, 100))
    
    @pytest.fixture
    def outlier_factor_data(self):
        """创建包含异常值的因子数据"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        # 添加异常值
        data[0] = 10
        data[1] = -10
        return pd.Series(data)
    
    def test_zscore_standardize(self, sample_factor_data):
        """测试Z-score标准化"""
        standardizer = FactorStandardizer()
        
        standardized = standardizer.zscore_standardize(sample_factor_data)
        
        assert isinstance(standardized, pd.Series)
        assert len(standardized) == len(sample_factor_data)
        
        # 检查标准化后的均值和标准差
        assert abs(standardized.mean()) < 1e-10  # 均值应该接近0
        assert abs(standardized.std() - 1) < 1e-10  # 标准差应该接近1
    
    def test_zscore_standardize_with_outliers(self, outlier_factor_data):
        """测试带异常值处理的Z-score标准化"""
        standardizer = FactorStandardizer()
        
        # 不处理异常值
        standardized_no_clip = standardizer.zscore_standardize(
            outlier_factor_data, 
            outlier_treatment=OutlierTreatment.NONE
        )
        
        # 截尾处理异常值
        standardized_clip = standardizer.zscore_standardize(
            outlier_factor_data,
            outlier_treatment=OutlierTreatment.CLIP,
            clip_quantiles=(0.05, 0.95)
        )
        
        # 截尾处理后的数据应该更稳定
        assert standardized_clip.std() < standardized_no_clip.std()
    
    def test_quantile_standardize(self, sample_factor_data):
        """测试分位数标准化"""
        standardizer = FactorStandardizer()
        
        standardized = standardizer.quantile_standardize(sample_factor_data)
        
        assert isinstance(standardized, pd.Series)
        assert len(standardized) == len(sample_factor_data)
        
        # 检查分位数标准化后的范围
        valid_values = standardized.dropna()
        assert valid_values.min() >= 0
        assert valid_values.max() <= 1
    
    def test_rank_standardize(self, sample_factor_data):
        """测试排名标准化"""
        standardizer = FactorStandardizer()
        
        standardized = standardizer.rank_standardize(sample_factor_data)
        
        assert isinstance(standardized, pd.Series)
        assert len(standardized) == len(sample_factor_data)
        
        # 检查排名标准化后的范围
        valid_values = standardized.dropna()
        if not valid_values.empty:
            assert valid_values.min() >= 0
            assert valid_values.max() <= 1
    
    def test_minmax_standardize(self, sample_factor_data):
        """测试最小最大值标准化"""
        standardizer = FactorStandardizer()
        
        standardized = standardizer.minmax_standardize(sample_factor_data)
        
        assert isinstance(standardized, pd.Series)
        assert len(standardized) == len(sample_factor_data)
        
        # 检查标准化后的范围
        valid_values = standardized.dropna()
        if not valid_values.empty:
            assert valid_values.min() >= 0
            assert valid_values.max() <= 1
    
    def test_robust_standardize(self, outlier_factor_data):
        """测试鲁棒标准化"""
        standardizer = FactorStandardizer()
        
        standardized = standardizer.robust_standardize(outlier_factor_data)
        
        assert isinstance(standardized, pd.Series)
        assert len(standardized) == len(outlier_factor_data)
        
        # 鲁棒标准化应该对异常值不敏感
        median_val = standardized.median()
        assert abs(median_val) < 1  # 中位数应该接近0
    
    def test_standardize_unified_interface(self, sample_factor_data):
        """测试统一标准化接口"""
        standardizer = FactorStandardizer()
        
        # 测试不同方法
        methods = [
            StandardizationMethod.ZSCORE,
            StandardizationMethod.QUANTILE,
            StandardizationMethod.RANK,
            StandardizationMethod.MINMAX,
            StandardizationMethod.ROBUST
        ]
        
        for method in methods:
            standardized = standardizer.standardize(sample_factor_data, method=method)
            assert isinstance(standardized, pd.Series)
            assert len(standardized) == len(sample_factor_data)
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        standardizer = FactorStandardizer()
        
        empty_data = pd.Series(dtype=float)
        result = standardizer.zscore_standardize(empty_data)
        
        assert result.empty
    
    def test_all_nan_data_handling(self):
        """测试全NaN数据处理"""
        standardizer = FactorStandardizer()
        
        nan_data = pd.Series([np.nan] * 10)
        result = standardizer.zscore_standardize(nan_data)
        
        assert result.isna().all()


class TestIndustryFactorScorer:
    """行业内因子评分器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def sample_factor_data(self):
        """创建测试因子数据"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'],
            'factor_value': [1.5, 2.0, -0.5, 1.0, 0.5]
        })
    
    @pytest.fixture
    def sample_industry_mapping(self):
        """创建测试行业映射"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'],
            'industry': ['银行', '银行', '银行', '银行', '电子'],
            'area': ['深圳', '深圳', '上海', '上海', '深圳']
        })
    
    def test_industry_standardize(self, mock_engine, sample_factor_data, sample_industry_mapping):
        """测试行业内标准化"""
        scorer = IndustryFactorScorer(mock_engine)
        
        # Mock获取行业映射
        with patch.object(scorer, 'get_industry_mapping', return_value=sample_industry_mapping):
            result = scorer.industry_standardize(sample_factor_data)
        
        assert 'standardized_value' in result.columns
        assert len(result) == len(sample_factor_data)
        
        # 检查银行行业内的标准化结果
        bank_stocks = result[result['ts_code'].isin(['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'])]
        bank_standardized = bank_stocks['standardized_value']
        
        # 行业内标准化后均值应该接近0
        if len(bank_standardized.dropna()) > 1:
            assert abs(bank_standardized.mean()) < 0.1
    
    def test_industry_rank(self, mock_engine, sample_factor_data, sample_industry_mapping):
        """测试行业内排名"""
        scorer = IndustryFactorScorer(mock_engine)
        
        # Mock获取行业映射
        with patch.object(scorer, 'get_industry_mapping', return_value=sample_industry_mapping):
            result = scorer.industry_rank(sample_factor_data)
        
        expected_cols = ['industry_rank', 'industry_rank_pct', 'industry_stock_count']
        for col in expected_cols:
            assert col in result.columns
        
        # 检查排名百分比范围
        rank_pct = result['industry_rank_pct'].dropna()
        if not rank_pct.empty:
            assert rank_pct.min() >= 0
            assert rank_pct.max() <= 1
    
    def test_industry_neutralize(self, mock_engine, sample_factor_data, sample_industry_mapping):
        """测试行业中性化"""
        scorer = IndustryFactorScorer(mock_engine)
        
        # Mock获取行业映射
        with patch.object(scorer, 'get_industry_mapping', return_value=sample_industry_mapping):
            result = scorer.industry_neutralize(sample_factor_data)
        
        assert 'neutralized_value' in result.columns
        assert len(result) == len(sample_factor_data)
        
        # 行业中性化后，行业内均值应该接近0
        merged_result = result.merge(sample_industry_mapping, on='ts_code')
        for industry, group in merged_result.groupby('industry'):
            if len(group) > 1:
                industry_mean = group['neutralized_value'].mean()
                assert abs(industry_mean) < 1e-10


class TestFactorQuantileCalculator:
    """因子分位数计算器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def sample_factor_data(self):
        """创建测试因子数据"""
        np.random.seed(42)
        return pd.Series(np.random.normal(0, 1, 100))
    
    def test_calculate_market_quantiles(self, mock_engine, sample_factor_data):
        """测试全市场分位数计算"""
        calculator = FactorQuantileCalculator(mock_engine)
        
        result = calculator.calculate_market_quantiles(sample_factor_data, n_quantiles=10)
        
        assert isinstance(result, pd.DataFrame)
        assert 'factor_value' in result.columns
        assert 'quantile' in result.columns
        assert 'quantile_rank' in result.columns
        
        # 检查分位数统计信息
        assert hasattr(result, 'quantile_stats')
        assert isinstance(result.quantile_stats, pd.DataFrame)
        assert len(result.quantile_stats) <= 10
    
    def test_update_dynamic_quantiles(self, mock_engine, sample_factor_data):
        """测试动态分位数更新"""
        calculator = FactorQuantileCalculator(mock_engine)
        
        # 计算初始分位数
        initial_quantiles = calculator.calculate_market_quantiles(sample_factor_data)
        
        # 生成新数据
        np.random.seed(43)
        new_data = pd.Series(np.random.normal(0.5, 1, 100))
        
        # 动态更新
        updated_quantiles = calculator.update_dynamic_quantiles(
            new_data, initial_quantiles, decay_factor=0.9
        )
        
        assert isinstance(updated_quantiles, pd.DataFrame)
        assert hasattr(updated_quantiles, 'quantile_stats')
        
        # 更新后的分位数应该与初始分位数不同
        initial_stats = initial_quantiles.quantile_stats
        updated_stats = updated_quantiles.quantile_stats
        
        if not initial_stats.empty and not updated_stats.empty:
            # 至少有一个统计值应该发生变化
            mean_diff = abs(updated_stats['mean_value'].iloc[0] - initial_stats['mean_value'].iloc[0])
            assert mean_diff > 0


class TestFactorComposer:
    """因子合成器测试"""
    
    @pytest.fixture
    def sample_multi_factor_data(self):
        """创建多因子测试数据"""
        np.random.seed(42)
        return pd.DataFrame({
            'factor1': np.random.normal(0, 1, 50),
            'factor2': np.random.normal(0.5, 1.5, 50),
            'factor3': np.random.normal(-0.2, 0.8, 50)
        })
    
    def test_equal_weight_compose(self, sample_multi_factor_data):
        """测试等权重因子合成"""
        composer = FactorComposer()
        
        factor_cols = ['factor1', 'factor2', 'factor3']
        composite = composer.equal_weight_compose(sample_multi_factor_data, factor_cols)
        
        assert isinstance(composite, pd.Series)
        assert len(composite) == len(sample_multi_factor_data)
        
        # 等权重合成应该是各因子的平均值
        expected_composite = sample_multi_factor_data[factor_cols].mean(axis=1)
        pd.testing.assert_series_equal(composite, expected_composite, check_names=False)
    
    def test_weighted_compose(self, sample_multi_factor_data):
        """测试加权因子合成"""
        composer = FactorComposer()
        
        factor_weights = {'factor1': 0.5, 'factor2': 0.3, 'factor3': 0.2}
        composite = composer.weighted_compose(sample_multi_factor_data, factor_weights)
        
        assert isinstance(composite, pd.Series)
        assert len(composite) == len(sample_multi_factor_data)
        
        # 验证加权合成结果
        expected_composite = (
            sample_multi_factor_data['factor1'] * 0.5 +
            sample_multi_factor_data['factor2'] * 0.3 +
            sample_multi_factor_data['factor3'] * 0.2
        )
        pd.testing.assert_series_equal(composite, expected_composite, check_names=False)
    
    def test_ic_weighted_compose(self, sample_multi_factor_data):
        """测试基于IC的加权合成"""
        composer = FactorComposer()
        
        factor_cols = ['factor1', 'factor2', 'factor3']
        ic_values = {'factor1': 0.15, 'factor2': -0.08, 'factor3': 0.12}
        
        composite = composer.ic_weighted_compose(
            sample_multi_factor_data, factor_cols, ic_values
        )
        
        assert isinstance(composite, pd.Series)
        assert len(composite) == len(sample_multi_factor_data)
        
        # IC加权应该使用IC绝对值作为权重
        total_ic_weight = abs(0.15) + abs(-0.08) + abs(0.12)
        expected_composite = (
            sample_multi_factor_data['factor1'] * (abs(0.15) / total_ic_weight) +
            sample_multi_factor_data['factor2'] * (abs(-0.08) / total_ic_weight) +
            sample_multi_factor_data['factor3'] * (abs(0.12) / total_ic_weight)
        )
        pd.testing.assert_series_equal(composite, expected_composite, check_names=False)
    
    def test_validate_composition(self, sample_multi_factor_data):
        """测试因子合成验证"""
        composer = FactorComposer()
        
        factor_cols = ['factor1', 'factor2', 'factor3']
        composite = composer.equal_weight_compose(sample_multi_factor_data, factor_cols)
        
        validation = composer.validate_composition(
            sample_multi_factor_data, composite, factor_cols
        )
        
        assert isinstance(validation, dict)
        assert 'correlations' in validation
        assert 'avg_correlation' in validation
        assert 'composite_stats' in validation
        
        # 检查相关性
        correlations = validation['correlations']
        for factor_col in factor_cols:
            assert factor_col in correlations
            assert isinstance(correlations[factor_col], float)
    
    def test_pca_compose(self, sample_multi_factor_data):
        """测试PCA因子合成"""
        composer = FactorComposer()
        
        factor_cols = ['factor1', 'factor2', 'factor3']
        
        try:
            composite = composer.pca_compose(sample_multi_factor_data, factor_cols)
            
            assert isinstance(composite, pd.Series)
            assert len(composite) == len(sample_multi_factor_data)
            
        except ImportError:
            # 如果没有安装scikit-learn，应该回退到等权重合成
            composite = composer.pca_compose(sample_multi_factor_data, factor_cols)
            expected_composite = composer.equal_weight_compose(sample_multi_factor_data, factor_cols)
            pd.testing.assert_series_equal(composite, expected_composite, check_names=False)


class TestFactorScoreManager:
    """因子评分历史管理器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def sample_scores_data(self):
        """创建测试评分数据"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'],
            'standardized_value': [1.5, -0.5, 0.2],
            'quantile': [8, 3, 5],
            'industry': ['银行', '银行', '银行']
        })
    
    def test_save_factor_scores(self, mock_engine, sample_scores_data):
        """测试保存因子评分"""
        manager = FactorScoreManager(mock_engine)
        
        # Mock to_sql方法
        with patch.object(sample_scores_data, 'to_sql') as mock_to_sql:
            result = manager.save_factor_scores(
                sample_scores_data,
                date(2024, 1, 1),
                'test_factor',
                'standardized'
            )
            
            assert result is True
            mock_to_sql.assert_called_once()
    
    def test_analyze_score_stability(self, mock_engine):
        """测试评分稳定性分析"""
        manager = FactorScoreManager(mock_engine)
        
        # Mock历史数据
        mock_historical_data = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 10 + ['000002.SZ'] * 10,
            'score_date': [date(2024, 1, i) for i in range(1, 11)] * 2,
            'standardized_value': np.random.normal(0, 0.5, 20),
            'factor_name': ['test_factor'] * 20
        })
        
        with patch.object(manager, 'load_factor_scores', return_value=mock_historical_data):
            stability = manager.analyze_score_stability('test_factor')
            
            assert isinstance(stability, dict)
            if 'overall_stability' in stability:
                assert 'avg_std' in stability['overall_stability']
                assert 'avg_cv' in stability['overall_stability']
                assert 'stable_stocks_ratio' in stability['overall_stability']


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])