#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子有效性检验分析器测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.compute.factor_effectiveness_analyzer import (
    FactorEffectivenessAnalyzer, FactorCorrelationAnalyzer, 
    FactorEffectivenessReporter, ReturnPeriod
)


class TestFactorEffectivenessAnalyzer:
    """因子有效性分析器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def sample_factor_data(self):
        """创建测试因子数据"""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        ts_codes = ['000001.SZ', '000002.SZ', '600000.SH']
        
        data = []
        for date in dates:
            for ts_code in ts_codes:
                data.append({
                    'ts_code': ts_code,
                    'trade_date': date,
                    'factor_value': np.random.normal(0, 1)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_return_data(self):
        """创建测试收益率数据"""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        ts_codes = ['000001.SZ', '000002.SZ', '600000.SH']
        
        data = []
        for date in dates:
            for ts_code in ts_codes:
                data.append({
                    'ts_code': ts_code,
                    'trade_date': date,
                    'return_rate': np.random.normal(0, 0.02)
                })
        
        return pd.DataFrame(data)
    
    def test_calculate_ic(self, mock_engine, sample_factor_data, sample_return_data):
        """测试IC计算"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        ic_data = analyzer.calculate_ic(sample_factor_data, sample_return_data)
        
        assert isinstance(ic_data, pd.DataFrame)
        if not ic_data.empty:
            expected_cols = ['trade_date', 'ic_pearson', 'ic_spearman', 
                           'p_value_pearson', 'p_value_spearman', 'sample_count']
            for col in expected_cols:
                assert col in ic_data.columns
            
            # 检查IC值范围
            assert ic_data['ic_pearson'].abs().max() <= 1
            assert ic_data['ic_spearman'].abs().max() <= 1
            
            # 检查p值范围
            assert ic_data['p_value_pearson'].min() >= 0
            assert ic_data['p_value_pearson'].max() <= 1
    
    def test_calculate_ic_statistics(self, mock_engine, sample_factor_data, sample_return_data):
        """测试IC统计指标计算"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        ic_data = analyzer.calculate_ic(sample_factor_data, sample_return_data)
        ic_stats = analyzer.calculate_ic_statistics(ic_data)
        
        if ic_stats:
            assert isinstance(ic_stats, dict)
            
            if 'ic_pearson' in ic_stats:
                pearson_stats = ic_stats['ic_pearson']
                expected_keys = ['mean', 'std', 'abs_mean', 'positive_ratio', 
                               'significant_ratio', 'max', 'min']
                for key in expected_keys:
                    assert key in pearson_stats
                
                # 检查统计值的合理性
                assert 0 <= pearson_stats['positive_ratio'] <= 1
                assert 0 <= pearson_stats['significant_ratio'] <= 1
    
    def test_calculate_ir(self, mock_engine, sample_factor_data, sample_return_data):
        """测试IR计算"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        ic_data = analyzer.calculate_ic(sample_factor_data, sample_return_data)
        
        if not ic_data.empty:
            ir_data = analyzer.calculate_ir(ic_data, window=5)
            
            assert isinstance(ir_data, pd.DataFrame)
            if not ir_data.empty:
                expected_cols = ['ic_mean', 'ic_std', 'ir', 'ic_cumsum', 
                               'ic_t_stat', 'ir_ci_lower', 'ir_ci_upper']
                for col in expected_cols:
                    assert col in ir_data.columns
    
    def test_calculate_ir_statistics(self, mock_engine, sample_factor_data, sample_return_data):
        """测试IR统计指标计算"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        ic_data = analyzer.calculate_ic(sample_factor_data, sample_return_data)
        
        if not ic_data.empty:
            ir_data = analyzer.calculate_ir(ic_data)
            ir_stats = analyzer.calculate_ir_statistics(ir_data)
            
            if ir_stats:
                expected_keys = ['ir_mean', 'ir_std', 'ir_positive_ratio', 
                               'ir_significant_ratio', 'ir_stability']
                for key in expected_keys:
                    assert key in ir_stats
                
                # 检查统计值的合理性
                assert 0 <= ir_stats['ir_positive_ratio'] <= 1
                assert 0 <= ir_stats['ir_significant_ratio'] <= 1
    
    def test_factor_layered_backtest(self, mock_engine, sample_factor_data, sample_return_data):
        """测试因子分层回测"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        layered_data = analyzer.factor_layered_backtest(
            sample_factor_data, sample_return_data, n_layers=5
        )
        
        assert isinstance(layered_data, pd.DataFrame)
        if not layered_data.empty:
            expected_cols = ['factor_quantile', 'mean', 'std', 'count', 'median', 'trade_date']
            for col in expected_cols:
                assert col in layered_data.columns
            
            # 检查分层数量
            unique_quantiles = layered_data['factor_quantile'].unique()
            assert len(unique_quantiles) <= 5
    
    def test_analyze_layered_performance(self, mock_engine, sample_factor_data, sample_return_data):
        """测试分层表现分析"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        layered_data = analyzer.factor_layered_backtest(
            sample_factor_data, sample_return_data, n_layers=5
        )
        
        if not layered_data.empty:
            performance = analyzer.analyze_layered_performance(layered_data)
            
            assert isinstance(performance, dict)
            if 'layer_performance' in performance:
                assert isinstance(performance['layer_performance'], list)
            
            if 'long_short' in performance:
                long_short = performance['long_short']
                expected_keys = ['return', 'volatility', 'sharpe_ratio']
                for key in expected_keys:
                    assert key in long_short
    
    def test_calculate_factor_decay(self, mock_engine, sample_factor_data, sample_return_data):
        """测试因子衰减分析"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        decay_data = analyzer.calculate_factor_decay(
            sample_factor_data, sample_return_data, max_periods=5
        )
        
        assert isinstance(decay_data, pd.DataFrame)
        if not decay_data.empty:
            expected_cols = ['period', 'avg_ic_pearson', 'avg_ic_spearman', 
                           'ic_std_pearson', 'ic_std_spearman']
            for col in expected_cols:
                assert col in decay_data.columns
            
            # 检查期数范围
            assert decay_data['period'].min() >= 1
            assert decay_data['period'].max() <= 5
    
    def test_analyze_factor_decay(self, mock_engine):
        """测试因子衰减特征分析"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        # 创建模拟衰减数据
        decay_data = pd.DataFrame({
            'period': range(1, 11),
            'abs_avg_ic_pearson': [0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001],
            'abs_avg_ic_spearman': [0.12, 0.09, 0.07, 0.055, 0.045, 0.035, 0.025, 0.015, 0.008, 0.002],
            'ic_std_pearson': [0.05] * 10
        })
        
        decay_analysis = analyzer.analyze_factor_decay(decay_data)
        
        assert isinstance(decay_analysis, dict)
        if 'optimal_period_pearson' in decay_analysis:
            optimal = decay_analysis['optimal_period_pearson']
            assert 'period' in optimal
            assert 'max_abs_ic' in optimal
        
        if 'decay_pattern' in decay_analysis:
            pattern = decay_analysis['decay_pattern']
            assert 'pattern' in pattern
            assert 'correlation' in pattern
    
    def test_empty_data_handling(self, mock_engine):
        """测试空数据处理"""
        analyzer = FactorEffectivenessAnalyzer(mock_engine)
        
        empty_df = pd.DataFrame()
        
        # 测试各个方法对空数据的处理
        ic_data = analyzer.calculate_ic(empty_df, empty_df)
        assert ic_data.empty
        
        ic_stats = analyzer.calculate_ic_statistics(empty_df)
        assert ic_stats == {}
        
        layered_data = analyzer.factor_layered_backtest(empty_df, empty_df)
        assert layered_data.empty


class TestFactorCorrelationAnalyzer:
    """因子相关性分析器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def sample_multi_factor_data(self):
        """创建多因子测试数据"""
        np.random.seed(42)
        n_samples = 100
        
        # 创建相关的因子数据
        factor1 = np.random.normal(0, 1, n_samples)
        factor2 = 0.7 * factor1 + 0.3 * np.random.normal(0, 1, n_samples)  # 与factor1高相关
        factor3 = np.random.normal(0, 1, n_samples)  # 独立因子
        
        return pd.DataFrame({
            'factor1': factor1,
            'factor2': factor2,
            'factor3': factor3,
            'trade_date': pd.date_range('2024-01-01', periods=n_samples, freq='D')
        })
    
    def test_calculate_factor_correlation_matrix(self, mock_engine, sample_multi_factor_data):
        """测试因子相关性矩阵计算"""
        analyzer = FactorCorrelationAnalyzer(mock_engine)
        
        factor_cols = ['factor1', 'factor2', 'factor3']
        corr_matrix = analyzer.calculate_factor_correlation_matrix(
            sample_multi_factor_data, factor_cols
        )
        
        assert isinstance(corr_matrix, pd.DataFrame)
        if not corr_matrix.empty:
            # 检查矩阵形状
            assert corr_matrix.shape == (3, 3)
            
            # 检查对角线为1
            for i in range(3):
                assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-10
            
            # 检查对称性
            for i in range(3):
                for j in range(3):
                    assert abs(corr_matrix.iloc[i, j] - corr_matrix.iloc[j, i]) < 1e-10
    
    def test_identify_high_correlation_factors(self, mock_engine, sample_multi_factor_data):
        """测试高相关因子识别"""
        analyzer = FactorCorrelationAnalyzer(mock_engine)
        
        factor_cols = ['factor1', 'factor2', 'factor3']
        corr_matrix = analyzer.calculate_factor_correlation_matrix(
            sample_multi_factor_data, factor_cols
        )
        
        if not corr_matrix.empty:
            high_corr_pairs = analyzer.identify_high_correlation_factors(
                corr_matrix, threshold=0.5
            )
            
            assert isinstance(high_corr_pairs, list)
            
            # 应该识别出factor1和factor2的高相关性
            if high_corr_pairs:
                for factor1, factor2, corr_value in high_corr_pairs:
                    assert abs(corr_value) >= 0.5
                    assert factor1 != factor2
    
    def test_calculate_factor_independence(self, mock_engine, sample_multi_factor_data):
        """测试因子独立性计算"""
        analyzer = FactorCorrelationAnalyzer(mock_engine)
        
        factor_cols = ['factor1', 'factor2', 'factor3']
        independence_scores = analyzer.calculate_factor_independence(
            sample_multi_factor_data, factor_cols
        )
        
        assert isinstance(independence_scores, dict)
        
        for factor in factor_cols:
            if factor in independence_scores:
                score = independence_scores[factor]
                assert 0 <= score <= 1
        
        # factor3应该有更高的独立性评分
        if 'factor3' in independence_scores and 'factor2' in independence_scores:
            assert independence_scores['factor3'] > independence_scores['factor2']
    
    def test_analyze_correlation_changes(self, mock_engine, sample_multi_factor_data):
        """测试相关性时间变化分析"""
        analyzer = FactorCorrelationAnalyzer(mock_engine)
        
        factor_cols = ['factor1', 'factor2', 'factor3']
        correlation_changes = analyzer.analyze_correlation_changes(
            sample_multi_factor_data, factor_cols, window=20
        )
        
        assert isinstance(correlation_changes, pd.DataFrame)
        if not correlation_changes.empty:
            expected_cols = ['trade_date', 'factor1', 'factor2', 'correlation']
            for col in expected_cols:
                assert col in correlation_changes.columns
            
            # 检查相关系数范围
            assert correlation_changes['correlation'].abs().max() <= 1


class TestFactorEffectivenessReporter:
    """因子有效性报告生成器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def sample_factor_data(self):
        """创建测试因子数据"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        ts_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        
        data = []
        for date in dates:
            for ts_code in ts_codes:
                data.append({
                    'ts_code': ts_code,
                    'trade_date': date,
                    'factor_value': np.random.normal(0, 1)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_return_data(self):
        """创建测试收益率数据"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        ts_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        
        data = []
        for date in dates:
            for ts_code in ts_codes:
                data.append({
                    'ts_code': ts_code,
                    'trade_date': date,
                    'return_rate': np.random.normal(0, 0.02)
                })
        
        return pd.DataFrame(data)
    
    def test_generate_comprehensive_report(self, mock_engine, sample_factor_data, sample_return_data):
        """测试综合报告生成"""
        reporter = FactorEffectivenessReporter(mock_engine)
        
        report = reporter.generate_comprehensive_report(
            sample_factor_data, sample_return_data, 'test_factor'
        )
        
        assert isinstance(report, dict)
        
        # 检查报告结构
        expected_sections = [
            'factor_name', 'analysis_date', 'data_summary',
            'ic_analysis', 'ir_analysis', 'layered_analysis',
            'decay_analysis', 'overall_rating'
        ]
        
        for section in expected_sections:
            assert section in report
        
        # 检查因子名称
        assert report['factor_name'] == 'test_factor'
        
        # 检查数据概览
        data_summary = report['data_summary']
        if 'factor_data' in data_summary:
            assert 'total_records' in data_summary['factor_data']
            assert 'unique_stocks' in data_summary['factor_data']
    
    def test_calculate_overall_rating(self, mock_engine):
        """测试综合评级计算"""
        reporter = FactorEffectivenessReporter(mock_engine)
        
        # 创建模拟报告数据
        mock_report = {
            'ic_analysis': {
                'ic_statistics': {
                    'ic_pearson': {
                        'mean': 0.05,
                        'positive_ratio': 0.7,
                        'significant_ratio': 0.6
                    }
                }
            },
            'ir_analysis': {
                'ir_statistics': {
                    'final_ir': 1.2,
                    'ir_positive_ratio': 0.8
                }
            },
            'layered_analysis': {
                'long_short': {
                    'sharpe_ratio': 1.5
                },
                'monotonicity': {
                    'is_monotonic': True,
                    'correlation': 0.8
                }
            }
        }
        
        rating = reporter._calculate_overall_rating(mock_report)
        
        assert isinstance(rating, dict)
        assert 'score' in rating
        assert 'grade' in rating
        assert 'components' in rating
        assert 'strengths' in rating
        assert 'weaknesses' in rating
        
        # 检查评分范围
        assert 0 <= rating['score'] <= 100
        
        # 检查评级等级
        assert rating['grade'] in ['A', 'B', 'C', 'D', 'F']
    
    def test_rank_factors(self, mock_engine):
        """测试因子排名"""
        reporter = FactorEffectivenessReporter(mock_engine)
        
        # 创建模拟因子报告
        factor_reports = [
            {
                'factor_name': 'factor1',
                'overall_rating': {'score': 85, 'grade': 'A'}
            },
            {
                'factor_name': 'factor2',
                'overall_rating': {'score': 70, 'grade': 'B'}
            },
            {
                'factor_name': 'factor3',
                'overall_rating': {'score': 90, 'grade': 'A'}
            }
        ]
        
        ranked_factors = reporter.rank_factors(factor_reports)
        
        assert isinstance(ranked_factors, list)
        assert len(ranked_factors) == 3
        
        # 检查排名顺序（应该按评分降序）
        assert ranked_factors[0]['factor_name'] == 'factor3'  # 最高分
        assert ranked_factors[1]['factor_name'] == 'factor1'
        assert ranked_factors[2]['factor_name'] == 'factor2'  # 最低分
        
        # 检查排名字段
        for i, factor in enumerate(ranked_factors):
            assert factor['rank'] == i + 1
    
    def test_generate_factor_recommendation(self, mock_engine):
        """测试因子推荐生成"""
        reporter = FactorEffectivenessReporter(mock_engine)
        
        # 创建排名后的因子数据
        ranked_factors = [
            {
                'factor_name': 'factor1',
                'rank': 1,
                'score': 90,
                'grade': 'A',
                'report': {
                    'overall_rating': {
                        'strengths': ['IC均值较高', 'IR值较高', '单调性良好']
                    }
                }
            },
            {
                'factor_name': 'factor2',
                'rank': 2,
                'score': 75,
                'grade': 'B',
                'report': {
                    'overall_rating': {
                        'strengths': ['稳定性较好']
                    }
                }
            }
        ]
        
        recommendation = reporter.generate_factor_recommendation(ranked_factors, top_n=2)
        
        assert isinstance(recommendation, dict)
        assert 'recommended_factors' in recommendation
        assert 'summary' in recommendation
        assert 'usage_suggestions' in recommendation
        
        # 检查推荐因子
        recommended = recommendation['recommended_factors']
        assert len(recommended) == 2
        assert recommended[0]['factor_name'] == 'factor1'
        
        # 检查总结
        summary = recommendation['summary']
        assert 'total_factors' in summary
        assert 'average_score' in summary
        assert 'best_factor' in summary


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])