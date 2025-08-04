#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪面因子计算单元测试
测试资金流向、市场关注度、情绪强度等指标的计算正确性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.compute.sentiment_engine import SentimentFactorEngine
from src.compute.sentiment import (
    MoneyFlowFactorCalculator, AttentionFactorCalculator,
    SentimentStrengthFactorCalculator, EventFactorCalculator
)

# 忽略pandas警告
warnings.filterwarnings('ignore', category=FutureWarning)

class TestSentimentDataGenerator:
    """情绪面测试数据生成器"""
    
    @staticmethod
    def generate_sentiment_data(length=50, seed=42):
        """生成情绪面测试数据"""
        np.random.seed(seed)
        
        dates = pd.date_range(start='2024-01-01', periods=length, freq='D')
        base_price = 100.0
        
        # 生成价格数据
        returns = np.random.normal(0, 0.02, length)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # 生成成交量数据（与价格变化相关）
        volume_base = 5000000
        volume_factor = 1 + np.abs(returns) * 10  # 价格波动大时成交量大
        volumes = volume_base * volume_factor * (1 + np.random.normal(0, 0.3, length))
        volumes = np.maximum(volumes, 100000)  # 确保最小成交量
        
        # 生成成交额数据
        amounts = prices * volumes * (1 + np.random.normal(0, 0.1, length))
        
        return pd.DataFrame({
            'ts_code': '000001.SZ',
            'trade_date': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, length)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, length))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, length))),
            'close': prices,
            'pre_close': np.concatenate([[base_price], prices[:-1]]),
            'volume': volumes.astype(int),
            'amount': amounts,
            'pct_chg': np.concatenate([[0], np.diff(prices) / prices[:-1] * 100]),
            'turnover_rate': np.random.uniform(0.5, 5.0, length)
        })
    
    @staticmethod
    def generate_event_data(length=50):
        """生成包含特殊事件的数据"""
        data = TestSentimentDataGenerator.generate_sentiment_data(length)
        
        # 添加涨停事件
        limit_up_idx = 10
        data.loc[limit_up_idx, 'pct_chg'] = 9.95
        data.loc[limit_up_idx, 'close'] = data.loc[limit_up_idx, 'pre_close'] * 1.0995
        
        # 添加跌停事件
        limit_down_idx = 20
        data.loc[limit_down_idx, 'pct_chg'] = -9.95
        data.loc[limit_down_idx, 'close'] = data.loc[limit_down_idx, 'pre_close'] * 0.9005
        
        # 添加异常成交量
        volume_spike_idx = 30
        data.loc[volume_spike_idx, 'volume'] = data['volume'].iloc[volume_spike_idx] * 5
        
        return data

class TestMoneyFlowFactorCalculator:
    """资金流向因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return MoneyFlowFactorCalculator(mock_engine)
    
    @pytest.fixture
    def sentiment_data(self):
        """情绪面数据"""
        return TestSentimentDataGenerator.generate_sentiment_data()
    
    def test_calculate_money_flow(self, calculator, sentiment_data):
        """测试资金流向计算"""
        money_flow = calculator.calculate_money_flow(sentiment_data, window=5)
        
        # 基本验证
        assert not money_flow.empty, "资金流向结果不应为空"
        assert len(money_flow) == len(sentiment_data), "资金流向长度应与输入数据一致"
        
        # 验证资金流向的合理性
        valid_flow = money_flow.dropna()
        assert len(valid_flow) > 0, "应该有有效的资金流向值"
    
    def test_calculate_net_inflow_rate(self, calculator, sentiment_data):
        """测试净流入率计算"""
        net_inflow = calculator.calculate_net_inflow_rate(sentiment_data)
        
        # 基本验证
        assert not net_inflow.empty, "净流入率结果不应为空"
        assert len(net_inflow) == len(sentiment_data), "净流入率长度应与输入数据一致"
        
        # 净流入率应在合理范围内
        valid_inflow = net_inflow.dropna()
        assert all(-100 <= val <= 100 for val in valid_inflow), "净流入率应在-100%到100%之间"
    
    def test_calculate_order_ratios(self, calculator, sentiment_data):
        """测试大中小单占比计算"""
        large_ratio = calculator.calculate_large_order_ratio(sentiment_data)
        medium_ratio = calculator.calculate_medium_order_ratio(sentiment_data)
        small_ratio = calculator.calculate_small_order_ratio(sentiment_data)
        
        # 基本验证
        assert not large_ratio.empty, "大单占比结果不应为空"
        assert not medium_ratio.empty, "中单占比结果不应为空"
        assert not small_ratio.empty, "小单占比结果不应为空"
        
        # 占比应在0-100%之间
        for ratio_data in [large_ratio, medium_ratio, small_ratio]:
            valid_ratio = ratio_data.dropna()
            assert all(0 <= val <= 100 for val in valid_ratio), "占比应在0-100%之间"

class TestAttentionFactorCalculator:
    """市场关注度因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return AttentionFactorCalculator(mock_engine)
    
    @pytest.fixture
    def sentiment_data(self):
        """情绪面数据"""
        return TestSentimentDataGenerator.generate_sentiment_data()
    
    def test_calculate_turnover_attention(self, calculator, sentiment_data):
        """测试换手率关注度计算"""
        turnover_attention = calculator.calculate_turnover_attention(sentiment_data)
        
        # 基本验证
        assert not turnover_attention.empty, "换手率关注度结果不应为空"
        assert len(turnover_attention) == len(sentiment_data), "换手率关注度长度应与输入数据一致"
        
        # 关注度应为正值
        valid_attention = turnover_attention.dropna()
        assert all(val >= 0 for val in valid_attention), "换手率关注度应为非负值"
    
    def test_calculate_volume_attention(self, calculator, sentiment_data):
        """测试成交量关注度计算"""
        volume_attention = calculator.calculate_volume_attention(sentiment_data)
        
        # 基本验证
        assert not volume_attention.empty, "成交量关注度结果不应为空"
        assert len(volume_attention) == len(sentiment_data), "成交量关注度长度应与输入数据一致"
        
        # 关注度应为正值
        valid_attention = volume_attention.dropna()
        assert all(val >= 0 for val in valid_attention), "成交量关注度应为非负值"
    
    def test_calculate_comprehensive_attention(self, calculator, sentiment_data):
        """测试综合关注度计算"""
        comprehensive_attention = calculator.calculate_comprehensive_attention(sentiment_data)
        
        # 基本验证
        assert not comprehensive_attention.empty, "综合关注度结果不应为空"
        assert len(comprehensive_attention) == len(sentiment_data), "综合关注度长度应与输入数据一致"
        
        # 综合关注度应为正值
        valid_attention = comprehensive_attention.dropna()
        assert all(val >= 0 for val in valid_attention), "综合关注度应为非负值"
    
    def test_calculate_attention_change_rate(self, calculator, sentiment_data):
        """测试关注度变化率计算"""
        change_rate = calculator.calculate_attention_change_rate(sentiment_data, window=5)
        
        # 基本验证
        assert not change_rate.empty, "关注度变化率结果不应为空"
        assert len(change_rate) == len(sentiment_data), "关注度变化率长度应与输入数据一致"
        
        # 变化率可以为正负值
        valid_change = change_rate.dropna()
        assert len(valid_change) > 0, "应该有有效的关注度变化率值"

class TestSentimentStrengthFactorCalculator:
    """情绪强度因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return SentimentStrengthFactorCalculator(mock_engine)
    
    @pytest.fixture
    def sentiment_data(self):
        """情绪面数据"""
        return TestSentimentDataGenerator.generate_sentiment_data()
    
    def test_calculate_price_momentum_sentiment(self, calculator, sentiment_data):
        """测试价格动量情绪计算"""
        momentum_sentiment = calculator.calculate_price_momentum_sentiment(sentiment_data, window=20)
        
        # 基本验证
        assert not momentum_sentiment.empty, "价格动量情绪结果不应为空"
        assert len(momentum_sentiment) == len(sentiment_data), "价格动量情绪长度应与输入数据一致"
        
        # 情绪值应在合理范围内
        valid_sentiment = momentum_sentiment.dropna()
        assert len(valid_sentiment) > 0, "应该有有效的价格动量情绪值"
    
    def test_calculate_volatility_sentiment(self, calculator, sentiment_data):
        """测试波动率情绪计算"""
        volatility_sentiment = calculator.calculate_volatility_sentiment(sentiment_data, window=20)
        
        # 基本验证
        assert not volatility_sentiment.empty, "波动率情绪结果不应为空"
        assert len(volatility_sentiment) == len(sentiment_data), "波动率情绪长度应与输入数据一致"
        
        # 波动率情绪应为正值
        valid_sentiment = volatility_sentiment.dropna()
        assert all(val >= 0 for val in valid_sentiment), "波动率情绪应为非负值"
    
    def test_calculate_bull_bear_ratio(self, calculator, sentiment_data):
        """测试看涨看跌比例计算"""
        bull_bear_ratio = calculator.calculate_bull_bear_ratio(sentiment_data)
        
        # 基本验证
        assert not bull_bear_ratio.empty, "看涨看跌比例结果不应为空"
        assert len(bull_bear_ratio) == len(sentiment_data), "看涨看跌比例长度应与输入数据一致"
        
        # 比例应为正值
        valid_ratio = bull_bear_ratio.dropna()
        assert all(val >= 0 for val in valid_ratio), "看涨看跌比例应为非负值"
    
    def test_calculate_sentiment_volatility(self, calculator, sentiment_data):
        """测试情绪波动率计算"""
        sentiment_volatility = calculator.calculate_sentiment_volatility(sentiment_data, window=20)
        
        # 基本验证
        assert not sentiment_volatility.empty, "情绪波动率结果不应为空"
        assert len(sentiment_volatility) == len(sentiment_data), "情绪波动率长度应与输入数据一致"
        
        # 波动率应为正值
        valid_volatility = sentiment_volatility.dropna()
        assert all(val >= 0 for val in valid_volatility), "情绪波动率应为非负值"

class TestEventFactorCalculator:
    """特殊事件因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return EventFactorCalculator(mock_engine)
    
    @pytest.fixture
    def event_data(self):
        """包含特殊事件的数据"""
        return TestSentimentDataGenerator.generate_event_data()
    
    def test_calculate_abnormal_volume(self, calculator, event_data):
        """测试异常成交量计算"""
        abnormal_volume = calculator.calculate_abnormal_volume(event_data, window=20)
        
        # 基本验证
        assert not abnormal_volume.empty, "异常成交量结果不应为空"
        assert len(abnormal_volume) == len(event_data), "异常成交量长度应与输入数据一致"
        
        # 应该检测到异常成交量
        valid_abnormal = abnormal_volume.dropna()
        assert any(val > 2 for val in valid_abnormal), "应该检测到异常成交量"
    
    def test_calculate_abnormal_return(self, calculator, event_data):
        """测试异常收益率计算"""
        abnormal_return = calculator.calculate_abnormal_return(event_data, window=20)
        
        # 基本验证
        assert not abnormal_return.empty, "异常收益率结果不应为空"
        assert len(abnormal_return) == len(event_data), "异常收益率长度应与输入数据一致"
        
        # 应该检测到异常收益率
        valid_abnormal = abnormal_return.dropna()
        assert len(valid_abnormal) > 0, "应该有有效的异常收益率值"
    
    def test_calculate_limit_signals(self, calculator, event_data):
        """测试涨跌停信号计算"""
        limit_up = calculator.calculate_limit_up_signal(event_data)
        limit_down = calculator.calculate_limit_down_signal(event_data)
        
        # 基本验证
        assert not limit_up.empty, "涨停信号结果不应为空"
        assert not limit_down.empty, "跌停信号结果不应为空"
        
        # 应该检测到涨跌停信号
        assert any(limit_up == 1), "应该检测到涨停信号"
        assert any(limit_down == 1), "应该检测到跌停信号"
    
    def test_calculate_gap_signals(self, calculator, event_data):
        """测试跳空信号计算"""
        gap_up = calculator.calculate_gap_up_signal(event_data)
        gap_down = calculator.calculate_gap_down_signal(event_data)
        
        # 基本验证
        assert not gap_up.empty, "向上跳空信号结果不应为空"
        assert not gap_down.empty, "向下跳空信号结果不应为空"
        
        # 信号值应为0或1
        assert all(val in [0, 1] for val in gap_up), "跳空信号应为0或1"
        assert all(val in [0, 1] for val in gap_down), "跳空信号应为0或1"
    
    def test_calculate_volume_spike(self, calculator, event_data):
        """测试成交量异动计算"""
        volume_spike = calculator.calculate_volume_spike(event_data, window=20)
        
        # 基本验证
        assert not volume_spike.empty, "成交量异动结果不应为空"
        assert len(volume_spike) == len(event_data), "成交量异动长度应与输入数据一致"
        
        # 应该检测到成交量异动
        valid_spike = volume_spike.dropna()
        assert any(val > 2 for val in valid_spike), "应该检测到成交量异动"

class TestSentimentFactorEngine:
    """情绪面因子引擎测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def sentiment_engine(self, mock_engine):
        """创建情绪面因子引擎"""
        return SentimentFactorEngine(mock_engine)
    
    def test_engine_initialization(self, sentiment_engine):
        """测试引擎初始化"""
        assert sentiment_engine is not None
        assert hasattr(sentiment_engine, 'money_flow_calculator')
        assert hasattr(sentiment_engine, 'attention_calculator')
        assert hasattr(sentiment_engine, 'sentiment_strength_calculator')
        assert hasattr(sentiment_engine, 'event_calculator')
    
    def test_calculate_all_factors(self, sentiment_engine):
        """测试计算所有情绪面因子"""
        sentiment_data = TestSentimentDataGenerator.generate_sentiment_data()
        
        # 模拟数据库查询
        with patch.object(sentiment_engine, '_get_stock_data', return_value=sentiment_data):
            factors = sentiment_engine.calculate_all_factors(['000001.SZ'])
        
        # 验证返回的因子
        assert isinstance(factors, pd.DataFrame)
        assert not factors.empty
        assert 'ts_code' in factors.columns
        
        # 验证包含主要情绪面指标
        expected_factors = ['money_flow_5', 'net_inflow_rate', 'comprehensive_attention', 'abnormal_volume']
        for factor in expected_factors:
            if factor in factors.columns:
                # 至少应该有一些非NaN值
                assert not factors[factor].isna().all(), f"因子 {factor} 不应全为NaN"

class TestDataQualityAndEdgeCases:
    """数据质量和边界条件测试"""
    
    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        # 创建包含缺失值的数据
        data = TestSentimentDataGenerator.generate_sentiment_data(20)
        data.loc[5:10, 'volume'] = np.nan
        data.loc[15:18, 'amount'] = np.nan
        
        calculator = MoneyFlowFactorCalculator(Mock())
        
        # 计算应该能处理缺失值
        money_flow = calculator.calculate_money_flow(data, window=5)
        assert not money_flow.empty, "应该能处理缺失数据"
    
    def test_extreme_values_handling(self):
        """测试极值处理"""
        data = TestSentimentDataGenerator.generate_sentiment_data(20)
        
        # 添加极值
        data.loc[10, 'volume'] = 1e12  # 极大成交量
        data.loc[11, 'pct_chg'] = 50   # 极大涨幅
        
        calculator = AttentionFactorCalculator(Mock())
        
        # 计算应该能处理极值
        attention = calculator.calculate_comprehensive_attention(data)
        valid_attention = attention.dropna()
        
        # 极值应该被合理处理，不应该产生无穷大或NaN
        assert all(np.isfinite(val) for val in valid_attention), "应该能处理极值"
    
    def test_zero_volume_handling(self):
        """测试零成交量处理"""
        data = TestSentimentDataGenerator.generate_sentiment_data(20)
        data.loc[10, 'volume'] = 0
        data.loc[10, 'amount'] = 0
        
        calculator = MoneyFlowFactorCalculator(Mock())
        
        # 零成交量应该被合理处理
        money_flow = calculator.calculate_money_flow(data, window=5)
        assert not money_flow.empty, "应该能处理零成交量"

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])