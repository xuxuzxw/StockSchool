#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术面因子计算单元测试
测试所有技术指标的数学正确性、边界条件和异常处理
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

from src.compute.technical_engine import TechnicalFactorEngine
from src.compute.technical import (
    MomentumFactorCalculator, TrendFactorCalculator, 
    VolatilityFactorCalculator, VolumeFactorCalculator
)

# 忽略pandas警告
warnings.filterwarnings('ignore', category=FutureWarning)

class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_stock_data(length=100, start_price=100.0, volatility=0.02, seed=42):
        """
        生成模拟股票数据
        
        Args:
            length: 数据长度
            start_price: 起始价格
            volatility: 波动率
            seed: 随机种子
            
        Returns:
            pd.DataFrame: 股票数据
        """
        np.random.seed(seed)
        
        # 生成日期序列
        dates = pd.date_range(start='2024-01-01', periods=length, freq='D')
        
        # 使用几何布朗运动生成价格
        returns = np.random.normal(0, volatility, length)
        prices = start_price * np.exp(np.cumsum(returns))
        
        # 生成OHLC数据
        high_factor = 1 + np.abs(np.random.normal(0, 0.01, length))
        low_factor = 1 - np.abs(np.random.normal(0, 0.01, length))
        
        data = pd.DataFrame({
            'ts_code': '000001.SZ',
            'trade_date': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, length)),
            'high': prices * high_factor,
            'low': prices * low_factor,
            'close': prices,
            'pre_close': np.concatenate([[start_price], prices[:-1]]),
            'volume': np.random.randint(1000000, 10000000, length),
            'amount': prices * np.random.randint(1000000, 10000000, length)
        })
        
        # 确保OHLC关系正确
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
        
        return data
    
    @staticmethod
    def generate_trending_data(length=100, trend=0.001, seed=42):
        """生成有趋势的数据"""
        np.random.seed(seed)
        
        dates = pd.date_range(start='2024-01-01', periods=length, freq='D')
        
        # 生成趋势数据
        trend_component = np.arange(length) * trend
        noise = np.random.normal(0, 0.01, length)
        prices = 100 * np.exp(trend_component + noise)
        
        return pd.DataFrame({
            'ts_code': '000001.SZ',
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'pre_close': np.concatenate([[100], prices[:-1]]),
            'volume': np.random.randint(1000000, 10000000, length),
            'amount': prices * np.random.randint(1000000, 10000000, length)
        })

class TestMomentumFactorCalculator:
    """动量类因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return MomentumFactorCalculator(mock_engine)
    
    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        return TestDataGenerator.generate_stock_data(50)
    
    def test_calculate_rsi_basic(self, calculator, sample_data):
        """测试RSI基本计算"""
        rsi = calculator.calculate_rsi(sample_data, window=14)
        
        # 基本验证
        assert not rsi.empty, "RSI结果不应为空"
        assert len(rsi) == len(sample_data), "RSI长度应与输入数据一致"
        
        # 数值范围验证
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi), "RSI值应在0-100之间"
        
        # 前期数据应为NaN
        assert pd.isna(rsi.iloc[:13]).all(), "前13个RSI值应为NaN"
        assert not pd.isna(rsi.iloc[14]), "第15个RSI值不应为NaN"
    
    def test_calculate_rsi_edge_cases(self, calculator):
        """测试RSI边界条件"""
        # 测试数据不足的情况
        short_data = TestDataGenerator.generate_stock_data(10)
        rsi = calculator.calculate_rsi(short_data, window=14)
        assert rsi.isna().all(), "数据不足时RSI应全为NaN"
        
        # 测试价格不变的情况
        constant_data = TestDataGenerator.generate_stock_data(30)
        constant_data['close'] = 100.0  # 价格不变
        rsi = calculator.calculate_rsi(constant_data, window=14)
        valid_rsi = rsi.dropna()
        assert all(abs(val - 50) < 1e-10 for val in valid_rsi), "价格不变时RSI应接近50"
    
    def test_calculate_rsi_mathematical_correctness(self, calculator):
        """测试RSI数学正确性"""
        # 使用已知数据验证RSI计算
        test_data = pd.DataFrame({
            'ts_code': '000001.SZ',
            'trade_date': pd.date_range('2024-01-01', periods=20, freq='D'),
            'close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 105, 
                     107, 108, 107, 109, 110, 109, 111, 112, 111, 113]
        })
        
        rsi = calculator.calculate_rsi(test_data, window=14)
        
        # 验证最后一个RSI值的合理性
        last_rsi = rsi.iloc[-1]
        assert 40 <= last_rsi <= 80, f"RSI值 {last_rsi} 应在合理范围内"
    
    def test_calculate_williams_r(self, calculator, sample_data):
        """测试威廉指标计算"""
        wr = calculator.calculate_williams_r(sample_data, window=14)
        
        # 基本验证
        assert not wr.empty, "威廉指标结果不应为空"
        assert len(wr) == len(sample_data), "威廉指标长度应与输入数据一致"
        
        # 数值范围验证
        valid_wr = wr.dropna()
        assert all(-100 <= val <= 0 for val in valid_wr), "威廉指标值应在-100到0之间"
    
    def test_calculate_momentum(self, calculator, sample_data):
        """测试动量指标计算"""
        momentum = calculator.calculate_momentum(sample_data, window=10)
        
        # 基本验证
        assert not momentum.empty, "动量指标结果不应为空"
        assert len(momentum) == len(sample_data), "动量指标长度应与输入数据一致"
        
        # 前期数据应为NaN
        assert pd.isna(momentum.iloc[:9]).all(), "前9个动量值应为NaN"
        assert not pd.isna(momentum.iloc[10]), "第11个动量值不应为NaN"
    
    def test_calculate_roc(self, calculator, sample_data):
        """测试变化率指标计算"""
        roc = calculator.calculate_roc(sample_data, window=10)
        
        # 基本验证
        assert not roc.empty, "ROC结果不应为空"
        assert len(roc) == len(sample_data), "ROC长度应与输入数据一致"
        
        # 验证ROC计算公式
        for i in range(10, len(sample_data)):
            expected_roc = (sample_data['close'].iloc[i] / sample_data['close'].iloc[i-10] - 1) * 100
            actual_roc = roc.iloc[i]
            assert abs(actual_roc - expected_roc) < 1e-10, f"ROC计算错误: 期望{expected_roc}, 实际{actual_roc}"

class TestTrendFactorCalculator:
    """趋势类因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return TrendFactorCalculator(mock_engine)
    
    @pytest.fixture
    def trending_data(self):
        """生成趋势数据"""
        return TestDataGenerator.generate_trending_data(50, trend=0.001)
    
    def test_calculate_sma(self, calculator, trending_data):
        """测试简单移动平均计算"""
        sma = calculator.calculate_sma(trending_data, window=10)
        
        # 基本验证
        assert not sma.empty, "SMA结果不应为空"
        assert len(sma) == len(trending_data), "SMA长度应与输入数据一致"
        
        # 验证SMA计算正确性
        for i in range(9, len(trending_data)):
            expected_sma = trending_data['close'].iloc[i-9:i+1].mean()
            actual_sma = sma.iloc[i]
            assert abs(actual_sma - expected_sma) < 1e-10, f"SMA计算错误: 期望{expected_sma}, 实际{actual_sma}"
    
    def test_calculate_ema(self, calculator, trending_data):
        """测试指数移动平均计算"""
        ema = calculator.calculate_ema(trending_data, window=12)
        
        # 基本验证
        assert not ema.empty, "EMA结果不应为空"
        assert len(ema) == len(trending_data), "EMA长度应与输入数据一致"
        
        # 验证EMA的平滑特性
        valid_ema = ema.dropna()
        ema_diff = valid_ema.diff().abs()
        close_diff = trending_data['close'].iloc[len(trending_data)-len(valid_ema):].diff().abs()
        
        # EMA应该比原始价格更平滑
        assert ema_diff.mean() < close_diff.mean(), "EMA应该比原始价格更平滑"
    
    def test_calculate_macd(self, calculator, trending_data):
        """测试MACD指标计算"""
        macd_line, signal_line, histogram = calculator.calculate_macd(trending_data)
        
        # 基本验证
        assert not macd_line.empty, "MACD线不应为空"
        assert not signal_line.empty, "信号线不应为空"
        assert not histogram.empty, "柱状图不应为空"
        
        # 验证MACD关系
        valid_indices = ~(macd_line.isna() | signal_line.isna() | histogram.isna())
        macd_valid = macd_line[valid_indices]
        signal_valid = signal_line[valid_indices]
        hist_valid = histogram[valid_indices]
        
        # 柱状图 = MACD线 - 信号线
        calculated_hist = macd_valid - signal_valid
        assert np.allclose(hist_valid, calculated_hist, atol=1e-10), "MACD柱状图计算错误"
    
    def test_calculate_bollinger_bands(self, calculator, trending_data):
        """测试布林带计算"""
        upper, lower, width, position = calculator.calculate_bollinger_bands(trending_data, window=20, std_dev=2)
        
        # 基本验证
        assert not upper.empty, "布林带上轨不应为空"
        assert not lower.empty, "布林带下轨不应为空"
        assert not width.empty, "布林带宽度不应为空"
        assert not position.empty, "布林带位置不应为空"
        
        # 验证布林带关系
        valid_indices = ~(upper.isna() | lower.isna())
        assert all(upper[valid_indices] >= lower[valid_indices]), "上轨应大于等于下轨"
        
        # 验证位置计算
        close_prices = trending_data['close']
        valid_pos_indices = ~position.isna()
        pos_values = position[valid_pos_indices]
        assert all(0 <= val <= 1 for val in pos_values), "布林带位置应在0-1之间"

class TestVolatilityFactorCalculator:
    """波动率类因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return VolatilityFactorCalculator(mock_engine)
    
    @pytest.fixture
    def volatile_data(self):
        """生成高波动数据"""
        return TestDataGenerator.generate_stock_data(50, volatility=0.05)
    
    def test_calculate_historical_volatility(self, calculator, volatile_data):
        """测试历史波动率计算"""
        volatility = calculator.calculate_historical_volatility(volatile_data, window=20)
        
        # 基本验证
        assert not volatility.empty, "历史波动率结果不应为空"
        assert len(volatility) == len(volatile_data), "历史波动率长度应与输入数据一致"
        
        # 波动率应为正值
        valid_vol = volatility.dropna()
        assert all(val >= 0 for val in valid_vol), "历史波动率应为非负值"
        
        # 验证波动率计算的合理性
        assert valid_vol.mean() > 0, "平均波动率应大于0"
    
    def test_calculate_atr(self, calculator, volatile_data):
        """测试ATR指标计算"""
        atr = calculator.calculate_atr(volatile_data, window=14)
        
        # 基本验证
        assert not atr.empty, "ATR结果不应为空"
        assert len(atr) == len(volatile_data), "ATR长度应与输入数据一致"
        
        # ATR应为正值
        valid_atr = atr.dropna()
        assert all(val >= 0 for val in valid_atr), "ATR应为非负值"
        
        # 验证ATR的平滑特性
        true_ranges = []
        for i in range(1, len(volatile_data)):
            tr1 = volatile_data['high'].iloc[i] - volatile_data['low'].iloc[i]
            tr2 = abs(volatile_data['high'].iloc[i] - volatile_data['close'].iloc[i-1])
            tr3 = abs(volatile_data['low'].iloc[i] - volatile_data['close'].iloc[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        # ATR应该比单日真实波幅更平滑
        tr_volatility = np.std(true_ranges)
        atr_volatility = np.std(valid_atr)
        assert atr_volatility <= tr_volatility, "ATR应该比真实波幅更平滑"

class TestVolumeFactorCalculator:
    """成交量类因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return VolumeFactorCalculator(mock_engine)
    
    @pytest.fixture
    def volume_data(self):
        """生成成交量数据"""
        data = TestDataGenerator.generate_stock_data(50)
        # 添加一些成交量异常
        data.loc[10, 'volume'] = data['volume'].iloc[10] * 5  # 异常放量
        data.loc[20, 'volume'] = data['volume'].iloc[20] * 0.2  # 异常缩量
        return data
    
    def test_calculate_volume_sma(self, calculator, volume_data):
        """测试成交量移动平均计算"""
        vol_sma = calculator.calculate_volume_sma(volume_data, window=5)
        
        # 基本验证
        assert not vol_sma.empty, "成交量SMA结果不应为空"
        assert len(vol_sma) == len(volume_data), "成交量SMA长度应与输入数据一致"
        
        # 验证计算正确性
        for i in range(4, len(volume_data)):
            expected_sma = volume_data['volume'].iloc[i-4:i+1].mean()
            actual_sma = vol_sma.iloc[i]
            assert abs(actual_sma - expected_sma) < 1e-6, f"成交量SMA计算错误"
    
    def test_calculate_volume_ratio(self, calculator, volume_data):
        """测试量比指标计算"""
        vol_ratio = calculator.calculate_volume_ratio(volume_data, window=5)
        
        # 基本验证
        assert not vol_ratio.empty, "量比结果不应为空"
        assert len(vol_ratio) == len(volume_data), "量比长度应与输入数据一致"
        
        # 量比应为正值
        valid_ratio = vol_ratio.dropna()
        assert all(val > 0 for val in valid_ratio), "量比应为正值"
        
        # 验证异常放量的检测
        assert vol_ratio.iloc[10] > 3, "应该检测到异常放量"
    
    def test_calculate_vpt(self, calculator, volume_data):
        """测试量价趋势指标计算"""
        vpt = calculator.calculate_vpt(volume_data)
        
        # 基本验证
        assert not vpt.empty, "VPT结果不应为空"
        assert len(vpt) == len(volume_data), "VPT长度应与输入数据一致"
        
        # VPT应该是累积值
        valid_vpt = vpt.dropna()
        assert len(valid_vpt) > 0, "应该有有效的VPT值"
    
    def test_calculate_mfi(self, calculator, volume_data):
        """测试资金流量指标计算"""
        mfi = calculator.calculate_mfi(volume_data, window=14)
        
        # 基本验证
        assert not mfi.empty, "MFI结果不应为空"
        assert len(mfi) == len(volume_data), "MFI长度应与输入数据一致"
        
        # MFI应在0-100之间
        valid_mfi = mfi.dropna()
        assert all(0 <= val <= 100 for val in valid_mfi), "MFI值应在0-100之间"

class TestTechnicalFactorEngine:
    """技术面因子引擎测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def technical_engine(self, mock_engine):
        """创建技术面因子引擎"""
        return TechnicalFactorEngine(mock_engine)
    
    @pytest.fixture
    def complete_data(self):
        """生成完整的测试数据"""
        return TestDataGenerator.generate_stock_data(100)
    
    def test_engine_initialization(self, technical_engine):
        """测试引擎初始化"""
        assert technical_engine is not None
        assert hasattr(technical_engine, 'momentum_calculator')
        assert hasattr(technical_engine, 'trend_calculator')
        assert hasattr(technical_engine, 'volatility_calculator')
        assert hasattr(technical_engine, 'volume_calculator')
    
    def test_calculate_all_factors(self, technical_engine, complete_data):
        """测试计算所有技术面因子"""
        # 模拟数据库查询
        with patch.object(technical_engine, '_get_stock_data', return_value=complete_data):
            factors = technical_engine.calculate_all_factors(['000001.SZ'])
        
        # 验证返回的因子
        assert isinstance(factors, pd.DataFrame)
        assert not factors.empty
        assert 'ts_code' in factors.columns
        
        # 验证包含主要技术指标
        expected_factors = ['sma_5', 'sma_20', 'rsi_14', 'macd', 'atr_14', 'volume_sma_5']
        for factor in expected_factors:
            if factor in factors.columns:
                assert not factors[factor].isna().all(), f"因子 {factor} 不应全为NaN"
    
    def test_calculate_factor_by_name(self, technical_engine, complete_data):
        """测试按名称计算特定因子"""
        with patch.object(technical_engine, '_get_stock_data', return_value=complete_data):
            # 测试计算RSI
            rsi_result = technical_engine.calculate_factor('rsi_14', ['000001.SZ'])
            assert not rsi_result.empty
            assert 'rsi_14' in rsi_result.columns
            
            # 测试计算SMA
            sma_result = technical_engine.calculate_factor('sma_20', ['000001.SZ'])
            assert not sma_result.empty
            assert 'sma_20' in sma_result.columns
    
    def test_batch_calculation(self, technical_engine, complete_data):
        """测试批量计算"""
        stock_codes = ['000001.SZ', '000002.SZ', '600000.SH']
        
        # 为每个股票返回相同的测试数据
        def mock_get_data(codes, start_date=None, end_date=None):
            result_data = []
            for code in codes:
                data = complete_data.copy()
                data['ts_code'] = code
                result_data.append(data)
            return pd.concat(result_data, ignore_index=True)
        
        with patch.object(technical_engine, '_get_stock_data', side_effect=mock_get_data):
            factors = technical_engine.calculate_all_factors(stock_codes)
        
        # 验证批量计算结果
        assert len(factors['ts_code'].unique()) == len(stock_codes)
        for code in stock_codes:
            code_data = factors[factors['ts_code'] == code]
            assert not code_data.empty, f"股票 {code} 应该有计算结果"
    
    def test_error_handling(self, technical_engine):
        """测试错误处理"""
        # 测试空数据
        with patch.object(technical_engine, '_get_stock_data', return_value=pd.DataFrame()):
            result = technical_engine.calculate_all_factors(['000001.SZ'])
            assert result.empty or len(result) == 0
        
        # 测试无效股票代码
        with patch.object(technical_engine, '_get_stock_data', return_value=pd.DataFrame()):
            result = technical_engine.calculate_factor('rsi_14', ['INVALID.CODE'])
            assert result.empty or len(result) == 0

class TestMathematicalAccuracy:
    """数学准确性测试"""
    
    def test_rsi_known_values(self):
        """使用已知数值测试RSI计算准确性"""
        # 使用标准的RSI测试数据
        test_prices = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03,
            46.83, 47.69, 46.49, 46.26, 47.09, 46.66, 46.80, 46.23, 46.38, 46.33
        ]
        
        data = pd.DataFrame({
            'ts_code': '000001.SZ',
            'trade_date': pd.date_range('2024-01-01', periods=len(test_prices), freq='D'),
            'close': test_prices
        })
        
        calculator = MomentumFactorCalculator(Mock())
        rsi = calculator.calculate_rsi(data, window=14)
        
        # 验证最后的RSI值（应该接近已知的标准值）
        last_rsi = rsi.iloc[-1]
        assert 65 <= last_rsi <= 75, f"RSI值 {last_rsi} 应在预期范围内"
    
    def test_sma_mathematical_correctness(self):
        """测试SMA数学正确性"""
        test_prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = pd.DataFrame({
            'ts_code': '000001.SZ',
            'trade_date': pd.date_range('2024-01-01', periods=len(test_prices), freq='D'),
            'close': test_prices
        })
        
        calculator = TrendFactorCalculator(Mock())
        sma = calculator.calculate_sma(data, window=5)
        
        # 验证已知的SMA值
        expected_sma_5 = 3.0  # (1+2+3+4+5)/5
        expected_sma_6 = 4.0  # (2+3+4+5+6)/5
        
        assert abs(sma.iloc[4] - expected_sma_5) < 1e-10, "SMA计算错误"
        assert abs(sma.iloc[5] - expected_sma_6) < 1e-10, "SMA计算错误"
    
    def test_volatility_calculation_accuracy(self):
        """测试波动率计算准确性"""
        # 使用已知波动率的数据
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)  # 2%日波动率
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'ts_code': '000001.SZ',
            'trade_date': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'close': prices
        })
        
        calculator = VolatilityFactorCalculator(Mock())
        volatility = calculator.calculate_historical_volatility(data, window=20)
        
        # 验证波动率在合理范围内
        valid_vol = volatility.dropna()
        mean_vol = valid_vol.mean()
        assert 0.01 <= mean_vol <= 0.05, f"计算的波动率 {mean_vol} 应在合理范围内"

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])