from datetime import date, datetime

from src.compute.factor_models import CalculationStatus, FactorType
from src.compute.technical_factor_engine import (Mock, """, __file__, from,
                                                 import)
from src.compute.technical_factor_engine import \
    numpy as np  # !/usr/bin/env python3; -*- coding: utf-8 -*-
from src.compute.technical_factor_engine import (os, os.path.abspath,
                                                 os.path.dirname)
from src.compute.technical_factor_engine import pandas as pd
from src.compute.technical_factor_engine import (patch, pytest, sys,
                                                 sys.path.append,
                                                 unittest.mock, 技术面因子引擎测试)

    TechnicalFactorEngine, MomentumFactorCalculator,
    TrendFactorCalculator, VolatilityFactorCalculator, VolumeFactorCalculator
)


class TestMomentumFactorCalculator:
    """动量类因子计算器测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        # 生成价格数据
        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        high_prices = close_prices + np.random.rand(50) * 2
        low_prices = close_prices - np.random.rand(50) * 2

        return pd.DataFrame({
            'trade_date': dates,
            'open': close_prices + np.random.randn(50) * 0.2,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'vol': np.random.randint(1000000, 10000000, 50)
        })

    def test_calculate_rsi(self, sample_data):
        """测试RSI计算"""
        calculator = MomentumFactorCalculator()

        rsi = calculator.calculate_rsi(sample_data, window=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)

        # RSI值应该在0-100之间
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi)

    def test_calculate_williams_r(self, sample_data):
        """测试威廉指标计算"""
        calculator = MomentumFactorCalculator()

        williams_r = calculator.calculate_williams_r(sample_data, window=14)

        assert isinstance(williams_r, pd.Series)
        assert len(williams_r) == len(sample_data)

        # 威廉指标值应该在-100到0之间
        valid_values = williams_r.dropna()
        assert all(-100 <= val <= 0 for val in valid_values)

    def test_calculate_momentum(self, sample_data):
        """测试动量指标计算"""
        calculator = MomentumFactorCalculator()

        momentum = calculator.calculate_momentum(sample_data, window=10)

        assert isinstance(momentum, pd.Series)
        assert len(momentum) == len(sample_data)

        # 前10个值应该是NaN
        assert pd.isna(momentum.iloc[:10]).all()

    def test_calculate_all_factors(self, sample_data):
        """测试计算所有动量类因子"""
        calculator = MomentumFactorCalculator()

        results = calculator.calculate(sample_data)

        expected_factors = [
            'rsi_14', 'rsi_6', 'williams_r_14',
            'momentum_5', 'momentum_10', 'momentum_20',
            'roc_5', 'roc_10', 'roc_20'
        ]

        for factor in expected_factors:
            assert factor in results
            assert isinstance(results[factor], pd.Series)


class TestTrendFactorCalculator:
    """趋势类因子计算器测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        return pd.DataFrame({
            'trade_date': dates,
            'close': close_prices,
            'vol': np.random.randint(1000000, 10000000, 100)
        })

    def test_calculate_sma(self, sample_data):
        """测试简单移动平均计算"""
        calculator = TrendFactorCalculator()

        sma_20 = calculator.calculate_sma(sample_data, window=20)

        assert isinstance(sma_20, pd.Series)
        assert len(sma_20) == len(sample_data)

        # 前19个值应该是NaN
        assert pd.isna(sma_20.iloc[:19]).all()

        # 第20个值应该是前20个收盘价的平均值
        expected_value = sample_data['close'].iloc[:20].mean()
        assert abs(sma_20.iloc[19] - expected_value) < 1e-10

    def test_calculate_ema(self, sample_data):
        """测试指数移动平均计算"""
        calculator = TrendFactorCalculator()

        ema_12 = calculator.calculate_ema(sample_data, window=12)

        assert isinstance(ema_12, pd.Series)
        assert len(ema_12) == len(sample_data)

        # EMA不应该有NaN值（除了第一个可能的NaN）
        assert not pd.isna(ema_12.iloc[1:]).any()

    def test_calculate_macd(self, sample_data):
        """测试MACD计算"""
        calculator = TrendFactorCalculator()

        macd_results = calculator.calculate_macd(sample_data)

        expected_keys = ['macd', 'macd_signal', 'macd_histogram']
        for key in expected_keys:
            assert key in macd_results
            assert isinstance(macd_results[key], pd.Series)

    def test_calculate_all_factors(self, sample_data):
        """测试计算所有趋势类因子"""
        calculator = TrendFactorCalculator()

        results = calculator.calculate(sample_data)

        expected_factors = [
            'sma_5', 'sma_10', 'sma_20', 'sma_60',
            'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram',
            'price_to_sma20', 'price_to_sma60'
        ]

        for factor in expected_factors:
            assert factor in results
            assert isinstance(results[factor], pd.Series)


class TestVolatilityFactorCalculator:
    """波动率类因子计算器测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        high_prices = close_prices + np.random.rand(50) * 2
        low_prices = close_prices - np.random.rand(50) * 2

        return pd.DataFrame({
            'trade_date': dates,
            'open': close_prices + np.random.randn(50) * 0.2,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        })

    def test_calculate_historical_volatility(self, sample_data):
        """测试历史波动率计算"""
        calculator = VolatilityFactorCalculator()

        volatility = calculator.calculate_historical_volatility(sample_data, window=20)

        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(sample_data)

        # 波动率应该是正数
        valid_values = volatility.dropna()
        assert all(val >= 0 for val in valid_values)

    def test_calculate_atr(self, sample_data):
        """测试ATR计算"""
        calculator = VolatilityFactorCalculator()

        atr = calculator.calculate_atr(sample_data, window=14)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_data)

        # ATR应该是正数
        valid_values = atr.dropna()
        assert all(val >= 0 for val in valid_values)

    def test_calculate_bollinger_bands(self, sample_data):
        """测试布林带计算"""
        calculator = VolatilityFactorCalculator()

        bb_results = calculator.calculate_bollinger_bands(sample_data)

        expected_keys = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position']
        for key in expected_keys:
            assert key in bb_results
            assert isinstance(bb_results[key], pd.Series)

        # 上轨应该大于中轨，中轨应该大于下轨
        valid_indices = bb_results['bb_upper'].notna()
        upper = bb_results['bb_upper'][valid_indices]
        middle = bb_results['bb_middle'][valid_indices]
        lower = bb_results['bb_lower'][valid_indices]

        assert all(upper >= middle)
        assert all(middle >= lower)


class TestVolumeFactorCalculator:
    """成交量类因子计算器测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        high_prices = close_prices + np.random.rand(50) * 2
        low_prices = close_prices - np.random.rand(50) * 2
        volumes = np.random.randint(1000000, 10000000, 50)

        return pd.DataFrame({
            'trade_date': dates,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'vol': volumes
        })

    def test_calculate_volume_sma(self, sample_data):
        """测试成交量移动平均计算"""
        calculator = VolumeFactorCalculator()

        volume_sma = calculator.calculate_volume_sma(sample_data, window=5)

        assert isinstance(volume_sma, pd.Series)
        assert len(volume_sma) == len(sample_data)

        # 成交量均值应该是正数
        valid_values = volume_sma.dropna()
        assert all(val > 0 for val in valid_values)

    def test_calculate_volume_ratio(self, sample_data):
        """测试量比计算"""
        calculator = VolumeFactorCalculator()

        volume_ratio = calculator.calculate_volume_ratio(sample_data, window=5)

        assert isinstance(volume_ratio, pd.Series)
        assert len(volume_ratio) == len(sample_data)

        # 量比应该是正数
        valid_values = volume_ratio.dropna()
        assert all(val > 0 for val in valid_values)

    def test_calculate_all_factors(self, sample_data):
        """测试计算所有成交量类因子"""
        calculator = VolumeFactorCalculator()

        results = calculator.calculate(sample_data)

        expected_factors = [
            'volume_sma_5', 'volume_sma_20',
            'volume_ratio_5', 'volume_ratio_20',
            'vpt', 'mfi'
        ]

        for factor in expected_factors:
            assert factor in results
            assert isinstance(results[factor], pd.Series)


class TestTechnicalFactorEngine:
    """技术面因子引擎测试"""

    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high_prices = close_prices + np.random.rand(100) * 2
        low_prices = close_prices - np.random.rand(100) * 2
        volumes = np.random.randint(1000000, 10000000, 100)

        return pd.DataFrame({
            'trade_date': pd.to_datetime(dates),
            'ts_code': ['000001.SZ'] * 100,
            'open': close_prices + np.random.randn(100) * 0.2,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'vol': volumes,
            'amount': volumes * close_prices
        })

    def test_initialization(self, mock_engine):
        """测试引擎初始化"""
        engine = TechnicalFactorEngine(mock_engine)

        assert engine.factor_type == FactorType.TECHNICAL
        assert len(engine.list_factors()) > 0

        # 检查各类计算器是否初始化
        assert hasattr(engine, 'momentum_calculator')
        assert hasattr(engine, 'trend_calculator')
        assert hasattr(engine, 'volatility_calculator')
        assert hasattr(engine, 'volume_calculator')

    def test_factor_registration(self, mock_engine):
        """测试因子注册"""
        engine = TechnicalFactorEngine(mock_engine)

        # 检查是否注册了预期的因子
        factors = engine.list_factors()

        # 动量类因子
        momentum_factors = ['rsi_14', 'rsi_6', 'williams_r_14', 'momentum_5', 'momentum_10', 'momentum_20']
        for factor in momentum_factors:
            assert factor in factors

        # 趋势类因子
        trend_factors = ['sma_5', 'sma_10', 'sma_20', 'macd', 'macd_signal']
        for factor in trend_factors:
            assert factor in factors

        # 波动率类因子
        volatility_factors = ['volatility_5', 'volatility_20', 'atr_14', 'bb_upper', 'bb_lower']
        for factor in volatility_factors:
            assert factor in factors

        # 成交量类因子
        volume_factors = ['volume_sma_5', 'volume_ratio_5', 'vpt', 'mfi']
        for factor in volume_factors:
            assert factor in factors

    @patch('src.compute.technical_factor_engine.TechnicalFactorEngine.get_required_data')
    def test_calculate_factors_success(self, mock_get_data, mock_engine, sample_data):
        """测试因子计算成功"""
        mock_get_data.return_value = sample_data

        engine = TechnicalFactorEngine(mock_engine)

        result = engine.calculate_factors('000001.SZ')

        assert result.ts_code == '000001.SZ'
        assert result.factor_type == FactorType.TECHNICAL
        assert result.status == CalculationStatus.SUCCESS
        assert result.data_points == len(sample_data)
        assert result.execution_time is not None
        assert len(result.factors) > 0

    @patch('src.compute.technical_factor_engine.TechnicalFactorEngine.get_required_data')
    def test_calculate_factors_no_data(self, mock_get_data, mock_engine):
        """测试无数据情况"""
        mock_get_data.return_value = pd.DataFrame()

        engine = TechnicalFactorEngine(mock_engine)

        result = engine.calculate_factors('000001.SZ')

        assert result.status == CalculationStatus.SKIPPED
        assert result.error_message == "无可用数据"

    @patch('src.compute.technical_factor_engine.TechnicalFactorEngine.get_required_data')
    def test_calculate_factors_error(self, mock_get_data, mock_engine):
        """测试计算错误情况"""
        mock_get_data.side_effect = Exception("数据库连接错误")

        engine = TechnicalFactorEngine(mock_engine)

        result = engine.calculate_factors('000001.SZ')

        assert result.status == CalculationStatus.FAILED
        assert "数据库连接错误" in result.error_message


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])