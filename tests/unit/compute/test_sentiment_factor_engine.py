from datetime import date, datetime

from src.compute.factor_models import CalculationStatus, FactorType
from src.compute.sentiment_factor_engine import (
    Any, AttentionFactorCalculator, Dict, EventFactorCalculator, Mock,
    MoneyFlowFactorCalculator, SentimentFactorEngine,
    SentimentStrengthFactorCalculator, """, from, import)
from src.compute.sentiment_factor_engine import \
    numpy as np  # !/usr/bin/env python3; -*- coding: utf-8 -*-
from src.compute.sentiment_factor_engine import pandas as pd
from src.compute.sentiment_factor_engine import (patch, pytest, typing,
                                                 unittest.mock, 情绪面因子引擎测试)


# 测试常量定义
class TestConstants:
    """测试用常量定义"""
    # 数值范围常量
    MONEY_FLOW_RANGE = (-1000, 1000)
    NET_INFLOW_RATE_RANGE = (-100, 100)
    BIG_ORDER_RATIO_RANGE = (-50, 50)
    VOLUME_PRICE_TREND_RANGE = (-100, 100)
    TURNOVER_RATE_RANGE = (0, 100)
    VOLUME_ATTENTION_RANGE = (0, 20)
    PRICE_ATTENTION_RANGE = (0, 50)
    ATTENTION_SCORE_RANGE = (0, 100)
    PRICE_MOMENTUM_RANGE = (-100, 100)
    VOLATILITY_SENTIMENT_RANGE = (0, 200)
    BULLISH_BEARISH_RATIO_RANGE = (0, 100)
    ABNORMAL_ZSCORE_RANGE = (-10, 10)
    SIGNAL_VALUES = {-1, 0, 1}
    BINARY_SIGNAL_VALUES = {0, 1}

    # 测试数据参数
    DEFAULT_DATA_POINTS = 30
    SMALL_DATA_POINTS = 20
    LARGE_DATA_POINTS = 100


class MarketDataFactory:
    """市场数据工厂类"""

    @staticmethod
    def create_basic_market_data(
        periods: int = TestConstants.DEFAULT_DATA_POINTS,
        ts_code: str = "000001.SZ",
        start_date: str = "2024-01-01",
        seed: int = 42
    ) -> pd.DataFrame:
        """创建基础市场数据"""
        np.random.seed(seed)

        return pd.DataFrame({
            'ts_code': [ts_code] * periods,
            'trade_date': pd.date_range(start_date, periods=periods, freq='D'),
            'open': np.random.uniform(10, 15, periods),
            'high': np.random.uniform(15, 20, periods),
            'low': np.random.uniform(8, 12, periods),
            'close': np.random.uniform(10, 15, periods),
            'pre_close': np.random.uniform(10, 15, periods),
            'vol': np.random.uniform(1000000, 5000000, periods),
            'amount': np.random.uniform(100000000, 500000000, periods),
            'turnover_rate': np.random.uniform(1, 10, periods),
            'circ_mv': np.random.uniform(1000000, 5000000, periods),
            'total_mv': np.random.uniform(1200000, 6000000, periods)
        })

    @staticmethod
    def create_minimal_market_data(
        periods: int = TestConstants.SMALL_DATA_POINTS,
        ts_code: str = "000001.SZ"
    ) -> pd.DataFrame:
        """创建最小化市场数据（用于测试缺失列的情况）"""
        np.random.seed(42)

        return pd.DataFrame({
            'ts_code': [ts_code] * periods,
            'trade_date': pd.date_range('2024-01-01', periods=periods, freq='D'),
            'close': np.random.uniform(10, 15, periods),
            'vol': np.random.uniform(1000000, 5000000, periods)
        })

    @staticmethod
    def create_limit_price_data(periods: int = 25) -> pd.DataFrame:
        """创建包含涨跌停的测试数据"""
        data = MarketDataFactory.create_basic_market_data(periods)

        # 添加涨停和跌停情况
        data.loc[0, 'close'] = data.loc[0, 'pre_close'] * 1.1  # 涨停
        data.loc[1, 'close'] = data.loc[1, 'pre_close'] * 0.9  # 跌停

        return data


class TestAssertions:
    """测试断言辅助类"""

    @staticmethod
    def assert_series_properties(series: pd.Series, expected_length: int, name: str = "Series"):
        """断言Series的基本属性"""
        assert isinstance(series, pd.Series), f"{name} 应该是 pd.Series 类型"
        assert len(series) == expected_length, f"{name} 长度应该是 {expected_length}"

    @staticmethod
    def assert_value_range(series: pd.Series, value_range: tuple, name: str = "Series"):
        """断言数值范围"""
        valid_values = series.dropna()
        if not valid_values.empty:
            min_val, max_val = value_range
            assert all(min_val <= val <= max_val for val in valid_values), \
                f"{name} 的值应该在 {value_range} 范围内"

    @staticmethod
    def assert_signal_values(series: pd.Series, expected_values: set, name: str = "Signal"):
        """断言信号值"""
        unique_values = set(series.unique())
        assert unique_values.issubset(expected_values), \
            f"{name} 的值应该在 {expected_values} 中"

    @staticmethod
    def assert_factor_results(results: Dict[str, pd.Series], expected_factors: list, data_length: int):
        """断言因子计算结果"""
        for factor in expected_factors:
            assert factor in results, f"应该包含因子 {factor}"
            assert isinstance(results[factor], pd.Series), f"因子 {factor} 应该是 pd.Series 类型"
            assert len(results[factor]) == data_length, f"因子 {factor} 长度应该是 {data_length}"


class TestMoneyFlowFactorCalculator:
    """资金流向因子计算器测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建测试市场数据"""
        return MarketDataFactory.create_basic_market_data(periods=20)

    @pytest.mark.parametrize("window", [5, 10, 20])
    def test_calculate_money_flow(self, sample_market_data, window):
        """测试资金流向计算"""
        calculator = MoneyFlowFactorCalculator()

        money_flow = calculator.calculate_money_flow(sample_market_data, window=window)

        TestAssertions.assert_series_properties(money_flow, len(sample_market_data), "资金流向")
        TestAssertions.assert_value_range(money_flow, TestConstants.MONEY_FLOW_RANGE, "资金流向")

    def test_calculate_net_inflow_rate(self, sample_market_data):
        """测试净流入率计算"""
        calculator = MoneyFlowFactorCalculator()

        net_inflow_rate = calculator.calculate_net_inflow_rate(sample_market_data)

        TestAssertions.assert_series_properties(net_inflow_rate, len(sample_market_data), "净流入率")
        TestAssertions.assert_value_range(net_inflow_rate, TestConstants.NET_INFLOW_RATE_RANGE, "净流入率")

    def test_calculate_big_order_ratio(self, sample_market_data):
        """测试大单占比计算"""
        calculator = MoneyFlowFactorCalculator()

        big_order_ratio = calculator.calculate_big_order_ratio(sample_market_data)

        TestAssertions.assert_series_properties(big_order_ratio, len(sample_market_data), "大单占比")
        TestAssertions.assert_value_range(big_order_ratio, TestConstants.BIG_ORDER_RATIO_RANGE, "大单占比")

    def test_calculate_volume_price_trend(self, sample_market_data):
        """测试量价趋势计算"""
        calculator = MoneyFlowFactorCalculator()

        vpt = calculator.calculate_volume_price_trend(sample_market_data)

        TestAssertions.assert_series_properties(vpt, len(sample_market_data), "量价趋势")
        TestAssertions.assert_value_range(vpt, TestConstants.VOLUME_PRICE_TREND_RANGE, "量价趋势")

    def test_calculate_all_factors(self, sample_market_data):
        """测试计算所有资金流向因子"""
        calculator = MoneyFlowFactorCalculator()

        results = calculator.calculate(sample_market_data)

        expected_factors = [
            'money_flow_5', 'money_flow_20', 'net_inflow_rate',
            'big_order_ratio', 'medium_order_ratio', 'small_order_ratio',
            'volume_price_trend'
        ]

        TestAssertions.assert_factor_results(results, expected_factors, len(sample_market_data))

    def test_missing_columns(self):
        """测试缺少必要列的情况"""
        calculator = MoneyFlowFactorCalculator()

        # 缺少amount列
        incomplete_data = MarketDataFactory.create_minimal_market_data(periods=3)

        money_flow = calculator.calculate_money_flow(incomplete_data)
        assert money_flow.empty or money_flow.isna().all()

    def test_empty_data(self):
        """测试空数据情况"""
        calculator = MoneyFlowFactorCalculator()
        empty_data = pd.DataFrame()

        money_flow = calculator.calculate_money_flow(empty_data)
        assert money_flow.empty

    @pytest.mark.parametrize("data_points", [1, 5, 10])
    def test_insufficient_data(self, data_points):
        """测试数据不足的情况"""
        calculator = MoneyFlowFactorCalculator()
        small_data = MarketDataFactory.create_basic_market_data(periods=data_points)

        # 对于小数据集，某些计算可能返回NaN或空值
        results = calculator.calculate(small_data)

        # 验证结果结构正确，即使值可能为NaN
        assert isinstance(results, dict)
        assert len(results) > 0


class TestAttentionFactorCalculator:
    """关注度因子计算器测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建测试市场数据"""
        return MarketDataFactory.create_basic_market_data(periods=20)

    def test_calculate_turnover_attention(self, sample_market_data):
        """测试换手率关注度计算"""
        calculator = AttentionFactorCalculator()

        turnover_attention = calculator.calculate_turnover_attention(sample_market_data)

        TestAssertions.assert_series_properties(turnover_attention, len(sample_market_data), "换手率关注度")
        TestAssertions.assert_value_range(turnover_attention, TestConstants.TURNOVER_RATE_RANGE, "换手率关注度")

    def test_calculate_volume_attention(self, sample_market_data):
        """测试成交量关注度计算"""
        calculator = AttentionFactorCalculator()

        volume_attention = calculator.calculate_volume_attention(sample_market_data)

        assert isinstance(volume_attention, pd.Series)
        assert len(volume_attention) == len(sample_market_data)

        # 成交量关注度应该大于0
        valid_values = volume_attention.dropna()
        if not valid_values.empty:
            assert all(val > 0 for val in valid_values)
            assert all(val < 20 for val in valid_values)

    def test_calculate_price_attention(self, sample_market_data):
        """测试价格波动关注度计算"""
        calculator = AttentionFactorCalculator()

        price_attention = calculator.calculate_price_attention(sample_market_data)

        assert isinstance(price_attention, pd.Series)
        assert len(price_attention) == len(sample_market_data)

        # 价格波动关注度应该大于等于0
        valid_values = price_attention.dropna()
        if not valid_values.empty:
            assert all(val >= 0 for val in valid_values)
            assert all(val < 50 for val in valid_values)

    def test_calculate_attention_score(self, sample_market_data):
        """测试综合关注度评分计算"""
        calculator = AttentionFactorCalculator()

        attention_score = calculator.calculate_attention_score(sample_market_data)

        assert isinstance(attention_score, pd.Series)
        assert len(attention_score) == len(sample_market_data)

        # 综合评分应该在合理范围内
        valid_values = attention_score.dropna()
        if not valid_values.empty:
            assert all(val >= 0 for val in valid_values)
            assert all(val <= 100 for val in valid_values)

    def test_calculate_all_factors(self, sample_market_data):
        """测试计算所有关注度因子"""
        calculator = AttentionFactorCalculator()

        results = calculator.calculate(sample_market_data)

        expected_factors = [
            'turnover_attention', 'volume_attention', 'price_attention',
            'attention_score', 'attention_change_rate'
        ]

        for factor in expected_factors:
            assert factor in results
            assert isinstance(results[factor], pd.Series)


class TestSentimentStrengthFactorCalculator:
    """情绪强度因子计算器测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建测试市场数据"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ'] * 30,
            'trade_date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'close': np.random.uniform(10, 15, 30),
            'vol': np.random.uniform(1000000, 5000000, 30)
        })

    def test_calculate_price_momentum_sentiment(self, sample_market_data):
        """测试价格动量情绪计算"""
        calculator = SentimentStrengthFactorCalculator()

        momentum_sentiment = calculator.calculate_price_momentum_sentiment(sample_market_data)

        assert isinstance(momentum_sentiment, pd.Series)
        assert len(momentum_sentiment) == len(sample_market_data)

        # 价格动量应该在合理范围内
        valid_values = momentum_sentiment.dropna()
        if not valid_values.empty:
            assert all(val > -100 for val in valid_values)
            assert all(val < 100 for val in valid_values)

    def test_calculate_volatility_sentiment(self, sample_market_data):
        """测试波动率情绪计算"""
        calculator = SentimentStrengthFactorCalculator()

        volatility_sentiment = calculator.calculate_volatility_sentiment(sample_market_data)

        assert isinstance(volatility_sentiment, pd.Series)
        assert len(volatility_sentiment) == len(sample_market_data)

        # 波动率应该大于等于0
        valid_values = volatility_sentiment.dropna()
        if not valid_values.empty:
            assert all(val >= 0 for val in valid_values)
            assert all(val < 200 for val in valid_values)

    def test_calculate_bullish_bearish_ratio(self, sample_market_data):
        """测试看涨看跌比例计算"""
        calculator = SentimentStrengthFactorCalculator()

        bullish_ratio, bearish_ratio = calculator.calculate_bullish_bearish_ratio(sample_market_data)

        assert isinstance(bullish_ratio, pd.Series)
        assert isinstance(bearish_ratio, pd.Series)
        assert len(bullish_ratio) == len(sample_market_data)
        assert len(bearish_ratio) == len(sample_market_data)

        # 看涨看跌比例之和应该接近100%
        valid_bullish = bullish_ratio.dropna()
        valid_bearish = bearish_ratio.dropna()

        if not valid_bullish.empty and not valid_bearish.empty:
            # 检查比例范围
            assert all(val >= 0 for val in valid_bullish)
            assert all(val <= 100 for val in valid_bullish)
            assert all(val >= 0 for val in valid_bearish)
            assert all(val <= 100 for val in valid_bearish)

    def test_calculate_all_factors(self, sample_market_data):
        """测试计算所有情绪强度因子"""
        calculator = SentimentStrengthFactorCalculator()

        results = calculator.calculate(sample_market_data)

        expected_factors = [
            'price_momentum_sentiment', 'volatility_sentiment', 'sentiment_strength',
            'bullish_ratio', 'bearish_ratio', 'sentiment_volatility'
        ]

        for factor in expected_factors:
            assert factor in results
            assert isinstance(results[factor], pd.Series)


class TestEventFactorCalculator:
    """事件因子计算器测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建测试市场数据"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ'] * 25,
            'trade_date': pd.date_range('2024-01-01', periods=25, freq='D'),
            'open': np.random.uniform(10, 15, 25),
            'close': np.random.uniform(10, 15, 25),
            'pre_close': np.random.uniform(10, 15, 25),
            'vol': np.random.uniform(1000000, 5000000, 25)
        })

    def test_calculate_abnormal_volume(self, sample_market_data):
        """测试异常成交量计算"""
        calculator = EventFactorCalculator()

        abnormal_volume = calculator.calculate_abnormal_volume(sample_market_data)

        assert isinstance(abnormal_volume, pd.Series)
        assert len(abnormal_volume) == len(sample_market_data)

        # 异常成交量应该在合理范围内（Z-score）
        valid_values = abnormal_volume.dropna()
        if not valid_values.empty:
            assert all(val > -10 for val in valid_values)
            assert all(val < 10 for val in valid_values)

    def test_calculate_abnormal_return(self, sample_market_data):
        """测试异常收益率计算"""
        calculator = EventFactorCalculator()

        abnormal_return = calculator.calculate_abnormal_return(sample_market_data)

        assert isinstance(abnormal_return, pd.Series)
        assert len(abnormal_return) == len(sample_market_data)

        # 异常收益率应该在合理范围内
        valid_values = abnormal_return.dropna()
        if not valid_values.empty:
            assert all(val > -10 for val in valid_values)
            assert all(val < 10 for val in valid_values)

    def test_calculate_price_limit_signal(self, sample_market_data):
        """测试涨跌停信号计算"""
        calculator = EventFactorCalculator()

        # 创建包含涨跌停的测试数据
        test_data = sample_market_data.copy()
        test_data.loc[0, 'close'] = test_data.loc[0, 'pre_close'] * 1.1  # 涨停
        test_data.loc[1, 'close'] = test_data.loc[1, 'pre_close'] * 0.9  # 跌停

        limit_signal = calculator.calculate_price_limit_signal(test_data)

        assert isinstance(limit_signal, pd.Series)
        assert len(limit_signal) == len(test_data)

        # 信号应该是-1, 0, 1
        unique_values = set(limit_signal.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_calculate_gap_signal(self, sample_market_data):
        """测试跳空信号计算"""
        calculator = EventFactorCalculator()

        gap_signal = calculator.calculate_gap_signal(sample_market_data)

        assert isinstance(gap_signal, pd.Series)
        assert len(gap_signal) == len(sample_market_data)

        # 跳空信号应该是-1, 0, 1
        unique_values = set(gap_signal.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_calculate_volume_spike(self, sample_market_data):
        """测试成交量异动计算"""
        calculator = EventFactorCalculator()

        volume_spike = calculator.calculate_volume_spike(sample_market_data)

        assert isinstance(volume_spike, pd.Series)
        assert len(volume_spike) == len(sample_market_data)

        # 成交量异动信号应该是0或1
        unique_values = set(volume_spike.unique())
        assert unique_values.issubset({0, 1})

    def test_calculate_all_factors(self, sample_market_data):
        """测试计算所有事件因子"""
        calculator = EventFactorCalculator()

        results = calculator.calculate(sample_market_data)

        expected_factors = [
            'abnormal_volume', 'abnormal_return', 'price_limit_signal',
            'gap_signal', 'volume_spike'
        ]

        for factor in expected_factors:
            assert factor in results
            assert isinstance(results[factor], pd.Series)


class TestSentimentFactorEngine:
    """情绪面因子引擎测试"""

    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()

    @pytest.fixture
    def sample_market_data(self):
        """创建测试市场数据"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ'] * 30,
            'trade_date': pd.to_datetime(pd.date_range('2024-01-01', periods=30, freq='D')),
            'open': np.random.uniform(10, 15, 30),
            'high': np.random.uniform(15, 20, 30),
            'low': np.random.uniform(8, 12, 30),
            'close': np.random.uniform(10, 15, 30),
            'pre_close': np.random.uniform(10, 15, 30),
            'vol': np.random.uniform(1000000, 5000000, 30),
            'amount': np.random.uniform(100000000, 500000000, 30),
            'turnover_rate': np.random.uniform(1, 10, 30),
            'circ_mv': np.random.uniform(1000000, 5000000, 30),
            'total_mv': np.random.uniform(1200000, 6000000, 30)
        })

    def test_initialization(self, mock_engine):
        """测试引擎初始化"""
        engine = SentimentFactorEngine(mock_engine)

        assert engine.factor_type == FactorType.SENTIMENT
        assert len(engine.list_factors()) > 0

        # 检查各类计算器是否初始化
        assert hasattr(engine, 'money_flow_calculator')
        assert hasattr(engine, 'attention_calculator')
        assert hasattr(engine, 'sentiment_calculator')
        assert hasattr(engine, 'event_calculator')

    def test_factor_registration(self, mock_engine):
        """测试因子注册"""
        engine = SentimentFactorEngine(mock_engine)

        # 检查是否注册了预期的因子
        factors = engine.list_factors()

        # 资金流向类因子
        money_flow_factors = ['money_flow_5', 'money_flow_20', 'net_inflow_rate', 'big_order_ratio']
        for factor in money_flow_factors:
            assert factor in factors

        # 关注度类因子
        attention_factors = ['turnover_attention', 'volume_attention', 'attention_score']
        for factor in attention_factors:
            assert factor in factors

        # 情绪强度类因子
        sentiment_factors = ['price_momentum_sentiment', 'sentiment_strength', 'bullish_ratio']
        for factor in sentiment_factors:
            assert factor in factors

        # 事件类因子
        event_factors = ['abnormal_volume', 'abnormal_return', 'price_limit_signal']
        for factor in event_factors:
            assert factor in factors

    @patch('src.compute.sentiment_factor_engine.SentimentFactorEngine.get_required_data')
    def test_calculate_factors_success(self, mock_get_data, mock_engine, sample_market_data):
        """测试因子计算成功"""
        mock_get_data.return_value = sample_market_data

        engine = SentimentFactorEngine(mock_engine)

        result = engine.calculate_factors('000001.SZ')

        assert result.ts_code == '000001.SZ'
        assert result.factor_type == FactorType.SENTIMENT
        assert result.status == CalculationStatus.SUCCESS
        assert result.data_points == len(sample_market_data)
        assert result.execution_time is not None
        assert len(result.factors) > 0

    @patch('src.compute.sentiment_factor_engine.SentimentFactorEngine.get_required_data')
    def test_calculate_factors_no_data(self, mock_get_data, mock_engine):
        """测试无数据情况"""
        mock_get_data.return_value = pd.DataFrame()

        engine = SentimentFactorEngine(mock_engine)

        result = engine.calculate_factors('000001.SZ')

        assert result.status == CalculationStatus.SKIPPED
        assert result.error_message == "无可用市场数据"

    @patch('src.compute.sentiment_factor_engine.SentimentFactorEngine.get_required_data')
    def test_calculate_factors_error(self, mock_get_data, mock_engine):
        """测试计算错误情况"""
        mock_get_data.side_effect = Exception("数据库连接错误")

        engine = SentimentFactorEngine(mock_engine)

        result = engine.calculate_factors('000001.SZ')

        assert result.status == CalculationStatus.FAILED
        assert "数据库连接错误" in result.error_message

    def test_factor_filtering(self, mock_engine, sample_market_data):
        """测试因子过滤功能"""
        with patch.object(SentimentFactorEngine, 'get_required_data', return_value=sample_market_data):
            engine = SentimentFactorEngine(mock_engine)

            # 只计算指定的因子
            specified_factors = ['money_flow_5', 'attention_score', 'sentiment_strength']
            result = engine.calculate_factors('000001.SZ', factor_names=specified_factors)

            assert result.status == CalculationStatus.SUCCESS

            # 检查只返回了指定的因子
            returned_factors = set(result.factors.keys())
            expected_factors = set(specified_factors)

            # 返回的因子应该是指定因子的子集
            assert returned_factors.issubset(expected_factors)


@pytest.mark.performance
class TestSentimentFactorEnginePerformance:
    """情绪面因子引擎性能测试"""

    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        import time

        # 创建大数据集
        large_data = MarketDataFactory.create_basic_market_data(
            periods=TestConstants.LARGE_DATA_POINTS * 10
        )

        calculator = MoneyFlowFactorCalculator()

        start_time = time.time()
        results = calculator.calculate(large_data)
        execution_time = time.time() - start_time

        # 验证性能要求：1000条数据应在1秒内完成
        assert execution_time < 1.0, f"计算时间过长: {execution_time:.2f}秒"
        assert len(results) > 0, "计算结果不应为空"

    def test_memory_usage(self):
        """测试内存使用情况"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行大量计算
        large_data = MarketDataFactory.create_basic_market_data(
            periods=TestConstants.LARGE_DATA_POINTS * 5
        )

        calculator = SentimentStrengthFactorCalculator()
        results = calculator.calculate(large_data)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # 验证内存使用合理（不超过50MB增长）
        assert memory_increase < 50, f"内存使用过多: {memory_increase:.2f}MB"
        assert len(results) > 0, "计算结果不应为空"


@pytest.mark.integration
class TestSentimentFactorEngineIntegration:
    """情绪面因子引擎集成测试"""

    def test_end_to_end_calculation_workflow(self):
        """测试端到端计算工作流"""
        mock_engine = Mock()

        # 创建测试数据
        test_data = MarketDataFactory.create_basic_market_data(periods=50)

        with patch.object(SentimentFactorEngine, 'get_required_data', return_value=test_data):
            engine = SentimentFactorEngine(mock_engine)

            # 执行完整的因子计算流程
            result = engine.calculate_factors('000001.SZ')

            # 验证结果
            assert result.status == CalculationStatus.SUCCESS
            assert result.data_points == len(test_data)
            assert len(result.factors) > 0

            # 验证各类因子都被计算
            factor_names = list(result.factors.keys())
            assert any('money_flow' in name for name in factor_names)
            assert any('attention' in name for name in factor_names)
            assert any('sentiment' in name for name in factor_names)
            assert any('abnormal' in name for name in factor_names)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])