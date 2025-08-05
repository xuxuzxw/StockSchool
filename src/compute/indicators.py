from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from src.config.unified_config import config

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标计算模块
提供常用的技术分析指标计算功能
"""


class TechnicalIndicators:
    """技术指标计算类"""

    @staticmethod
    def sma(data: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
        """简单移动平均线 (Simple Moving Average)

        Args:
            data: 价格数据
            window: 窗口期

        Returns:
            SMA值
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        return data.rolling(window=window).mean()

    @staticmethod
    def ema(data: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
        """指数移动平均线 (Exponential Moving Average)

        Args:
            data: 价格数据
            window: 窗口期

        Returns:
            EMA值
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        return data.ewm(span=window).mean()

    @staticmethod
    def rsi(data: Union[pd.Series, np.ndarray], window: int = None) -> pd.Series:
        """相对强弱指数 (Relative Strength Index)

        Args:
            data: 价格数据
            window: 窗口期，默认14

        Returns:
            RSI值
        """
        if window is None:
            window = config.get("factor_params.rsi.window", 14)
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        hundred = 100  # 常数，不需要外部化
        rsi = hundred - (hundred / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        data: Union[pd.Series, np.ndarray], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> pd.DataFrame:
        """MACD指标 (Moving Average Convergence Divergence)

        Args:
            data: 价格数据
            fast_period: 快线周期，默认12
            slow_period: 慢线周期，默认26
            signal_period: 信号线周期，默认9

        Returns:
            包含MACD、Signal、Histogram的DataFrame
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)

        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Histogram": histogram})

    @staticmethod
    def bollinger_bands(data: Union[pd.Series, np.ndarray], window: int = None, num_std: float = 2.0) -> pd.DataFrame:
        """布林带 (Bollinger Bands)

        Args:
            data: 价格数据
            window: 窗口期，默认20
            num_std: 标准差倍数，默认2.0

        Returns:
            包含Upper、Middle、Lower的DataFrame
        """
        if window is None:
            window = config.get("factor_params.bollinger.window", 20)
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        middle = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return pd.DataFrame({"Upper": upper, "Middle": middle, "Lower": lower})

    @staticmethod
    def stochastic(
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        k_period: int = None,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """随机指标 (Stochastic Oscillator)

        Args:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            k_period: K线周期，默认14
            d_period: D线周期，默认3

        Returns:
            包含%K、%D的DataFrame
        """
        if k_period is None:
            k_period = config.get("factor_params.stochastic.k_period", 14)
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)

    @staticmethod
    def williams_r(
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        window: int = None,
    ) -> pd.Series:
        """威廉指标 (Williams %R)

        Args:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            window: 窗口期，默认14

        Returns:
            Williams %R值
        """
        if window is None:
            window = config.get("factor_params.williams_r.window", 14)
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)

        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))

        return williams_r

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range (ATR)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = np.max([high_low, high_close, low_close], axis=0)
        return TechnicalIndicators.sma(tr, window)

    @staticmethod
    def vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Price Trend (VPT)"""
        return (volume * close.pct_change()).cumsum()

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index (MFI)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0))
        negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0))

        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()

        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    @staticmethod
    def momentum(data: Union[pd.Series, np.ndarray], window: int = None) -> pd.Series:
        """动量指标 (Momentum)

        Args:
            data: 价格数据
            window: 窗口期，默认10

        Returns:
            Momentum值
        """
        if window is None:
            window = config.get("factor_params.momentum.window", 10)
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        return data.diff(window)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        hundred = 100  # 常数，不需要外部化
        k_percent = hundred * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return pd.DataFrame({"%K": k_percent, "%D": d_percent})

    @staticmethod
    def atr(
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        window: int = None,
    ) -> pd.Series:
        """平均真实波幅 (Average True Range)

        Args:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            window: 窗口期，默认14

        Returns:
            ATR值
        """
        if window is None:
            window = config.get("factor_params.atr.window", 14)
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        return atr

    @staticmethod
    def williams_r(
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        window: int = None,
    ) -> pd.Series:
        """威廉指标 (Williams %R)

        Args:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            window: 窗口期，默认14

        Returns:
            Williams %R值
        """
        if window is None:
            window = config.get("factor_params.williams.window", 14)
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)

        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()

        hundred = 100  # 常数，不需要外部化
        williams_r = -hundred * ((highest_high - close) / (highest_high - lowest_low))

        return williams_r


class FundamentalIndicators:
    """基本面指标计算类 - 已重构为Calculator类，保留类名用于向后兼容

    注意：所有计算方法已迁移到以下Calculator类：
    - ProfitabilityFactorCalculator: roe, roa, gross_margin, net_margin
    - QualityFactorCalculator: debt_to_equity, current_ratio, quick_ratio
    - GrowthFactorCalculator: revenue_yoy, net_profit_yoy

    请使用对应的Calculator类进行因子计算。
    """

    pass

    @classmethod
    def calculate_gross_margin(cls, revenue: float, cost_of_sales: float) -> float:
        """计算毛利率 - 已弃用，请使用ProfitabilityFactorCalculator"""
        return ((revenue - cost_of_sales) / revenue) * 100 if revenue != 0 else np.nan

    @classmethod
    def calculate_net_margin(cls, net_profit: float, revenue: float) -> float:
        """计算净利率 - 已弃用，请使用ProfitabilityFactorCalculator"""
        return (net_profit / revenue) * 100 if revenue != 0 else np.nan

    @classmethod
    def calculate_debt_to_equity(cls, total_liab: float, total_equity: float) -> float:
        """计算资产负债率 - 已弃用，请使用QualityFactorCalculator"""
        return (total_liab / total_equity) * 100 if total_equity != 0 else np.nan

    @classmethod
    def calculate_current_ratio(cls, current_assets: float, current_liab: float) -> float:
        """计算流动比率 - 已弃用，请使用QualityFactorCalculator"""
        return current_assets / current_liab if current_liab != 0 else np.nan

    @classmethod
    def calculate_quick_ratio(cls, current_assets: float, inventories: float, current_liab: float) -> float:
        """计算速动比率 - 已弃用，请使用QualityFactorCalculator"""
        return (current_assets - (inventories or 0)) / current_liab if current_liab != 0 else np.nan


if __name__ == "__main__":
    # 测试代码
    import random

    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    test_data = pd.DataFrame(
        {
            "trade_date": dates,
            "open": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "high": 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
            "low": 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
            "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "vol": np.random.randint(1000000, 10000000, 100),
        }
    )

    # 确保high >= low
    test_data["high"] = np.maximum(test_data["high"], test_data["low"])

    calculator = FactorCalculator()
    result = calculator.calculate_all_factors(test_data)

    print("技术指标计算测试完成")
    print(f"原始数据列数: {len(test_data.columns)}")
    print(f"计算后列数: {len(result.columns)}")
    print(f"新增因子数: {len(result.columns) - len(test_data.columns)}")
    print("\n部分因子示例:")
    factor_cols = [col for col in result.columns if col not in test_data.columns]
    print(result[factor_cols[:10]].tail())

    # 测试基本面因子计算
    print("\n\n基本面因子计算测试:")

    # 生成测试财务数据
    fundamental_test_data = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"] * 4,
            "end_date": [
                "2023-03-31",
                "2023-03-31",
                "2023-06-30",
                "2023-06-30",
                "2023-09-30",
                "2023-09-30",
                "2023-12-31",
                "2023-12-31",
            ],
            "revenue": [1000000, 800000, 1100000, 850000, 1200000, 900000, 1300000, 950000],
            "net_profit": [100000, 80000, 110000, 85000, 120000, 90000, 130000, 95000],
            "total_assets": [5000000, 4000000, 5100000, 4100000, 5200000, 4200000, 5300000, 4300000],
            "total_hldr_eqy_exc_min_int": [2000000, 1600000, 2100000, 1700000, 2200000, 1800000, 2300000, 1900000],
            "oper_cost": [800000, 640000, 880000, 680000, 960000, 720000, 1040000, 760000],
        }
    )

    fundamental_calculator = FundamentalFactorCalculator()
    fundamental_result = fundamental_calculator.calculate_all_fundamental_factors(fundamental_test_data)

    print(f"基本面因子计算完成，共生成 {len(fundamental_result.columns) - len(fundamental_test_data.columns)} 个因子")
    print("\n基本面因子示例:")
    fundamental_factor_cols = [col for col in fundamental_result.columns if col not in fundamental_test_data.columns]
    print(fundamental_result[["ts_code", "end_date"] + fundamental_factor_cols].head())
