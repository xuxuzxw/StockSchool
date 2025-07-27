#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标计算模块
提供常用的技术分析指标计算功能
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from loguru import logger


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
    def rsi(data: Union[pd.Series, np.ndarray], window: int = 14) -> pd.Series:
        """相对强弱指数 (Relative Strength Index)
        
        Args:
            data: 价格数据
            window: 窗口期，默认14
            
        Returns:
            RSI值
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: Union[pd.Series, np.ndarray], 
             fast_period: int = 12, 
             slow_period: int = 26, 
             signal_period: int = 9) -> pd.DataFrame:
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
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data: Union[pd.Series, np.ndarray], 
                       window: int = 20, 
                       num_std: float = 2.0) -> pd.DataFrame:
        """布林带 (Bollinger Bands)
        
        Args:
            data: 价格数据
            window: 窗口期，默认20
            num_std: 标准差倍数，默认2.0
            
        Returns:
            包含Upper、Middle、Lower的DataFrame
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        middle = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return pd.DataFrame({
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        })
    
    @staticmethod
    def stochastic(high: Union[pd.Series, np.ndarray],
                  low: Union[pd.Series, np.ndarray],
                  close: Union[pd.Series, np.ndarray],
                  k_period: int = 14,
                  d_period: int = 3) -> pd.DataFrame:
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
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            '%K': k_percent,
            '%D': d_percent
        })
    
    @staticmethod
    def atr(high: Union[pd.Series, np.ndarray],
           low: Union[pd.Series, np.ndarray],
           close: Union[pd.Series, np.ndarray],
           window: int = 14) -> pd.Series:
        """平均真实波幅 (Average True Range)
        
        Args:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            window: 窗口期，默认14
            
        Returns:
            ATR值
        """
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
    def williams_r(high: Union[pd.Series, np.ndarray],
                  low: Union[pd.Series, np.ndarray],
                  close: Union[pd.Series, np.ndarray],
                  window: int = 14) -> pd.Series:
        """威廉指标 (Williams %R)
        
        Args:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            window: 窗口期，默认14
            
        Returns:
            Williams %R值
        """
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


class FactorCalculator:
    """因子计算器"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
    def calculate_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量因子
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含动量因子的DataFrame
        """
        result = df.copy()
        
        # 价格动量
        result['momentum_5'] = df['close'].pct_change(5)
        result['momentum_10'] = df['close'].pct_change(10)
        result['momentum_20'] = df['close'].pct_change(20)
        
        # RSI
        result['rsi_14'] = self.indicators.rsi(df['close'], 14)
        
        # 威廉指标
        result['williams_r_14'] = self.indicators.williams_r(
            df['high'], df['low'], df['close'], 14
        )
        
        return result
    
    def calculate_trend_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势因子
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含趋势因子的DataFrame
        """
        result = df.copy()
        
        # 移动平均线
        result['sma_5'] = self.indicators.sma(df['close'], 5)
        result['sma_10'] = self.indicators.sma(df['close'], 10)
        result['sma_20'] = self.indicators.sma(df['close'], 20)
        result['sma_60'] = self.indicators.sma(df['close'], 60)
        
        # EMA
        result['ema_12'] = self.indicators.ema(df['close'], 12)
        result['ema_26'] = self.indicators.ema(df['close'], 26)
        
        # MACD
        macd_data = self.indicators.macd(df['close'])
        result['macd'] = macd_data['MACD']
        result['macd_signal'] = macd_data['Signal']
        result['macd_histogram'] = macd_data['Histogram']
        
        # 价格相对于移动平均线的位置
        result['price_to_sma20'] = df['close'] / result['sma_20'] - 1
        result['price_to_sma60'] = df['close'] / result['sma_60'] - 1
        
        return result
    
    def calculate_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率因子
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含波动率因子的DataFrame
        """
        result = df.copy()
        
        # 历史波动率
        returns = df['close'].pct_change()
        result['volatility_5'] = returns.rolling(5).std() * np.sqrt(252)
        result['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        result['volatility_60'] = returns.rolling(60).std() * np.sqrt(252)
        
        # ATR
        result['atr_14'] = self.indicators.atr(
            df['high'], df['low'], df['close'], 14
        )
        
        # 布林带
        bb_data = self.indicators.bollinger_bands(df['close'])
        result['bb_upper'] = bb_data['Upper']
        result['bb_middle'] = bb_data['Middle']
        result['bb_lower'] = bb_data['Lower']
        result['bb_width'] = (bb_data['Upper'] - bb_data['Lower']) / bb_data['Middle']
        result['bb_position'] = (df['close'] - bb_data['Lower']) / (bb_data['Upper'] - bb_data['Lower'])
        
        return result
    
    def calculate_volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量因子
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含成交量因子的DataFrame
        """
        result = df.copy()
        
        # 成交量移动平均
        result['volume_sma_5'] = self.indicators.sma(df['vol'], 5)
        result['volume_sma_20'] = self.indicators.sma(df['vol'], 20)
        
        # 相对成交量
        result['volume_ratio_5'] = df['vol'] / result['volume_sma_5']
        result['volume_ratio_20'] = df['vol'] / result['volume_sma_20']
        
        # 成交量价格趋势 (VPT)
        price_change = df['close'].pct_change()
        result['vpt'] = (price_change * df['vol']).cumsum()
        
        # 资金流量指标 (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['vol']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(14).sum()
        negative_flow_sum = negative_flow.rolling(14).sum()
        
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        result['mfi'] = 100 - (100 / (1 + money_flow_ratio))
        
        return result
    
    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含所有因子的DataFrame
        """
        logger.info(f"开始计算技术因子，数据量: {len(df)}")
        
        result = df.copy()
        
        # 计算各类因子
        result = self.calculate_momentum_factors(result)
        result = self.calculate_trend_factors(result)
        result = self.calculate_volatility_factors(result)
        result = self.calculate_volume_factors(result)
        
        logger.info(f"技术因子计算完成，共生成 {len(result.columns) - len(df.columns)} 个因子")
        
        return result


if __name__ == "__main__":
    # 测试代码
    import random
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    test_data = pd.DataFrame({
        'trade_date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'vol': np.random.randint(1000000, 10000000, 100)
    })
    
    # 确保high >= low
    test_data['high'] = np.maximum(test_data['high'], test_data['low'])
    
    calculator = FactorCalculator()
    result = calculator.calculate_all_factors(test_data)
    
    print("技术指标计算测试完成")
    print(f"原始数据列数: {len(test_data.columns)}")
    print(f"计算后列数: {len(result.columns)}")
    print(f"新增因子数: {len(result.columns) - len(test_data.columns)}")
    print("\n部分因子示例:")
    factor_cols = [col for col in result.columns if col not in test_data.columns]
    print(result[factor_cols[:10]].tail())