import pandas as pd
import numpy as np
from typing import Optional, Union

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    计算相对强弱指数(RSI)
    
    Args:
        df: 包含'close'列的DataFrame
        window: 计算窗口期，默认14
    
    Returns:
        RSI序列
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    计算简单移动平均线(SMA)
    
    Args:
        df: 数据DataFrame
        window: 计算窗口期
        column: 计算列名
    
    Returns:
        SMA序列
    """
    return df[column].rolling(window=window).mean()

def calculate_ema(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    计算指数移动平均线(EMA)
    
    Args:
        df: 数据DataFrame
        window: 计算窗口期
        column: 计算列名
    
    Returns:
        EMA序列
    """
    return df[column].ewm(span=window).mean()

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    计算MACD指标
    
    Args:
        df: 包含'close'列的DataFrame
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期
    
    Returns:
        包含MACD、信号线和柱状图的DataFrame
    """
    ema_fast = calculate_ema(df, fast)
    ema_slow = calculate_ema(df, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """
    计算布林带
    
    Args:
        df: 包含'close'列的DataFrame
        window: 计算窗口期
        std_dev: 标准差倍数
    
    Returns:
        包含上轨、中轨、下轨的DataFrame
    """
    sma = calculate_sma(df, window)
    std = df['close'].rolling(window=window).std()
    
    return pd.DataFrame({
        'upper': sma + (std * std_dev),
        'middle': sma,
        'lower': sma - (std * std_dev)
    })

def calculate_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    计算随机指标(KD)
    
    Args:
        df: 包含'high', 'low', 'close'列的DataFrame
        k_window: K值计算窗口
        d_window: D值计算窗口
    
    Returns:
        包含K值和D值的DataFrame
    """
    lowest_low = df['low'].rolling(window=k_window).min()
    highest_high = df['high'].rolling(window=k_window).max()
    
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return pd.DataFrame({
        'k': k_percent,
        'd': d_percent
    })

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    计算平均真实波幅(ATR)
    
    Args:
        df: 包含'high', 'low', 'close'列的DataFrame
        window: 计算窗口期
    
    Returns:
        ATR序列
    """
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr

def calculate_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    计算威廉指标(%R)
    
    Args:
        df: 包含'high', 'low', 'close'列的DataFrame
        window: 计算窗口期
    
    Returns:
        威廉指标序列
    """
    highest_high = df['high'].rolling(window=window).max()
    lowest_low = df['low'].rolling(window=window).min()
    
    williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    
    return williams_r

def calculate_momentum(df: pd.DataFrame, window: int = 10, column: str = 'close') -> pd.Series:
    """
    计算动量指标
    
    Args:
        df: 数据DataFrame
        window: 计算窗口期
        column: 计算列名
    
    Returns:
        动量指标序列
    """
    return df[column] - df[column].shift(window)

def calculate_roc(df: pd.DataFrame, window: int = 10, column: str = 'close') -> pd.Series:
    """
    计算变化率(ROC)
    
    Args:
        df: 数据DataFrame
        window: 计算窗口期
        column: 计算列名
    
    Returns:
        ROC序列
    """
    return ((df[column] - df[column].shift(window)) / df[column].shift(window)) * 100

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    计算能量潮指标(OBV)
    
    Args:
        df: 包含'close', 'vol'列的DataFrame
    
    Returns:
        OBV序列
    """
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['vol'].iloc[0]
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['vol'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['vol'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标
    
    Args:
        df: 包含OHLCV数据的DataFrame
    
    Returns:
        包含所有技术指标的DataFrame
    """
    result = df.copy()
    
    # 趋势指标
    result['sma_5'] = calculate_sma(df, 5)
    result['sma_10'] = calculate_sma(df, 10)
    result['sma_20'] = calculate_sma(df, 20)
    result['sma_60'] = calculate_sma(df, 60)
    result['ema_12'] = calculate_ema(df, 12)
    result['ema_26'] = calculate_ema(df, 26)
    
    # MACD
    macd_data = calculate_macd(df)
    result['macd'] = macd_data['macd']
    result['macd_signal'] = macd_data['signal']
    result['macd_histogram'] = macd_data['histogram']
    
    # 布林带
    bb_data = calculate_bollinger_bands(df)
    result['bb_upper'] = bb_data['upper']
    result['bb_middle'] = bb_data['middle']
    result['bb_lower'] = bb_data['lower']
    
    # 振荡指标
    result['rsi'] = calculate_rsi(df)
    
    stoch_data = calculate_stochastic(df)
    result['stoch_k'] = stoch_data['k']
    result['stoch_d'] = stoch_data['d']
    
    result['williams_r'] = calculate_williams_r(df)
    
    # 波动率指标
    result['atr'] = calculate_atr(df)
    
    # 动量指标
    result['momentum'] = calculate_momentum(df)
    result['roc'] = calculate_roc(df)
    
    # 成交量指标
    result['obv'] = calculate_obv(df)
    
    return result

if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils.db import get_db_engine
    from sqlalchemy import text
    
    print("测试技术指标计算模块...")
    
    # 从数据库获取测试数据
    engine = get_db_engine()
    query = """
    SELECT trade_date, open, high, low, close, vol
    FROM stock_daily 
    WHERE ts_code = '000001.SZ'
    ORDER BY trade_date
    LIMIT 100
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if not df.empty:
        print(f"获取到 {len(df)} 条测试数据")
        
        # 计算所有技术指标
        result = calculate_all_indicators(df)
        
        print("\n技术指标计算完成:")
        print(f"- RSI: {result['rsi'].iloc[-1]:.2f}")
        print(f"- MACD: {result['macd'].iloc[-1]:.4f}")
        print(f"- 布林带上轨: {result['bb_upper'].iloc[-1]:.2f}")
        print(f"- ATR: {result['atr'].iloc[-1]:.2f}")
        print(f"- 威廉指标: {result['williams_r'].iloc[-1]:.2f}")
        
        print("\n技术指标计算模块测试完成!")
    else:
        print("未获取到测试数据")