#!/usr/bin/env python3
"""
情绪面因子引擎测试脚本
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compute.sentiment_factor_engine import SentimentFactorEngine
from sqlalchemy import create_engine

def create_test_market_data():
    """创建测试用的市场数据"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    # 创建模拟的市场数据
    data = []
    for i, date in enumerate(dates):
        base_price = 100 + i * 0.5 + np.random.normal(0, 1)
        data.append({
            'trade_date': date,
            'ts_code': '000001.SZ',
            'open': base_price + np.random.normal(0, 0.5),
            'high': base_price + 1 + np.random.normal(0, 0.5),
            'low': base_price - 1 + np.random.normal(0, 0.5),
            'close': base_price,
            'vol': np.random.randint(100000, 1000000),
            'amount': np.random.randint(1000000, 10000000)
        })
    
    df = pd.DataFrame(data)
    df = df.set_index('trade_date')
    return df

def test_sentiment_calculators():
    """直接测试各个计算器"""
    print("开始测试情绪面因子计算器...")
    
    from src.compute.sentiment_factor_engine import (
        MoneyFlowFactorCalculator,
        AttentionFactorCalculator,
        SentimentStrengthFactorCalculator,
        EventFactorCalculator,
        NewsSentimentFactorCalculator
    )
    
    # 创建测试数据
    market_data = create_test_market_data()
    print(f"测试数据创建完成，数据量: {len(market_data)}")
    
    calculators = [
        ("资金流向", MoneyFlowFactorCalculator()),
        ("关注度", AttentionFactorCalculator()),
        ("情绪强度", SentimentStrengthFactorCalculator()),
        ("事件", EventFactorCalculator()),
        ("新闻情感", NewsSentimentFactorCalculator())
    ]
    
    try:
        total_factors = 0
        for name, calculator in calculators:
            print(f"\n测试{name}计算器...")
            factors = calculator.calculate(market_data)
            print(f"  计算了 {len(factors)} 个因子: {list(factors.keys())}")
            total_factors += len(factors)
            
            # 显示前3天的数据
            factor_df = pd.DataFrame(factors)
            print(f"  前3天数据:")
            print(factor_df.head(3))
        
        print(f"\n✅ 所有计算器测试通过！总共计算了 {total_factors} 个因子")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sentiment_calculators()
    if success:
        print("\n✅ 情绪面因子引擎测试通过！")
    else:
        print("\n❌ 情绪面因子引擎测试失败！")