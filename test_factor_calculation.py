#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool 因子计算功能测试

验证技术因子计算引擎的核心功能
"""

import os
import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 设置测试环境
os.environ['DATABASE_URL'] = f'sqlite:///{project_root}/data/test_stock_data.db'

def create_test_data():
    """创建测试数据"""
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(data_dir / "test_stock_data.db")
    cursor = conn.cursor()
    
    # 创建股票日线数据表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        pre_close REAL,
        change REAL,
        pct_chg REAL,
        vol REAL,
        amount REAL,
        UNIQUE(ts_code, trade_date)
    )
    """)
    
    # 插入测试数据
    test_data = []
    stock_codes = ['000001.SZ', '000002.SZ', '600000.SH']
    
    for ts_code in stock_codes:
        # 生成30天的测试数据
        base_price = 50.0
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            
            # 模拟价格波动
            volatility = 0.02
            trend = 0.001 * i  # 轻微上升趋势
            noise = np.random.normal(0, volatility)
            
            if i == 0:
                close = base_price * (1 + trend + noise)
            else:
                close = test_data[-1][5] * (1 + trend + noise)
            
            open_price = close * (1 + np.random.normal(0, 0.01))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.02)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.02)))
            pre_close = close / (1 + trend + noise) if i > 0 else close
            change = close - pre_close
            pct_chg = (change / pre_close) * 100
            vol = np.random.uniform(100000, 1000000)
            amount = vol * close
            
            test_data.append((
                ts_code, date, open_price, high, low, close, pre_close,
                change, pct_chg, vol, amount
            ))
    
    cursor.executemany("""
    INSERT OR IGNORE INTO stock_daily 
    (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, test_data)
    
    conn.commit()
    conn.close()
    print(f"✅ 测试数据创建完成，共插入 {len(test_data)} 条记录")

def test_technical_indicators():
    """测试技术指标计算"""
    try:
        from src.compute.indicators import TechnicalIndicators
        
        # 获取测试数据
        conn = sqlite3.connect(project_root / "data" / "test_stock_data.db")
        df = pd.read_sql_query("SELECT * FROM stock_daily WHERE ts_code = '000001.SZ' ORDER BY trade_date", conn)
        conn.close()
        
        if df.empty:
            print("❌ 测试数据为空")
            return False
        
        # 测试RSI计算
        rsi = TechnicalIndicators.rsi(df['close'], 14)
        print(f"✅ RSI计算成功: {len(rsi)} 个值")
        
        # 测试MACD计算
        macd_line, signal_line, histogram = TechnicalIndicators.macd(df['close'])
        print(f"✅ MACD计算成功: {len(macd_line)} 个值")
        
        # 测试移动平均线
        ma20 = TechnicalIndicators.sma(df['close'], 20)
        print(f"✅ MA20计算成功: {len(ma20)} 个值")
        
        return True
        
    except Exception as e:
        print(f"❌ 技术指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factor_engine():
    """测试因子引擎"""
    try:
        from src.compute.engine_factory import EngineFactory
        
        engine = EngineFactory.create_technical_engine()
        
        # 测试获取股票数据
        data = engine._get_stock_data('000001.SZ', '2024-01-01', '2024-12-31')
        if data.empty:
            print("❌ 无法获取股票数据")
            return False
        
        print(f"✅ 获取股票数据成功: {len(data)} 条记录")
        
        # 测试因子计算
        results = engine.calculate_factors(
            ts_code='000001.SZ',
            start_date='2024-01-01',
            end_date='2024-12-31',
            factor_names=['MA20', 'RSI14']
        )
        print(f"✅ 因子计算成功: {len(results)} 个结果")
        
        return True
        
    except Exception as e:
        print(f"❌ 因子引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=== StockSchool 因子计算功能测试 ===")
    
    try:
        # 创建测试数据
        create_test_data()
        
        # 测试技术指标
        tech_success = test_technical_indicators()
        
        # 测试因子引擎
        engine_success = test_factor_engine()
        
        if tech_success and engine_success:
            print("\n🎉 所有测试通过！因子计算功能正常")
            return True
        else:
            print("\n❌ 部分测试失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始因子计算测试...")
    
    # 确保表存在
    data_dir = project_root / "data"
    db_path = data_dir / "test_stock_data.db"
    if not db_path.exists():
        print("📦 创建测试数据库...")
        create_test_data()
    
    success = main()
    sys.exit(0 if success else 1)