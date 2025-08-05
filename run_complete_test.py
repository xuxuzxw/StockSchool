#!/usr/bin/env python3
"""
完整因子计算测试脚本
确保SQLite数据库和表结构正确创建
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_sqlite_test_db():
    """设置SQLite测试数据库"""
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    db_path = data_dir / "test_stock_data.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    # 创建表结构
    create_tables_sql = """
    CREATE TABLE IF NOT EXISTS stock_basic (
        ts_code TEXT PRIMARY KEY,
        symbol TEXT,
        name TEXT,
        area TEXT,
        industry TEXT,
        list_date TEXT,
        market TEXT
    );

    CREATE TABLE IF NOT EXISTS trade_calendar (
        exchange TEXT,
        cal_date TEXT,
        is_open INTEGER,
        PRIMARY KEY (exchange, cal_date)
    );

    CREATE TABLE IF NOT EXISTS stock_daily (
        ts_code TEXT,
        trade_date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        pre_close REAL,
        change REAL,
        pct_chg REAL,
        vol REAL,
        amount REAL,
        PRIMARY KEY (ts_code, trade_date)
    );

    CREATE TABLE IF NOT EXISTS technical_factors (
        ts_code TEXT,
        trade_date TEXT,
        factor_name TEXT,
        factor_value REAL,
        PRIMARY KEY (ts_code, trade_date, factor_name)
    );
    """
    
    with engine.connect() as conn:
        for statement in create_tables_sql.split(';'):
            if statement.strip():
                conn.execute(text(statement.strip()))
        conn.commit()
    
    return engine

def insert_test_data(engine):
    """插入测试数据"""
    
    # 插入股票基本信息
    stock_basic_data = pd.DataFrame([{
        'ts_code': '000001.SZ',
        'symbol': '000001',
        'name': '平安银行',
        'area': '深圳',
        'industry': '银行',
        'list_date': '19910403',
        'market': '主板'
    }])
    
    stock_basic_data.to_sql('stock_basic', engine, if_exists='replace', index=False)
    
    # 生成60天的测试行情数据以满足最小数据要求
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(60)]
    
    daily_data = []
    base_price = 10.0
    
    for i, date in enumerate(dates):
        price = base_price + (i * 0.1) + (i % 7 - 3.5) * 0.2
        daily_data.append({
            'ts_code': '000001.SZ',
            'trade_date': date.strftime('%Y%m%d'),
            'open': round(price + 0.1, 2),
            'high': round(price + 0.3, 2),
            'low': round(price - 0.2, 2),
            'close': round(price, 2),
            'pre_close': round(base_price + (i-1)*0.1 if i > 0 else price-0.1, 2),
            'change': round(price - (base_price + (i-1)*0.1 if i > 0 else price-0.1), 2),
            'pct_chg': round(((price - (base_price + (i-1)*0.1 if i > 0 else price-0.1)) / (base_price + (i-1)*0.1 if i > 0 else price-0.1)) * 100, 2),
            'vol': round(1000000 + i * 10000, 0),
            'amount': round((1000000 + i * 10000) * price, 0)
        })
    
    daily_df = pd.DataFrame(daily_data)
    daily_df.to_sql('stock_daily', engine, if_exists='replace', index=False)
    
    # 插入交易日历
    calendar_data = []
    for date in dates:
        calendar_data.append({
            'exchange': 'SSE',
            'cal_date': date.strftime('%Y%m%d'),
            'is_open': 1
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    calendar_df.to_sql('trade_calendar', engine, if_exists='replace', index=False)
    
    return daily_df

def test_indicators():
    """测试技术指标计算"""
    try:
        from src.compute.technical_factor_engine import TechnicalFactorEngine
        
        # 设置环境变量使用SQLite
        os.environ['DATABASE_TYPE'] = 'sqlite'
        
        engine = setup_sqlite_test_db()
        daily_data = insert_test_data(engine)
        
        # 创建因子引擎
        from src.compute.factor_calculation_config import TechnicalFactorCalculationConfig
        config = TechnicalFactorCalculationConfig()
        factor_engine = TechnicalFactorEngine(engine, config)
        
        # 测试数据获取
        stock_data = factor_engine._get_stock_data('000001.SZ', '20240101', '20240130')
        print(f"✅ 数据获取成功: {len(stock_data)} 条记录")
        
        # 测试因子计算
        results = factor_engine.calculate_factors(
            ts_code='000001.SZ',
            start_date='20240101',
            end_date='20240301'  # 覆盖60天数据以满足最小要求
        )
        
        if results:
            print(f"✅ 因子计算成功: {len(results)} 条结果")
            print("前5条结果:")
            for i, result in enumerate(results[:5]):
                print(f"  {i+1}. {result}")
            return True
        else:
            print("❌ 因子计算无结果")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始完整因子计算测试...")
    
    try:
        success = test_indicators()
        
        if success:
            print("\n🎉 所有测试通过！")
            return True
        else:
            print("\n❌ 测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)