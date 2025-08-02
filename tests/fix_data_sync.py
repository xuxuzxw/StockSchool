#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据同步修复脚本
用于解决历史数据缺失和当日数据同步问题
"""

import os
import sys
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.config.database import get_db_engine
except ImportError:
    try:
        from src.utils.db import get_db_engine
    except ImportError:
        # 如果导入失败，直接创建数据库连接
        def get_db_engine():
            DATABASE_URL = "postgresql://stockschool:stockschool123@localhost:15432/stockschool"
            return create_engine(DATABASE_URL)

try:
    from src.data.tushare_sync import TushareSynchronizer
except ImportError:
    print("无法导入TushareSynchronizer，请检查模块路径")
    sys.exit(1)

def check_missing_trade_dates():
    """检查缺失的交易日数据"""
    engine = get_db_engine()
    
    print("=== 检查缺失的交易日数据 ===")
    
    with engine.connect() as conn:
        # 获取交易日历中的所有交易日
        trade_dates = conn.execute(text("""
            SELECT cal_date 
            FROM trade_calendar 
            WHERE is_open = 1 AND cal_date <= CURRENT_DATE
            ORDER BY cal_date
        """)).fetchall()
        
        # 获取已有数据的交易日
        existing_dates = conn.execute(text("""
            SELECT DISTINCT trade_date 
            FROM stock_daily 
            ORDER BY trade_date
        """)).fetchall()
        
        trade_date_set = {row[0] for row in trade_dates}
        existing_date_set = {row[0] for row in existing_dates}
        
        missing_dates = sorted(trade_date_set - existing_date_set)
        
        print(f"交易日历中的交易日总数: {len(trade_date_set)}")
        print(f"已有数据的交易日数: {len(existing_date_set)}")
        print(f"缺失的交易日数: {len(missing_dates)}")
        
        if missing_dates:
            print("\n缺失的交易日 (最近20个):")
            for date in missing_dates[-20:]:
                print(f"  {date}")
        
        return missing_dates

def sync_missing_data(missing_dates, max_days_per_batch=10):
    """同步缺失的数据"""
    if not missing_dates:
        print("没有缺失的数据需要同步")
        return
    
    print(f"\n=== 开始同步缺失数据 ===")
    print(f"总共需要同步 {len(missing_dates)} 个交易日")
    
    try:
        sync = TushareSynchronizer()
        
        # 分批同步，避免一次性同步太多数据
        for i in range(0, len(missing_dates), max_days_per_batch):
            batch_dates = missing_dates[i:i + max_days_per_batch]
            print(f"\n批次 {i//max_days_per_batch + 1}: 同步 {len(batch_dates)} 个交易日")
            
            for date in batch_dates:
                date_str = date.strftime('%Y%m%d')
                print(f"正在同步 {date_str}...")
                
                try:
                    # 同步日线数据
                    df = sync.pro.daily(trade_date=date_str)
                    
                    if df.empty:
                        print(f"  ⚠️ {date_str} 无数据")
                        continue
                    
                    # 插入数据库
                    engine = get_db_engine()
                    with engine.connect() as conn:
                        for _, row in df.iterrows():
                            conn.execute(text("""
                                INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                                VALUES (:ts_code, :trade_date, :open, :high, :low, :close, :pre_close, :change, :pct_chg, :vol, :amount)
                                ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                                    open = EXCLUDED.open,
                                    high = EXCLUDED.high,
                                    low = EXCLUDED.low,
                                    close = EXCLUDED.close,
                                    pre_close = EXCLUDED.pre_close,
                                    change = EXCLUDED.change,
                                    pct_chg = EXCLUDED.pct_chg,
                                    vol = EXCLUDED.vol,
                                    amount = EXCLUDED.amount
                            """), {
                                'ts_code': row['ts_code'],
                                'trade_date': row['trade_date'],
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'pre_close': row['pre_close'],
                                'change': row['change'],
                                'pct_chg': row['pct_chg'],
                                'vol': row['vol'],
                                'amount': row['amount']
                            })
                        conn.commit()
                    
                    print(f"  ✅ {date_str} 同步完成，{len(df)} 条记录")
                    
                except Exception as e:
                    print(f"  ❌ {date_str} 同步失败: {e}")
                    continue
            
            print(f"批次 {i//max_days_per_batch + 1} 完成")
    
    except Exception as e:
        print(f"同步过程中发生错误: {e}")

def sync_today_data():
    """尝试同步今日数据"""
    print("\n=== 尝试同步今日数据 ===")
    
    today = datetime.now().strftime('%Y%m%d')
    print(f"今日日期: {today}")
    
    try:
        sync = TushareSynchronizer()
        
        # 检查今日是否为交易日
        engine = get_db_engine()
        with engine.connect() as conn:
            is_trade_day = conn.execute(text("""
                SELECT is_open FROM trade_calendar WHERE cal_date = :today
            """), {'today': today}).scalar()
            
            if is_trade_day != 1:
                print(f"今日({today})不是交易日，跳过同步")
                return
        
        print(f"今日({today})是交易日，尝试获取数据...")
        
        # 获取今日数据
        df = sync.pro.daily(trade_date=today)
        
        if df.empty:
            print(f"⚠️ 今日({today})暂无数据，可能数据还未发布")
            return
        
        # 插入数据库
        with engine.connect() as conn:
            for _, row in df.iterrows():
                conn.execute(text("""
                    INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                    VALUES (:ts_code, :trade_date, :open, :high, :low, :close, :pre_close, :change, :pct_chg, :vol, :amount)
                    ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        pre_close = EXCLUDED.pre_close,
                        change = EXCLUDED.change,
                        pct_chg = EXCLUDED.pct_chg,
                        vol = EXCLUDED.vol,
                        amount = EXCLUDED.amount
                """), {
                    'ts_code': row['ts_code'],
                    'trade_date': row['trade_date'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'pre_close': row['pre_close'],
                    'change': row['change'],
                    'pct_chg': row['pct_chg'],
                    'vol': row['vol'],
                    'amount': row['amount']
                })
            conn.commit()
        
        print(f"✅ 今日({today})数据同步完成，{len(df)} 条记录")
        
    except Exception as e:
        print(f"❌ 今日数据同步失败: {e}")

def main():
    """主函数"""
    print("数据同步修复脚本启动")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 检查缺失的交易日数据
    missing_dates = check_missing_trade_dates()
    
    # 2. 尝试同步今日数据
    sync_today_data()
    
    # 3. 询问是否同步历史数据
    if missing_dates:
        print(f"\n发现 {len(missing_dates)} 个交易日的数据缺失")
        
        # 只同步最近的缺失数据（比如最近30个交易日）
        recent_missing = missing_dates[-30:] if len(missing_dates) > 30 else missing_dates
        
        if recent_missing:
            print(f"准备同步最近 {len(recent_missing)} 个缺失的交易日数据")
            print("日期范围:", recent_missing[0], "到", recent_missing[-1])
            
            response = input("\n是否继续同步？(y/n): ")
            if response.lower() == 'y':
                sync_missing_data(recent_missing)
            else:
                print("跳过历史数据同步")
    
    print("\n数据同步修复脚本完成")

if __name__ == "__main__":
    main()