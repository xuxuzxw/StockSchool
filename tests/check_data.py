#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

from src.utils.db import get_db_engine
from sqlalchemy import text
from datetime import datetime, timedelta

def check_stock_data():
    """检查数据库中的股票数据情况"""
    engine = get_db_engine()
    
    with engine.connect() as conn:
        # 检查股票基本信息
        result = conn.execute(text("""
            SELECT COUNT(*) as stock_count 
            FROM stock_basic 
            WHERE list_status = 'L'
        """))
        stock_count = result.fetchone()[0]
        print(f"活跃股票数量: {stock_count}")
        
        # 检查日线数据
        result = conn.execute(text("""
            SELECT 
                ts_code,
                COUNT(*) as data_count,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date
            FROM stock_daily 
            WHERE ts_code = '000001.SZ'
            GROUP BY ts_code
        """))
        
        row = result.fetchone()
        if row:
            print(f"\n股票 000001.SZ 数据情况:")
            print(f"  数据条数: {row[1]}")
            print(f"  最早日期: {row[2]}")
            print(f"  最新日期: {row[3]}")
        else:
            print("\n未找到 000001.SZ 的数据")
        
        # 检查最近60个交易日的数据
        result = conn.execute(text("""
            SELECT COUNT(*) as recent_count
            FROM stock_daily 
            WHERE ts_code = '000001.SZ'
            AND trade_date >= (
                SELECT cal_date 
                FROM trade_calendar 
                WHERE is_open = 1 
                ORDER BY cal_date DESC 
                LIMIT 1 OFFSET 59
            )
        """))
        
        recent_count = result.fetchone()[0]
        print(f"  最近60个交易日数据: {recent_count} 条")
        
        # 检查交易日历
        result = conn.execute(text("""
            SELECT COUNT(*) as calendar_count
            FROM trade_calendar 
            WHERE is_open = 1
        """))
        
        calendar_count = result.fetchone()[0]
        print(f"\n交易日历记录数: {calendar_count}")
        
        # 检查最新的交易日
        result = conn.execute(text("""
            SELECT cal_date
            FROM trade_calendar 
            WHERE is_open = 1
            ORDER BY cal_date DESC 
            LIMIT 5
        """))
        
        print("\n最近5个交易日:")
        for row in result:
            print(f"  {row[0]}")

if __name__ == "__main__":
    check_stock_data()