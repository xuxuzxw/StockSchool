#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据库状态脚本
用于诊断数据同步问题
"""

import os
import sys
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
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

def check_database_status():
    """检查数据库状态"""
    engine = get_db_engine()
    
    print("=== 数据库状态检查 ===")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    with engine.connect() as conn:
        # 检查stock_daily表
        print("1. stock_daily表状态:")
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT ts_code) as stock_count,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date
            FROM stock_daily
        """)).fetchone()
        
        if result and result[0] > 0:
            print(f"   总记录数: {result[0]:,}")
            print(f"   股票数量: {result[1]:,}")
            print(f"   最早日期: {result[2]}")
            print(f"   最新日期: {result[3]}")
            
            # 检查最近几天的数据
            recent_data = conn.execute(text("""
                SELECT trade_date, COUNT(*) as stock_count
                FROM stock_daily 
                WHERE trade_date >= (SELECT MAX(trade_date) - INTERVAL '10 days' FROM stock_daily)
                GROUP BY trade_date
                ORDER BY trade_date DESC
            """)).fetchall()
            
            print("   最近10天数据:")
            for row in recent_data:
                print(f"     {row[0]}: {row[1]:,} 只股票")
        else:
            print("   表为空")
        
        print()
        
        # 检查trade_calendar表
        print("2. trade_calendar表状态:")
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                MIN(cal_date) as min_date,
                MAX(cal_date) as max_date,
                COUNT(CASE WHEN is_open = 1 THEN 1 END) as trade_days
            FROM trade_calendar
        """)).fetchone()
        
        if result and result[0] > 0:
            print(f"   总记录数: {result[0]:,}")
            print(f"   最早日期: {result[1]}")
            print(f"   最新日期: {result[2]}")
            print(f"   交易日数: {result[3]:,}")
            
            # 检查今天是否为交易日
            today = datetime.now().strftime('%Y-%m-%d')
            today_status = conn.execute(text("""
                SELECT is_open FROM trade_calendar WHERE cal_date = :today
            """), {'today': today}).scalar()
            
            if today_status is not None:
                print(f"   今天({today})是否交易日: {'是' if today_status == 1 else '否'}")
            else:
                print(f"   今天({today})不在交易日历中")
        else:
            print("   表为空")
        
        print()
        
        # 检查stock_basic表
        print("3. stock_basic表状态:")
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN list_status = 'L' THEN 1 END) as active_stocks
            FROM stock_basic
        """)).fetchone()
        
        if result and result[0] > 0:
            print(f"   总记录数: {result[0]:,}")
            print(f"   活跃股票数: {result[1]:,}")
        else:
            print("   表为空")
        
        print()
        
        # 分析数据同步问题
        print("4. 数据同步问题分析:")
        
        # 检查get_last_trade_date会返回什么
        last_date_result = conn.execute(text("SELECT max(trade_date) FROM stock_daily")).scalar()
        if last_date_result:
            last_date_str = last_date_result.strftime('%Y%m%d')
            print(f"   get_last_trade_date会返回: {last_date_str}")
        else:
            fallback_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            print(f"   get_last_trade_date会返回(表为空): {fallback_date}")
        
        # 检查应该同步的交易日
        print("   应该同步的最近交易日:")
        recent_trade_days = conn.execute(text("""
            SELECT cal_date 
            FROM trade_calendar 
            WHERE is_open = 1 AND cal_date <= CURRENT_DATE
            ORDER BY cal_date DESC 
            LIMIT 10
        """)).fetchall()
        
        for i, row in enumerate(recent_trade_days):
            status = "✓" if i == 0 else " "
            print(f"     {status} {row[0]}")

if __name__ == "__main__":
    check_database_status()