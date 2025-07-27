#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查交易日历数据的准确性
验证2025-07-28及周边日期的交易日状态
"""

import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.db import get_db_engine
from sqlalchemy import text

def check_trading_calendar():
    """检查交易日历数据"""
    engine = get_db_engine()
    
    print("=== 交易日历数据检查 ===")
    print(f"当前时间: {datetime.now()}")
    print()
    
    # 检查交易日历表的基本信息
    with engine.connect() as conn:
        # 检查交易日历记录总数
        result = conn.execute(text("SELECT COUNT(*) as count FROM trade_calendar"))
        total_count = result.fetchone()[0]
        print(f"交易日历总记录数: {total_count}")
        
        # 检查最新和最早的日期
        result = conn.execute(text("""
            SELECT 
                MIN(cal_date) as earliest_date,
                MAX(cal_date) as latest_date
            FROM trade_calendar
        """))
        date_range = result.fetchone()
        print(f"日期范围: {date_range[0]} 到 {date_range[1]}")
        print()
        
        # 检查2025年7月最后几天的交易日状态
        print("=== 2025年7月最后几天的交易日状态 ===")
        result = conn.execute(text("""
            SELECT 
                cal_date,
                is_open,
                CASE 
                    WHEN EXTRACT(DOW FROM cal_date) = 0 THEN '周日'
                    WHEN EXTRACT(DOW FROM cal_date) = 1 THEN '周一'
                    WHEN EXTRACT(DOW FROM cal_date) = 2 THEN '周二'
                    WHEN EXTRACT(DOW FROM cal_date) = 3 THEN '周三'
                    WHEN EXTRACT(DOW FROM cal_date) = 4 THEN '周四'
                    WHEN EXTRACT(DOW FROM cal_date) = 5 THEN '周五'
                    WHEN EXTRACT(DOW FROM cal_date) = 6 THEN '周六'
                END as weekday
            FROM trade_calendar 
            WHERE cal_date >= '2025-07-24' AND cal_date <= '2025-07-31'
            ORDER BY cal_date
        """))
        
        calendar_data = result.fetchall()
        for row in calendar_data:
            status = "交易日" if row[1] == 1 else "非交易日"
            print(f"{row[0]} ({row[2]}): {status}")
        
        print()
        
        # 检查2025-07-28具体状态
        print("=== 2025-07-28 详细信息 ===")
        result = conn.execute(text("""
            SELECT 
                cal_date,
                is_open,
                pretrade_date
            FROM trade_calendar 
            WHERE cal_date = '2025-07-28'
        """))
        
        today_info = result.fetchone()
        if today_info:
            status = "交易日" if today_info[1] == 1 else "非交易日"
            print(f"日期: {today_info[0]}")
            print(f"状态: {status}")
            print(f"前一交易日: {today_info[2]}")
        else:
            print("2025-07-28 在交易日历中未找到记录！")
        
        print()
        
        # 检查最近的交易日
        print("=== 最近的交易日 ===")
        result = conn.execute(text("""
            SELECT cal_date
            FROM trade_calendar 
            WHERE is_open = 1 AND cal_date <= '2025-07-28'
            ORDER BY cal_date DESC
            LIMIT 5
        """))
        
        recent_trading_days = result.fetchall()
        print("最近5个交易日:")
        for i, row in enumerate(recent_trading_days, 1):
            print(f"{i}. {row[0]}")
        
        print()
        
        # 检查下一个交易日
        print("=== 下一个交易日 ===")
        result = conn.execute(text("""
            SELECT cal_date
            FROM trade_calendar 
            WHERE is_open = 1 AND cal_date > '2025-07-28'
            ORDER BY cal_date ASC
            LIMIT 1
        """))
        
        next_trading_day = result.fetchone()
        if next_trading_day:
            print(f"下一个交易日: {next_trading_day[0]}")
        else:
            print("未找到2025-07-28之后的交易日")

if __name__ == "__main__":
    check_trading_calendar()