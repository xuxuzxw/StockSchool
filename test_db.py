#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.utils.db import get_db_engine
from sqlalchemy import text

def test_database():
    """测试数据库数据"""
    engine = get_db_engine()
    
    with engine.connect() as conn:
        # 检查股票基本信息表
        result = conn.execute(text('SELECT COUNT(*) FROM stock_basic'))
        stock_count = result.scalar()
        print(f'股票基本信息表记录数: {stock_count}')
        
        # 检查交易日历表
        result = conn.execute(text('SELECT COUNT(*) FROM trade_calendar'))
        calendar_count = result.scalar()
        print(f'交易日历表记录数: {calendar_count}')
        
        # 检查日线数据表
        result = conn.execute(text('SELECT COUNT(*) FROM stock_daily'))
        daily_count = result.scalar()
        print(f'日线数据表记录数: {daily_count}')
        
        # 检查最新交易日期
        result = conn.execute(text('SELECT MAX(trade_date) FROM stock_daily'))
        latest_date = result.scalar()
        print(f'最新交易日期: {latest_date}')
        
        # 检查股票代码示例
        result = conn.execute(text('SELECT ts_code, name FROM stock_basic LIMIT 5'))
        stocks = result.fetchall()
        print('\n股票示例:')
        for stock in stocks:
            print(f'  {stock[0]} - {stock[1]}')

if __name__ == '__main__':
    test_database()