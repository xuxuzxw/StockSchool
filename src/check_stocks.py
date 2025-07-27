#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.db import get_db_engine
from sqlalchemy import text

engine = get_db_engine()

with engine.connect() as conn:
    # 检查stock_basic表总记录数
    result = conn.execute(text('SELECT COUNT(*) FROM stock_basic'))
    total_count = result.fetchone()[0]
    print(f'stock_basic表总记录数: {total_count}')
    
    # 检查上市状态为L的记录数
    result2 = conn.execute(text("SELECT COUNT(*) FROM stock_basic WHERE list_status = 'L'"))
    listed_count = result2.fetchone()[0]
    print(f'上市状态为L的记录数: {listed_count}')
    
    # 检查所有list_status值
    result3 = conn.execute(text('SELECT DISTINCT list_status FROM stock_basic'))
    statuses = [row[0] for row in result3.fetchall()]
    print(f'所有list_status值: {statuses}')
    
    # 如果有数据，显示前几条记录
    if total_count > 0:
        result4 = conn.execute(text('SELECT ts_code, symbol, name, list_status FROM stock_basic LIMIT 5'))
        records = result4.fetchall()
        print('\n前5条记录:')
        for record in records:
            print(f'  {record[0]} - {record[1]} - {record[2]} - {record[3]}')