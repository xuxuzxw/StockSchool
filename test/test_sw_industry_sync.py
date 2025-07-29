#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申万行业分类数据同步测试脚本
用于验证申万行业分类数据同步功能的正确性
"""

import sys
import os
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.tushare_sync import TushareSynchronizer
from src.utils.db import get_db_engine

def test_sw_industry_sync():
    """测试申万行业分类数据同步功能"""
    print("开始测试申万行业分类数据同步功能...")
    
    try:
        # 创建同步器实例
        synchronizer = TushareSynchronizer()
        
        # 测试同步一级行业数据
        print("\n1. 测试同步申万一级行业数据...")
        synchronizer.sync_sw_industry(level='L1')
        
        # 验证数据是否正确插入
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM sw_industry_history WHERE sw_l1 IS NOT NULL").fetchone()
            count = result[0]
            print(f"   数据库中一级行业数据条数: {count}")
            
            if count > 0:
                print("   ✅ 一级行业数据同步成功")
            else:
                print("   ❌ 一级行业数据同步失败")
        
        # 测试完整行业数据同步
        print("\n2. 测试完整行业数据同步...")
        synchronizer.sync_sw_industry_full()
        
        # 验证所有层级数据
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM sw_industry_history").fetchone()
            count = result[0]
            print(f"   数据库中行业数据总条数: {count}")
            
            if count > 0:
                print("   ✅ 完整行业数据同步成功")
            else:
                print("   ❌ 完整行业数据同步失败")
        
        # 测试特定股票行业数据更新
        print("\n3. 测试特定股票行业数据更新...")
        # 获取几个测试股票
        with engine.connect() as conn:
            stocks = conn.execute("SELECT ts_code FROM stock_basic LIMIT 3").fetchall()
            stock_list = [stock[0] for stock in stocks]
        
        if stock_list:
            synchronizer.update_sw_industry_for_stocks(stock_list=stock_list)
            print(f"   ✅ 特定股票行业数据更新完成: {stock_list}")
        else:
            print("   ⚠️  没有找到测试股票，跳过特定股票更新测试")
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_sw_industry_sync()