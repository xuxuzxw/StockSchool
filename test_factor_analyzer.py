#!/usr/bin/env python3
"""
因子有效性分析器测试脚本
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compute.factor_effectiveness_analyzer import FactorEffectivenessAnalyzer
from src.compute.factor_effectiveness_analyzer import ReturnPeriod

def create_test_factor_data():
    """创建测试用的因子数据"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    
    data = []
    for date in dates:
        for stock in stocks:
            # 创建模拟的因子值和收益率
            factor_value = np.random.normal(0.5, 0.2)
            return_rate = np.random.normal(0.001, 0.02) + 0.1 * factor_value  # 因子与收益有一定相关性
            
            data.append({
                'trade_date': date,
                'ts_code': stock,
                'factor_value': factor_value,
                'return_rate': return_rate
            })
    
    df = pd.DataFrame(data)
    return df

def create_test_return_data():
    """创建测试用的收益率数据"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    
    data = []
    for date in dates:
        for stock in stocks:
            # 创建模拟的收益率
            return_rate = np.random.normal(0.001, 0.02)
            
            data.append({
                'trade_date': date,
                'ts_code': stock,
                'return_rate': return_rate
            })
    
    df = pd.DataFrame(data)
    return df

def test_factor_analyzer():
    """测试因子有效性分析器"""
    print("开始测试因子有效性分析器...")
    
    # 创建内存数据库引擎用于测试
    engine = create_engine('sqlite:///:memory:')
    
    # 创建分析器实例
    analyzer = FactorEffectivenessAnalyzer(engine)
    
    # 创建测试数据
    factor_data = create_test_factor_data()
    return_data = create_test_return_data()
    
    print(f"因子数据量: {len(factor_data)}")
    print(f"收益率数据量: {len(return_data)}")
    
    try:
        # 测试IC计算
        print("\n1. 测试IC计算...")
        ic_data = analyzer.calculate_ic(
            factor_data=factor_data,
            return_data=return_data,
            factor_col='factor_value',
            return_col='return_rate'
        )
        
        if not ic_data.empty:
            print(f"  IC计算完成，共 {len(ic_data)} 个有效数据点")
            print(f"  Pearson IC均值: {ic_data['ic_pearson'].mean():.4f}")
            print(f"  Spearman IC均值: {ic_data['ic_spearman'].mean():.4f}")
            
            # 测试IC统计
            ic_stats = analyzer.calculate_ic_statistics(ic_data)
            print(f"  IC统计信息: {len(ic_stats)} 项统计指标")
        else:
            print("  警告: IC计算结果为空")
        
        # 测试IR计算
        print("\n2. 测试IR计算...")
        ir_data = analyzer.calculate_ir(ic_data)
        if not ir_data.empty:
            print(f"  IR计算完成")
            print(f"  最终IR值: {ir_data['ir'].iloc[-1]:.4f}")
            
            # 测试IR统计
            ir_stats = analyzer.calculate_ir_statistics(ir_data)
            print(f"  IR统计信息: {len(ir_stats)} 项统计指标")
        else:
            print("  警告: IR计算结果为空")
        
        # 测试分层回测
        print("\n3. 测试分层回测...")
        layered_data = analyzer.factor_layered_backtest(
            factor_data=factor_data,
            return_data=return_data,
            n_layers=5
        )
        
        if not layered_data.empty:
            print(f"  分层回测完成，共 {len(layered_data)} 条记录")
            
            # 测试分层分析
            layer_analysis = analyzer.analyze_layered_performance(layered_data)
            print(f"  分层分析结果: {len(layer_analysis)} 项分析指标")
            
            if 'long_short' in layer_analysis:
                print(f"  多空收益: {layer_analysis['long_short']['return']:.4f}")
        else:
            print("  警告: 分层回测结果为空")
        
        # 测试因子衰减
        print("\n4. 测试因子衰减分析...")
        decay_data = analyzer.calculate_factor_decay(
            factor_data=factor_data,
            return_data=return_data,
            max_periods=5
        )
        
        if not decay_data.empty:
            print(f"  因子衰减分析完成，共 {len(decay_data)} 个周期")
            print(f"  第1期IC: {decay_data.iloc[0]['ic_pearson']:.4f}")
            print(f"  第5期IC: {decay_data.iloc[-1]['ic_pearson']:.4f}")
        else:
            print("  警告: 因子衰减分析结果为空")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_factor_analyzer()
    if success:
        print("\n✅ 因子有效性分析器测试通过！")
    else:
        print("\n❌ 因子有效性分析器测试失败！")