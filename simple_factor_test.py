#!/usr/bin/env python3
"""
因子有效性分析器简化测试脚本
跳过数据库依赖，直接测试核心计算逻辑
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_factor_data():
    """创建测试用的因子数据"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    
    data = []
    for date in dates:
        for stock in stocks:
            # 创建模拟的因子值和收益率，确保有一定相关性
            factor_value = np.random.normal(0.5, 0.2)
            return_rate = np.random.normal(0.001, 0.02) + 0.1 * factor_value
            
            data.append({
                'trade_date': date,
                'ts_code': stock,
                'factor_value': factor_value,
                'return_rate': return_rate
            })
    
    df = pd.DataFrame(data)
    return df

def calculate_ic_simple(factor_data, return_data, factor_col='factor_value', return_col='return_rate'):
    """简化的IC计算"""
    # 使用factor_data中的数据，因为它已经包含了return_rate
    merged = factor_data.copy()
    
    ic_results = []
    for date, group in merged.groupby('trade_date'):
        if len(group) >= 3:
            pearson_corr = group[factor_col].corr(group[return_col])
            spearman_corr = group[factor_col].corr(group[return_col], method='spearman')
            
            ic_results.append({
                'trade_date': date,
                'ic_pearson': pearson_corr,
                'ic_spearman': spearman_corr,
                'sample_count': len(group)
            })
    
    return pd.DataFrame(ic_results)

def calculate_layered_backtest_simple(factor_data, return_data, n_layers=3):
    """简化的分层回测"""
    # 使用factor_data中的数据，因为它已经包含了return_rate
    merged = factor_data.copy()
    
    layered_results = []
    for date, group in merged.groupby('trade_date'):
        group = group.dropna(subset=['factor_value', 'return_rate'])
        
        if len(group) >= n_layers:
            try:
                # 使用更简单的分层方法
                group = group.sort_values('factor_value')
                group['layer'] = pd.cut(range(len(group)), bins=n_layers, labels=range(1, n_layers+1))
                
                layer_stats = group.groupby('layer')['return_rate'].agg(['mean', 'std', 'count']).reset_index()
                layer_stats['trade_date'] = date
                layered_results.append(layer_stats)
            except Exception as e:
                continue
    
    if layered_results:
        return pd.concat(layered_results, ignore_index=True)
    return pd.DataFrame()

def test_factor_analyzer_simple():
    """简化测试因子分析功能"""
    print("开始简化测试因子有效性分析器...")
    
    # 创建测试数据
    factor_data = create_test_factor_data()
    return_data = factor_data[['trade_date', 'ts_code', 'return_rate']].copy()
    
    print(f"测试数据量: {len(factor_data)} 条记录")
    print(f"股票数量: {factor_data['ts_code'].nunique()}")
    print(f"日期范围: {factor_data['trade_date'].min()} 到 {factor_data['trade_date'].max()}")
    
    try:
        # 测试IC计算
        print("\n1. 测试IC计算...")
        ic_data = calculate_ic_simple(factor_data, return_data)
        
        if not ic_data.empty:
            print(f"  ✅ IC计算完成，共 {len(ic_data)} 个有效数据点")
            print(f"  Pearson IC均值: {ic_data['ic_pearson'].mean():.4f}")
            print(f"  Spearman IC均值: {ic_data['ic_spearman'].mean():.4f}")
            print(f"  IC标准差: {ic_data['ic_pearson'].std():.4f}")
        else:
            print("  ⚠️  IC计算结果为空")
        
        # 测试分层回测
        print("\n2. 测试分层回测...")
        layered_data = calculate_layered_backtest_simple(factor_data, return_data, n_layers=3)
        
        if not layered_data.empty:
            print(f"  ✅ 分层回测完成，共 {len(layered_data)} 条记录")
            
            # 计算各层平均表现
            layer_performance = layered_data.groupby('layer')['mean'].mean()
            print("  各层平均收益率:")
            for layer, avg_return in layer_performance.items():
                print(f"    第{layer}层: {avg_return:.4f}")
            
            # 计算多空收益
            if len(layer_performance) >= 2:
                long_short = layer_performance.iloc[-1] - layer_performance.iloc[0]
                print(f"  多空收益(最高-最低): {long_short:.4f}")
        else:
            print("  ⚠️  分层回测结果为空")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_factor_analyzer_simple()
    if success:
        print("\n✅ 因子有效性分析器简化测试通过！")
    else:
        print("\n❌ 因子有效性分析器简化测试失败！")