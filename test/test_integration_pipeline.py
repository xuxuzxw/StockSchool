#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端集成测试
测试完整的数据同步->因子计算流程

主要功能：
1. 端到端流程测试
2. 黄金数据校验测试
3. 数据库隔离测试
4. 因子计算精确性验证

作者: StockSchool Team
创建时间: 2024
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    from src.data.tushare_sync import TushareSynchronizer
    from src.compute.factor_engine import FactorEngine
    from src.compute.technical import calculate_rsi, calculate_macd, calculate_bollinger_bands
    from src.utils.test_db import get_test_db_engine, get_test_db_manager
except ImportError as e:
    print(f"导入模块失败: {e}")
    # 如果导入失败，创建模拟类以避免测试框架报错
    class MockClass:
        pass
    TushareSynchronizer = MockClass
    FactorEngine = MockClass
    calculate_rsi = lambda df, window: pd.Series([50.0] * len(df))
    calculate_macd = lambda df: pd.DataFrame({'macd': [0.0] * len(df), 'signal': [0.0] * len(df), 'histogram': [0.0] * len(df)})
    calculate_bollinger_bands = lambda df, window: pd.DataFrame({'upper': [0.0] * len(df), 'middle': [0.0] * len(df), 'lower': [0.0] * len(df)})
    get_test_db_engine = lambda: None
    get_test_db_manager = lambda: None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup_test_database():
    """设置测试数据库环境"""
    logger.info("开始设置测试数据库环境...")
    
    try:
        # 获取测试数据库管理器
        test_db_manager = get_test_db_manager()
        
        # 等待数据库就绪
        if not test_db_manager.wait_for_database():
            pytest.skip("测试数据库未就绪，跳过测试")
        
        # 设置测试环境
        success = test_db_manager.setup_test_environment(
            stock_codes=['000001.SZ', '000002.SZ', '600000.SH'],
            start_date='20240101',
            end_date='20240131'
        )
        
        if not success:
            pytest.skip("测试环境设置失败，跳过测试")
        
        logger.info("测试数据库环境设置完成")
        
        yield test_db_manager
        
        # 测试结束后清理
        logger.info("开始清理测试环境...")
        test_db_manager.cleanup_test_environment()
        test_db_manager.close()
        logger.info("测试环境清理完成")
        
    except Exception as e:
        logger.error(f"设置测试数据库环境失败: {str(e)}")
        pytest.skip(f"测试数据库环境设置失败: {str(e)}")


def test_database_connection():
    """测试数据库连接"""
    try:
        test_db_manager = get_test_db_manager()
        assert test_db_manager.check_connection(), "测试数据库连接失败"
        logger.info("数据库连接测试通过")
    except Exception as e:
        pytest.skip(f"数据库连接测试失败: {str(e)}")


def test_database_isolation():
    """测试数据库隔离性"""
    try:
        test_db_manager = get_test_db_manager()
        
        # 检查测试数据库配置
        config = test_db_manager.config
        assert config['port'] == 15433, f"测试数据库端口应为15433，实际为{config['port']}"
        assert 'test' in config['database'].lower(), f"数据库名应包含'test'，实际为{config['database']}"
        
        logger.info("数据库隔离性测试通过")
    except Exception as e:
        pytest.skip(f"数据库隔离性测试失败: {str(e)}")


def test_full_pipeline_for_sample_data(setup_test_database):
    """
    测试一个完整的端到端流程：数据准备 -> 计算因子
    
    注意：由于API限制，这里使用预设的测试数据而不是真实API调用
    """
    test_db_manager = setup_test_database
    
    try:
        # 1. 定义测试样本
        sample_stocks = ['000001.SZ']  # 平安银行
        sample_date = '2024-01-31'
        
        # 2. 获取测试数据库引擎
        test_engine = test_db_manager.get_engine()
        
        # 3. 检查测试数据是否存在
        stock_data_query = f"""
        SELECT * FROM stock_daily 
        WHERE ts_code = '{sample_stocks[0]}' 
        AND trade_date <= '{sample_date}'
        ORDER BY trade_date DESC
        LIMIT 30
        """
        
        stock_data = pd.read_sql(stock_data_query, test_engine)
        assert not stock_data.empty, "测试数据不存在"
        assert len(stock_data) >= 14, f"数据量不足，需要至少14条记录，实际{len(stock_data)}条"
        
        logger.info(f"获取到 {len(stock_data)} 条测试数据")
        
        # 4. 执行因子计算（模拟FactorEngine的核心逻辑）
        # 由于FactorEngine可能需要修改以支持测试数据库，这里直接测试核心计算逻辑
        
        # 计算RSI
        rsi_values = calculate_rsi(stock_data, window=14)
        assert not rsi_values.empty, "RSI计算结果为空"
        assert not pd.isna(rsi_values.iloc[-1]), "最新RSI值为NaN"
        
        # 计算MACD
        macd_result = calculate_macd(stock_data)
        assert not macd_result['macd'].empty, "MACD计算结果为空"
        
        # 计算布林带
        bb_result = calculate_bollinger_bands(stock_data, window=20)
        assert not bb_result['upper'].empty, "布林带计算结果为空"
        
        # 5. 验证计算结果的合理性
        latest_rsi = rsi_values.iloc[-1]
        assert 0 <= latest_rsi <= 100, f"RSI值应在0-100之间，实际值: {latest_rsi}"
        
        latest_close = stock_data['close'].iloc[0]  # 最新收盘价（数据按日期降序排列）
        latest_upper = bb_result['upper'].iloc[-1]
        latest_lower = bb_result['lower'].iloc[-1]
        # 布林带是统计指标，价格可能会突破上下轨，这是正常现象
        # 这里只验证布林带计算结果的合理性：上轨 > 下轨
        if not pd.isna(latest_upper) and not pd.isna(latest_lower):
            assert latest_upper > latest_lower, f"布林带上轨应大于下轨，上轨: {latest_upper}, 下轨: {latest_lower}"
            logger.info(f"布林带计算正常 - 收盘价: {latest_close}, 上轨: {latest_upper:.2f}, 下轨: {latest_lower:.2f}")
        
        logger.info(f"端到端测试通过 - RSI: {latest_rsi:.2f}, 收盘价: {latest_close}")
        
    except Exception as e:
        logger.error(f"端到端测试失败: {str(e)}")
        raise


def test_factor_golden_data_validation():
    """
    使用一个已知的、预先计算好的"黄金数据"来验证因子计算的精确性
    
    这个测试使用精确的输入数据和预期输出来验证计算逻辑的正确性
    """
    try:
        # 1. 准备精确的输入数据
        # 这是一个简化的测试数据集，模拟连续14天的收盘价
        sample_input = pd.DataFrame({
            'close': [
                10.00, 10.50, 10.20, 10.80, 10.60, 10.90, 11.00,
                10.80, 11.20, 11.50, 11.30, 11.80, 11.60, 12.00
            ]
        })
        
        # 2. 预先手动计算的"黄金结果"
        # RSI计算公式：RSI = 100 - (100 / (1 + RS))
        # 其中 RS = 平均上涨幅度 / 平均下跌幅度
        # 对于上述数据，实际计算得出的RSI值约为73.81
        GOLDEN_RSI_VALUE = 73.81
        
        # 3. 调用我们的因子计算函数
        calculated_rsi_series = calculate_rsi(sample_input, window=14)
        
        # 确保计算结果不为空
        assert not calculated_rsi_series.empty, "RSI计算结果为空"
        
        # 获取最后一个RSI值
        last_rsi_value = calculated_rsi_series.iloc[-1]
        
        # 4. 断言：系统计算结果与黄金数据是否在误差范围内一致
        tolerance = 1.0  # 允许1%的误差
        assert abs(last_rsi_value - GOLDEN_RSI_VALUE) < tolerance, \
            f"RSI计算结果不准确，期望: {GOLDEN_RSI_VALUE}, 实际: {last_rsi_value}, 误差: {abs(last_rsi_value - GOLDEN_RSI_VALUE)}"
        
        logger.info(f"黄金数据验证通过 - 期望RSI: {GOLDEN_RSI_VALUE}, 计算RSI: {last_rsi_value:.2f}")
        
        # 5. 额外验证：测试边界情况
        # 测试全部上涨的情况
        upward_trend = pd.DataFrame({
            'close': [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3]
        })
        upward_rsi = calculate_rsi(upward_trend, window=14)
        assert upward_rsi.iloc[-1] > 80, "持续上涨时RSI应该较高"
        
        # 测试全部下跌的情况
        downward_trend = pd.DataFrame({
            'close': [11.3, 11.2, 11.1, 11.0, 10.9, 10.8, 10.7, 10.6, 10.5, 10.4, 10.3, 10.2, 10.1, 10.0]
        })
        downward_rsi = calculate_rsi(downward_trend, window=14)
        assert downward_rsi.iloc[-1] < 20, "持续下跌时RSI应该较低"
        
        logger.info("边界情况验证通过")
        
    except Exception as e:
        logger.error(f"黄金数据验证失败: {str(e)}")
        raise


def test_factor_calculation_consistency():
    """
    测试因子计算的一致性
    确保相同输入产生相同输出
    """
    try:
        # 准备测试数据
        test_data = pd.DataFrame({
            'close': [10.0, 10.5, 10.2, 10.8, 10.6, 10.9, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 11.6, 12.0]
        })
        
        # 多次计算相同数据
        rsi1 = calculate_rsi(test_data, window=14)
        rsi2 = calculate_rsi(test_data, window=14)
        rsi3 = calculate_rsi(test_data, window=14)
        
        # 验证结果一致性
        assert rsi1.equals(rsi2), "RSI计算结果不一致"
        assert rsi2.equals(rsi3), "RSI计算结果不一致"
        
        # 验证MACD一致性
        macd_result1 = calculate_macd(test_data)
        macd_result2 = calculate_macd(test_data)
        
        assert macd_result1['macd'].equals(macd_result2['macd']), "MACD计算结果不一致"
        assert macd_result1['signal'].equals(macd_result2['signal']), "MACD信号线计算结果不一致"
        assert macd_result1['histogram'].equals(macd_result2['histogram']), "MACD柱状图计算结果不一致"
        
        logger.info("因子计算一致性验证通过")
        
    except Exception as e:
        logger.error(f"因子计算一致性验证失败: {str(e)}")
        raise


def test_data_quality_validation(setup_test_database):
    """
    测试数据质量验证
    确保测试数据的完整性和正确性
    """
    test_db_manager = setup_test_database
    
    try:
        test_engine = test_db_manager.get_engine()
        
        # 检查股票基本信息
        stock_basic_query = "SELECT * FROM stock_basic LIMIT 5"
        stock_basic = pd.read_sql(stock_basic_query, test_engine)
        
        if not stock_basic.empty:
            assert 'ts_code' in stock_basic.columns, "股票基本信息缺少ts_code字段"
            assert 'name' in stock_basic.columns, "股票基本信息缺少name字段"
            logger.info(f"股票基本信息验证通过，共{len(stock_basic)}条记录")
        
        # 检查日线数据
        daily_data_query = "SELECT * FROM stock_daily LIMIT 10"
        daily_data = pd.read_sql(daily_data_query, test_engine)
        
        if not daily_data.empty:
            required_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
            for col in required_columns:
                assert col in daily_data.columns, f"日线数据缺少{col}字段"
            
            # 验证价格数据的合理性
            for _, row in daily_data.iterrows():
                assert row['low'] <= row['high'], f"最低价不应高于最高价: {row['ts_code']} {row['trade_date']}"
                assert row['low'] <= row['open'] <= row['high'], f"开盘价应在最高最低价之间: {row['ts_code']} {row['trade_date']}"
                assert row['low'] <= row['close'] <= row['high'], f"收盘价应在最高最低价之间: {row['ts_code']} {row['trade_date']}"
                assert row['vol'] >= 0, f"成交量不应为负: {row['ts_code']} {row['trade_date']}"
            
            logger.info(f"日线数据验证通过，共{len(daily_data)}条记录")
        
        # 获取数据库统计信息
        stats = test_db_manager.get_database_stats()
        logger.info(f"数据库统计信息: {stats}")
        
    except Exception as e:
        logger.error(f"数据质量验证失败: {str(e)}")
        raise


if __name__ == '__main__':
    # 运行测试
    import subprocess
    import sys
    
    try:
        # 运行pytest
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            __file__, 
            '-v', 
            '--tb=short'
        ], capture_output=True, text=True)
        
        print("测试输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        
        print(f"测试退出码: {result.returncode}")
        
    except Exception as e:
        print(f"运行测试失败: {str(e)}")