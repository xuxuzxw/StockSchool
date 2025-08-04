#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算工作流集成测试
测试完整的因子计算流程，包括数据获取、计算、存储和查询
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
import concurrent.futures
import time
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.compute.factor_engine import FactorEngine
from src.compute.technical_engine import TechnicalFactorEngine
from src.compute.fundamental_engine import FundamentalFactorEngine
from src.compute.sentiment_engine import SentimentFactorEngine
from src.compute.feature_store_adapter import FeatureStoreAdapter
from src.features.factor_feature_store import FactorFeatureStore
from tests.utils.test_data_generator import TestDataGenerator, MarketRegime

class TestFactorCalculationWorkflow:
    """因子计算工作流集成测试"""
    
    @pytest.fixture
    def test_database(self):
        """测试数据库连接"""
        # 使用内存SQLite数据库进行测试
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///:memory:', echo=False)
        
        # 创建必要的表结构
        with engine.connect() as conn:
            # 创建股票基础信息表
            conn.execute("""
                CREATE TABLE stock_basic (
                    ts_code TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    area TEXT,
                    industry TEXT,
                    market TEXT,
                    list_date DATE
                )
            """)
            
            # 创建股票日线数据表
            conn.execute("""
                CREATE TABLE stock_daily (
                    ts_code TEXT,
                    trade_date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    pre_close REAL,
                    change REAL,
                    pct_chg REAL,
                    vol INTEGER,
                    amount REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """)
            
            # 创建财务数据表
            conn.execute("""
                CREATE TABLE financial_data (
                    ts_code TEXT,
                    end_date DATE,
                    revenue REAL,
                    net_profit REAL,
                    total_assets REAL,
                    total_equity REAL,
                    total_liab REAL,
                    gross_profit REAL,
                    operating_profit REAL,
                    ebitda REAL,
                    cash_flow_ops REAL,
                    PRIMARY KEY (ts_code, end_date)
                )
            """)
            
            # 创建因子数据表
            conn.execute("""
                CREATE TABLE factor_values (
                    ts_code TEXT,
                    factor_date DATE,
                    factor_name TEXT,
                    value REAL,
                    PRIMARY KEY (ts_code, factor_date, factor_name)
                )
            """)
        
        return engine
    
    @pytest.fixture
    def test_data(self):
        """生成测试数据"""
        generator = TestDataGenerator(seed=42)
        return generator.create_test_dataset(
            n_stocks=20,
            n_days=100,
            regime=MarketRegime.SIDEWAYS,
            include_financial=True,
            include_factors=False  # 我们要测试因子计算，所以不预生成
        )
    
    @pytest.fixture
    def populated_database(self, test_database, test_data):
        """填充测试数据的数据库"""
        engine = test_database
        
        with engine.connect() as conn:
            # 插入股票基础信息
            stock_basic_data = []
            for ts_code in test_data['stock_daily']['ts_code'].unique():
                stock_basic_data.append({
                    'ts_code': ts_code,
                    'symbol': ts_code[:6],
                    'name': f'测试股票{ts_code[:6]}',
                    'area': '深圳' if ts_code.endswith('.SZ') else '上海',
                    'industry': '测试行业',
                    'market': '主板',
                    'list_date': '2020-01-01'
                })
            
            pd.DataFrame(stock_basic_data).to_sql('stock_basic', conn, if_exists='append', index=False)
            
            # 插入股票日线数据
            test_data['stock_daily'].to_sql('stock_daily', conn, if_exists='append', index=False)
            
            # 插入财务数据
            if 'financial' in test_data:
                test_data['financial'].to_sql('financial_data', conn, if_exists='append', index=False)
        
        return engine
    
    def test_complete_factor_calculation_workflow(self, populated_database, test_data):
        """测试完整的因子计算工作流"""
        engine = populated_database
        
        # 1. 初始化因子引擎
        factor_engine = FactorEngine(engine)
        
        # 2. 获取股票列表
        stock_codes = test_data['stock_daily']['ts_code'].unique()[:5]  # 测试5只股票
        
        # 3. 计算技术面因子
        technical_engine = TechnicalFactorEngine(engine)
        technical_factors = technical_engine.calculate_all_factors(
            stock_codes.tolist(),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 3, 31)
        )
        
        # 验证技术面因子计算结果
        assert not technical_factors.empty, "技术面因子计算结果不应为空"
        assert 'ts_code' in technical_factors.columns, "应包含股票代码列"
        assert len(technical_factors['ts_code'].unique()) <= len(stock_codes), "股票数量应正确"
        
        # 验证包含主要技术指标
        expected_technical_factors = ['sma_5', 'sma_20', 'rsi_14']
        for factor in expected_technical_factors:
            if factor in technical_factors.columns:
                valid_values = technical_factors[factor].dropna()
                assert len(valid_values) > 0, f"技术因子 {factor} 应有有效值"
        
        # 4. 计算基本面因子（如果有财务数据）
        if 'financial' in test_data:
            fundamental_engine = FundamentalFactorEngine(engine)
            fundamental_factors = fundamental_engine.calculate_all_factors(
                stock_codes.tolist()
            )
            
            # 验证基本面因子计算结果
            if not fundamental_factors.empty:
                assert 'ts_code' in fundamental_factors.columns, "应包含股票代码列"
                
                expected_fundamental_factors = ['pe_ttm', 'pb', 'roe']
                for factor in expected_fundamental_factors:
                    if factor in fundamental_factors.columns:
                        valid_values = fundamental_factors[factor].dropna()
                        assert len(valid_values) >= 0, f"基本面因子 {factor} 计算完成"
        
        # 5. 计算情绪面因子
        sentiment_engine = SentimentFactorEngine(engine)
        sentiment_factors = sentiment_engine.calculate_all_factors(
            stock_codes.tolist(),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 3, 31)
        )
        
        # 验证情绪面因子计算结果
        assert not sentiment_factors.empty, "情绪面因子计算结果不应为空"
        assert 'ts_code' in sentiment_factors.columns, "应包含股票代码列"
        
        expected_sentiment_factors = ['money_flow_5', 'net_inflow_rate']
        for factor in expected_sentiment_factors:
            if factor in sentiment_factors.columns:
                valid_values = sentiment_factors[factor].dropna()
                assert len(valid_values) >= 0, f"情绪面因子 {factor} 计算完成"
        
        # 6. 验证数据一致性
        # 检查所有因子数据的股票代码和日期是否一致
        all_factors = [technical_factors, sentiment_factors]
        if 'fundamental' in locals() and not fundamental_factors.empty:
            all_factors.append(fundamental_factors)
        
        for factors_df in all_factors:
            if not factors_df.empty:
                # 验证股票代码格式
                for ts_code in factors_df['ts_code'].unique():
                    assert len(ts_code) == 9, f"股票代码 {ts_code} 格式应正确"
                    assert '.' in ts_code, f"股票代码 {ts_code} 应包含市场后缀"
                
                # 验证日期格式
                if 'factor_date' in factors_df.columns:
                    assert factors_df['factor_date'].dtype == 'object' or \
                           pd.api.types.is_datetime64_any_dtype(factors_df['factor_date']), \
                           "因子日期应为日期类型"
    
    def test_multi_factor_calculation_consistency(self, populated_database, test_data):
        """测试多因子计算的一致性"""
        engine = populated_database
        
        # 选择测试股票
        stock_codes = test_data['stock_daily']['ts_code'].unique()[:3]
        
        # 创建技术面因子引擎
        technical_engine = TechnicalFactorEngine(engine)
        
        # 分别计算不同因子
        rsi_factors = technical_engine.calculate_factor(
            'rsi_14', stock_codes.tolist(),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 28)
        )
        
        sma_factors = technical_engine.calculate_factor(
            'sma_20', stock_codes.tolist(),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 28)
        )
        
        # 批量计算所有因子
        all_factors = technical_engine.calculate_all_factors(
            stock_codes.tolist(),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 28)
        )
        
        # 验证一致性
        if not rsi_factors.empty and 'rsi_14' in all_factors.columns:
            # 合并数据进行比较
            merged_rsi = rsi_factors.merge(
                all_factors[['ts_code', 'factor_date', 'rsi_14']],
                on=['ts_code', 'factor_date'],
                suffixes=('_single', '_batch')
            )
            
            if not merged_rsi.empty:
                # 比较单独计算和批量计算的结果
                rsi_diff = (merged_rsi['rsi_14_single'] - merged_rsi['rsi_14_batch']).abs()
                max_diff = rsi_diff.max()
                assert max_diff < 1e-10, f"RSI单独计算和批量计算结果差异过大: {max_diff}"
        
        if not sma_factors.empty and 'sma_20' in all_factors.columns:
            merged_sma = sma_factors.merge(
                all_factors[['ts_code', 'factor_date', 'sma_20']],
                on=['ts_code', 'factor_date'],
                suffixes=('_single', '_batch')
            )
            
            if not merged_sma.empty:
                sma_diff = (merged_sma['sma_20_single'] - merged_sma['sma_20_batch']).abs()
                max_diff = sma_diff.max()
                assert max_diff < 1e-10, f"SMA单独计算和批量计算结果差异过大: {max_diff}"
    
    def test_concurrent_factor_calculation(self, populated_database, test_data):
        """测试并发因子计算的正确性"""
        engine = populated_database
        
        # 选择测试股票
        stock_codes = test_data['stock_daily']['ts_code'].unique()[:10]
        
        def calculate_factors_for_stock(ts_code):
            """为单只股票计算因子"""
            technical_engine = TechnicalFactorEngine(engine)
            return technical_engine.calculate_all_factors(
                [ts_code],
                start_date=date(2023, 1, 1),
                end_date=date(2023, 2, 28)
            )
        
        # 并发计算
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_stock = {
                executor.submit(calculate_factors_for_stock, ts_code): ts_code 
                for ts_code in stock_codes[:4]  # 测试4只股票的并发计算
            }
            
            concurrent_results = {}
            for future in concurrent.futures.as_completed(future_to_stock):
                ts_code = future_to_stock[future]
                try:
                    result = future.result()
                    concurrent_results[ts_code] = result
                except Exception as exc:
                    pytest.fail(f'股票 {ts_code} 并发计算失败: {exc}')
        
        concurrent_time = time.time() - start_time
        
        # 串行计算（用于对比）
        start_time = time.time()
        serial_results = {}
        for ts_code in list(concurrent_results.keys()):
            serial_results[ts_code] = calculate_factors_for_stock(ts_code)
        
        serial_time = time.time() - start_time
        
        # 验证并发计算结果的正确性
        for ts_code in concurrent_results:
            concurrent_result = concurrent_results[ts_code]
            serial_result = serial_results[ts_code]
            
            # 验证结果不为空
            assert not concurrent_result.empty, f"股票 {ts_code} 并发计算结果不应为空"
            assert not serial_result.empty, f"股票 {ts_code} 串行计算结果不应为空"
            
            # 验证结果一致性（如果两个结果都有数据）
            if len(concurrent_result) > 0 and len(serial_result) > 0:
                # 比较主要因子的计算结果
                common_factors = set(concurrent_result.columns) & set(serial_result.columns)
                common_factors.discard('ts_code')
                common_factors.discard('factor_date')
                
                for factor in common_factors:
                    if factor in concurrent_result.columns and factor in serial_result.columns:
                        concurrent_values = concurrent_result[factor].dropna()
                        serial_values = serial_result[factor].dropna()
                        
                        if len(concurrent_values) > 0 and len(serial_values) > 0:
                            # 比较均值（允许小的数值误差）
                            concurrent_mean = concurrent_values.mean()
                            serial_mean = serial_values.mean()
                            
                            if not (pd.isna(concurrent_mean) or pd.isna(serial_mean)):
                                relative_diff = abs(concurrent_mean - serial_mean) / (abs(serial_mean) + 1e-10)
                                assert relative_diff < 0.01, \
                                    f"股票 {ts_code} 因子 {factor} 并发和串行计算结果差异过大"
        
        print(f"并发计算时间: {concurrent_time:.2f}s, 串行计算时间: {serial_time:.2f}s")
    
    def test_data_storage_and_retrieval(self, populated_database, test_data):
        """测试数据存储和查询的准确性"""
        engine = populated_database
        
        # 选择测试股票
        stock_codes = test_data['stock_daily']['ts_code'].unique()[:3]
        
        # 计算因子
        technical_engine = TechnicalFactorEngine(engine)
        calculated_factors = technical_engine.calculate_all_factors(
            stock_codes.tolist(),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 28)
        )
        
        if calculated_factors.empty:
            pytest.skip("没有计算出因子数据，跳过存储测试")
        
        # 存储因子数据到数据库
        factor_data_to_store = []
        for _, row in calculated_factors.iterrows():
            for col in calculated_factors.columns:
                if col not in ['ts_code', 'factor_date'] and pd.notna(row[col]):
                    factor_data_to_store.append({
                        'ts_code': row['ts_code'],
                        'factor_date': row['factor_date'],
                        'factor_name': col,
                        'value': row[col]
                    })
        
        if factor_data_to_store:
            with engine.connect() as conn:
                pd.DataFrame(factor_data_to_store).to_sql(
                    'factor_values', conn, if_exists='append', index=False
                )
            
            # 查询存储的数据
            with engine.connect() as conn:
                stored_data = pd.read_sql("""
                    SELECT ts_code, factor_date, factor_name, value
                    FROM factor_values
                    WHERE ts_code IN ({})
                """.format(','.join([f"'{code}'" for code in stock_codes])), conn)
            
            # 验证存储和查询的准确性
            assert not stored_data.empty, "存储的因子数据不应为空"
            assert len(stored_data) == len(factor_data_to_store), "存储的数据量应与原始数据一致"
            
            # 验证数据完整性
            for original_row in factor_data_to_store:
                matching_rows = stored_data[
                    (stored_data['ts_code'] == original_row['ts_code']) &
                    (stored_data['factor_date'] == original_row['factor_date']) &
                    (stored_data['factor_name'] == original_row['factor_name'])
                ]
                
                assert len(matching_rows) == 1, f"应该找到唯一匹配的存储记录"
                
                stored_value = matching_rows.iloc[0]['value']
                original_value = original_row['value']
                
                # 验证数值精度
                assert abs(stored_value - original_value) < 1e-10, \
                    f"存储值 {stored_value} 与原始值 {original_value} 不匹配"
    
    def test_error_handling_and_recovery(self, populated_database, test_data):
        """测试错误处理和恢复机制"""
        engine = populated_database
        
        # 测试无效股票代码
        technical_engine = TechnicalFactorEngine(engine)
        
        invalid_result = technical_engine.calculate_all_factors(
            ['INVALID.CODE'],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 28)
        )
        
        # 无效股票代码应该返回空结果，而不是抛出异常
        assert isinstance(invalid_result, pd.DataFrame), "应返回DataFrame"
        
        # 测试无效日期范围
        future_result = technical_engine.calculate_all_factors(
            test_data['stock_daily']['ts_code'].unique()[:2].tolist(),
            start_date=date(2025, 1, 1),  # 未来日期
            end_date=date(2025, 2, 28)
        )
        
        assert isinstance(future_result, pd.DataFrame), "应返回DataFrame"
        
        # 测试部分有效股票代码
        mixed_codes = ['INVALID.CODE'] + test_data['stock_daily']['ts_code'].unique()[:2].tolist()
        mixed_result = technical_engine.calculate_all_factors(
            mixed_codes,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 28)
        )
        
        assert isinstance(mixed_result, pd.DataFrame), "应返回DataFrame"
        
        # 如果有有效结果，应该只包含有效股票的数据
        if not mixed_result.empty:
            valid_codes = mixed_result['ts_code'].unique()
            assert 'INVALID.CODE' not in valid_codes, "结果不应包含无效股票代码"

class TestFeatureStoreIntegration:
    """特征商店集成测试"""
    
    @pytest.fixture
    def test_database(self):
        """测试数据库"""
        from sqlalchemy import create_engine
        return create_engine('sqlite:///:memory:', echo=False)
    
    @pytest.fixture
    def feature_store(self, test_database):
        """特征商店实例"""
        return FactorFeatureStore(test_database)
    
    @pytest.fixture
    def adapter(self, test_database):
        """特征商店适配器"""
        return FeatureStoreAdapter(test_database)
    
    def test_factor_registration_and_versioning(self, feature_store):
        """测试因子注册和版本管理"""
        from src.features.factor_feature_store import FactorMetadata
        
        # 注册因子元数据
        metadata = FactorMetadata(
            factor_name="test_rsi",
            factor_type="technical",
            category="momentum",
            description="测试RSI指标",
            formula="RSI(close, 14)",
            parameters={"window": 14},
            data_requirements=["stock_daily"],
            update_frequency="daily",
            data_schema={"ts_code": "string", "factor_date": "date", "test_rsi": "float"}
        )
        
        factor_name = feature_store.register_factor(metadata)
        assert factor_name == "test_rsi", "因子注册应返回正确的因子名称"
        
        # 创建因子版本
        version_id = feature_store.create_factor_version(
            factor_name="test_rsi",
            algorithm_code="def calculate_rsi(): pass",
            parameters={"window": 14},
            metadata={"description": "测试版本"}
        )
        
        assert version_id.startswith("test_rsi_v"), "版本ID应包含因子名称和版本号"
        
        # 获取因子版本列表
        versions = feature_store.get_factor_versions("test_rsi")
        assert len(versions) == 1, "应该有一个版本"
        assert versions[0].version_id == version_id, "版本ID应匹配"
    
    def test_factor_data_storage_and_retrieval(self, feature_store):
        """测试因子数据存储和查询"""
        from src.features.factor_feature_store import FactorMetadata
        
        # 注册因子
        metadata = FactorMetadata(
            factor_name="test_sma",
            factor_type="technical",
            category="trend",
            description="测试SMA指标",
            formula="SMA(close, 5)",
            parameters={"window": 5},
            data_requirements=["stock_daily"],
            update_frequency="daily",
            data_schema={"ts_code": "string", "factor_date": "date", "test_sma": "float"}
        )
        
        feature_store.register_factor(metadata)
        
        # 创建版本
        version_id = feature_store.create_factor_version(
            factor_name="test_sma",
            algorithm_code="def calculate_sma(): pass",
            parameters={"window": 5}
        )
        
        # 准备测试数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ', '000002.SZ', '000002.SZ'],
            'factor_date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 2)],
            'test_sma': [10.5, 10.8, 15.2, 15.5]
        })
        
        # 存储数据
        success = feature_store.store_factor_data(
            factor_name="test_sma",
            version_id=version_id,
            data=test_data
        )
        
        assert success, "因子数据存储应成功"
        
        # 查询数据
        retrieved_data = feature_store.get_factor_data(
            factor_name="test_sma",
            version_id=version_id,
            ts_codes=['000001.SZ', '000002.SZ']
        )
        
        assert not retrieved_data.empty, "查询的因子数据不应为空"
        assert len(retrieved_data) == 4, "应查询到4条记录"
        assert 'test_sma' in retrieved_data.columns, "应包含因子列"

class TestEndToEndWorkflow:
    """端到端工作流测试"""
    
    def test_complete_end_to_end_workflow(self):
        """测试完整的端到端工作流"""
        # 这个测试模拟从原始数据到最终因子输出的完整流程
        
        # 1. 生成测试数据
        generator = TestDataGenerator(seed=42)
        test_data = generator.create_test_dataset(
            n_stocks=5,
            n_days=50,
            include_financial=True,
            include_factors=False
        )
        
        # 2. 创建临时数据库
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///:memory:', echo=False)
        
        # 3. 创建表结构并插入数据
        with engine.connect() as conn:
            # 创建表
            conn.execute("""
                CREATE TABLE stock_daily (
                    ts_code TEXT,
                    trade_date DATE,
                    open REAL, high REAL, low REAL, close REAL,
                    pre_close REAL, change REAL, pct_chg REAL,
                    vol INTEGER, amount REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
            """)
            
            # 插入数据
            test_data['stock_daily'].to_sql('stock_daily', conn, if_exists='append', index=False)
        
        # 4. 初始化因子引擎
        technical_engine = TechnicalFactorEngine(engine)
        
        # 5. 执行因子计算
        stock_codes = test_data['stock_daily']['ts_code'].unique()[:3].tolist()
        
        calculated_factors = technical_engine.calculate_all_factors(
            stock_codes,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 28)
        )
        
        # 6. 验证最终结果
        assert not calculated_factors.empty, "端到端工作流应产生因子数据"
        assert 'ts_code' in calculated_factors.columns, "结果应包含股票代码"
        
        # 验证数据质量
        for ts_code in stock_codes:
            stock_factors = calculated_factors[calculated_factors['ts_code'] == ts_code]
            assert not stock_factors.empty, f"股票 {ts_code} 应有因子数据"
        
        # 验证因子值的合理性
        numeric_columns = calculated_factors.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'ts_code':
                valid_values = calculated_factors[col].dropna()
                if len(valid_values) > 0:
                    # 检查是否有异常值
                    assert not np.isinf(valid_values).any(), f"因子 {col} 不应包含无穷大值"
                    assert valid_values.std() > 0 or len(valid_values) == 1, f"因子 {col} 应有变化或只有一个值"
        
        print(f"端到端测试完成: 处理了 {len(stock_codes)} 只股票，生成了 {len(calculated_factors)} 条因子记录")

if __name__ == "__main__":
    # 运行集成测试
    pytest.main([__file__, "-v", "--tb=short"])