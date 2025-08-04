#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面因子计算单元测试
测试估值、盈利能力等基本面指标的计算正确性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.compute.fundamental_engine import FundamentalFactorEngine
from src.compute.fundamental import (
    ValuationFactorCalculator, ProfitabilityFactorCalculator
)

# 忽略pandas警告
warnings.filterwarnings('ignore', category=FutureWarning)

class TestFundamentalDataGenerator:
    """基本面测试数据生成器"""
    
    @staticmethod
    def generate_market_data(length=20, start_price=100.0):
        """生成市场数据"""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='D')
        prices = start_price + np.random.normal(0, 5, length)
        
        return pd.DataFrame({
            'ts_code': '000001.SZ',
            'trade_date': dates,
            'close': prices,
            'total_mv': prices * 1000000,  # 总市值
            'circ_mv': prices * 800000     # 流通市值
        })
    
    @staticmethod
    def generate_financial_data():
        """生成财务数据"""
        return pd.DataFrame({
            'ts_code': '000001.SZ',
            'end_date': ['2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31'],
            'revenue': [1000000, 750000, 500000, 250000],
            'net_profit': [100000, 75000, 50000, 25000],
            'total_assets': [5000000, 4800000, 4600000, 4400000],
            'total_equity': [2000000, 1900000, 1800000, 1700000],
            'total_liab': [3000000, 2900000, 2800000, 2700000],
            'gross_profit': [300000, 225000, 150000, 75000],
            'operating_profit': [150000, 112500, 75000, 37500],
            'ebitda': [200000, 150000, 100000, 50000],
            'cash_flow_ops': [120000, 90000, 60000, 30000]
        })

class TestValuationFactorCalculator:
    """估值因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return ValuationFactorCalculator(mock_engine)
    
    @pytest.fixture
    def market_data(self):
        """市场数据"""
        return TestFundamentalDataGenerator.generate_market_data()
    
    @pytest.fixture
    def financial_data(self):
        """财务数据"""
        return TestFundamentalDataGenerator.generate_financial_data()
    
    def test_calculate_pe_ttm(self, calculator, market_data, financial_data):
        """测试市盈率TTM计算"""
        pe_ttm = calculator.calculate_pe_ttm(market_data, financial_data)
        
        # 基本验证
        assert not pe_ttm.empty, "PE TTM结果不应为空"
        assert 'pe_ttm' in pe_ttm.columns, "应包含pe_ttm列"
        
        # 验证PE计算逻辑
        valid_pe = pe_ttm['pe_ttm'].dropna()
        assert all(val > 0 for val in valid_pe), "PE值应为正数"
        
        # 验证PE计算公式: PE = 市值 / 净利润
        latest_mv = market_data['total_mv'].iloc[-1]
        latest_profit = financial_data['net_profit'].iloc[0]  # 最新年度净利润
        expected_pe = latest_mv / latest_profit
        
        actual_pe = pe_ttm['pe_ttm'].iloc[-1]
        assert abs(actual_pe - expected_pe) / expected_pe < 0.01, "PE计算错误"  
  
    def test_calculate_pb(self, calculator, market_data, financial_data):
        """测试市净率计算"""
        pb = calculator.calculate_pb(market_data, financial_data)
        
        # 基本验证
        assert not pb.empty, "PB结果不应为空"
        assert 'pb' in pb.columns, "应包含pb列"
        
        # 验证PB计算逻辑
        valid_pb = pb['pb'].dropna()
        assert all(val > 0 for val in valid_pb), "PB值应为正数"
        
        # 验证PB计算公式: PB = 市值 / 净资产
        latest_mv = market_data['total_mv'].iloc[-1]
        latest_equity = financial_data['total_equity'].iloc[0]
        expected_pb = latest_mv / latest_equity
        
        actual_pb = pb['pb'].iloc[-1]
        assert abs(actual_pb - expected_pb) / expected_pb < 0.01, "PB计算错误"
    
    def test_calculate_ps_ttm(self, calculator, market_data, financial_data):
        """测试市销率TTM计算"""
        ps_ttm = calculator.calculate_ps_ttm(market_data, financial_data)
        
        # 基本验证
        assert not ps_ttm.empty, "PS TTM结果不应为空"
        assert 'ps_ttm' in ps_ttm.columns, "应包含ps_ttm列"
        
        # 验证PS计算逻辑
        valid_ps = ps_ttm['ps_ttm'].dropna()
        assert all(val > 0 for val in valid_ps), "PS值应为正数"
    
    def test_calculate_ev_ebitda(self, calculator, market_data, financial_data):
        """测试EV/EBITDA计算"""
        ev_ebitda = calculator.calculate_ev_ebitda(market_data, financial_data)
        
        # 基本验证
        assert not ev_ebitda.empty, "EV/EBITDA结果不应为空"
        assert 'ev_ebitda' in ev_ebitda.columns, "应包含ev_ebitda列"
        
        # 验证EV/EBITDA计算逻辑
        valid_ev_ebitda = ev_ebitda['ev_ebitda'].dropna()
        assert all(val > 0 for val in valid_ev_ebitda), "EV/EBITDA值应为正数"
    
    def test_edge_cases(self, calculator, market_data):
        """测试边界条件"""
        # 测试零净利润情况
        zero_profit_data = TestFundamentalDataGenerator.generate_financial_data()
        zero_profit_data['net_profit'] = 0
        
        pe_ttm = calculator.calculate_pe_ttm(market_data, zero_profit_data)
        assert pe_ttm['pe_ttm'].isna().all(), "零净利润时PE应为NaN"
        
        # 测试负净利润情况
        negative_profit_data = TestFundamentalDataGenerator.generate_financial_data()
        negative_profit_data['net_profit'] = -100000
        
        pe_ttm = calculator.calculate_pe_ttm(market_data, negative_profit_data)
        assert pe_ttm['pe_ttm'].isna().all(), "负净利润时PE应为NaN"

class TestProfitabilityFactorCalculator:
    """盈利能力因子计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        mock_engine = Mock()
        return ProfitabilityFactorCalculator(mock_engine)
    
    @pytest.fixture
    def financial_data(self):
        """财务数据"""
        return TestFundamentalDataGenerator.generate_financial_data()
    
    def test_calculate_roe(self, calculator, financial_data):
        """测试净资产收益率计算"""
        roe = calculator.calculate_roe(financial_data)
        
        # 基本验证
        assert not roe.empty, "ROE结果不应为空"
        assert 'roe' in roe.columns, "应包含roe列"
        
        # 验证ROE计算逻辑
        valid_roe = roe['roe'].dropna()
        assert all(-100 <= val <= 100 for val in valid_roe), "ROE值应在合理范围内"
        
        # 验证ROE计算公式: ROE = 净利润 / 净资产 * 100
        latest_profit = financial_data['net_profit'].iloc[0]
        latest_equity = financial_data['total_equity'].iloc[0]
        expected_roe = (latest_profit / latest_equity) * 100
        
        actual_roe = roe['roe'].iloc[0]
        assert abs(actual_roe - expected_roe) < 0.01, "ROE计算错误"
    
    def test_calculate_roa(self, calculator, financial_data):
        """测试总资产收益率计算"""
        roa = calculator.calculate_roa(financial_data)
        
        # 基本验证
        assert not roa.empty, "ROA结果不应为空"
        assert 'roa' in roa.columns, "应包含roa列"
        
        # 验证ROA计算逻辑
        valid_roa = roa['roa'].dropna()
        assert all(-100 <= val <= 100 for val in valid_roa), "ROA值应在合理范围内"
        
        # 验证ROA计算公式: ROA = 净利润 / 总资产 * 100
        latest_profit = financial_data['net_profit'].iloc[0]
        latest_assets = financial_data['total_assets'].iloc[0]
        expected_roa = (latest_profit / latest_assets) * 100
        
        actual_roa = roa['roa'].iloc[0]
        assert abs(actual_roa - expected_roa) < 0.01, "ROA计算错误"
    
    def test_calculate_gross_margin(self, calculator, financial_data):
        """测试毛利率计算"""
        gross_margin = calculator.calculate_gross_margin(financial_data)
        
        # 基本验证
        assert not gross_margin.empty, "毛利率结果不应为空"
        assert 'gross_margin' in gross_margin.columns, "应包含gross_margin列"
        
        # 验证毛利率计算逻辑
        valid_margin = gross_margin['gross_margin'].dropna()
        assert all(0 <= val <= 100 for val in valid_margin), "毛利率应在0-100%之间"
        
        # 验证毛利率计算公式: 毛利率 = 毛利润 / 营收 * 100
        latest_gross_profit = financial_data['gross_profit'].iloc[0]
        latest_revenue = financial_data['revenue'].iloc[0]
        expected_margin = (latest_gross_profit / latest_revenue) * 100
        
        actual_margin = gross_margin['gross_margin'].iloc[0]
        assert abs(actual_margin - expected_margin) < 0.01, "毛利率计算错误"
    
    def test_calculate_net_margin(self, calculator, financial_data):
        """测试净利率计算"""
        net_margin = calculator.calculate_net_margin(financial_data)
        
        # 基本验证
        assert not net_margin.empty, "净利率结果不应为空"
        assert 'net_margin' in net_margin.columns, "应包含net_margin列"
        
        # 验证净利率计算逻辑
        valid_margin = net_margin['net_margin'].dropna()
        assert all(-100 <= val <= 100 for val in valid_margin), "净利率应在合理范围内"

class TestFundamentalFactorEngine:
    """基本面因子引擎测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def fundamental_engine(self, mock_engine):
        """创建基本面因子引擎"""
        return FundamentalFactorEngine(mock_engine)
    
    def test_engine_initialization(self, fundamental_engine):
        """测试引擎初始化"""
        assert fundamental_engine is not None
        assert hasattr(fundamental_engine, 'valuation_calculator')
        assert hasattr(fundamental_engine, 'profitability_calculator')
    
    def test_calculate_all_factors(self, fundamental_engine):
        """测试计算所有基本面因子"""
        market_data = TestFundamentalDataGenerator.generate_market_data()
        financial_data = TestFundamentalDataGenerator.generate_financial_data()
        
        # 模拟数据库查询
        with patch.object(fundamental_engine, '_get_market_data', return_value=market_data), \
             patch.object(fundamental_engine, '_get_financial_data', return_value=financial_data):
            
            factors = fundamental_engine.calculate_all_factors(['000001.SZ'])
        
        # 验证返回的因子
        assert isinstance(factors, pd.DataFrame)
        assert not factors.empty
        assert 'ts_code' in factors.columns
        
        # 验证包含主要基本面指标
        expected_factors = ['pe_ttm', 'pb', 'roe', 'roa']
        for factor in expected_factors:
            if factor in factors.columns:
                # 至少应该有一些非NaN值
                assert not factors[factor].isna().all(), f"因子 {factor} 不应全为NaN"

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])