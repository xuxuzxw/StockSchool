#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试配置文件

提供共享的测试fixtures和配置
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import tempfile
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        'database': {
            'url': 'sqlite:///:memory:',
            'pool_size': 5
        },
        'tushare': {
            'token': 'test_token',
            'api_limit': 200
        },
        'akshare': {
            'api_limit': 100,
            'retry_times': 3
        }
    }


@pytest.fixture
def mock_db_engine():
    """模拟数据库引擎"""
    engine = Mock()
    connection = Mock()
    engine.connect.return_value.__enter__.return_value = connection
    engine.connect.return_value.__exit__.return_value = None
    return engine


@pytest.fixture
def sample_stock_data():
    """生成样本股票数据"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # 确保可重现性
    
    base_price = 100
    prices = base_price + np.random.randn(100).cumsum() * 2
    volumes = np.random.randint(1000, 10000, 100)
    
    return pd.DataFrame({
        'ts_code': ['000001.SZ'] * 100,
        'trade_date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes,
        'amount': volumes * prices
    })


@pytest.fixture
def sample_financial_data():
    """生成样本财务数据"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
        'end_date': ['2024-03-31', '2024-03-31', '2024-03-31'],
        'revenue': [1000000, 800000, 1200000],
        'net_profit': [100000, 80000, 120000],
        'total_assets': [5000000, 4000000, 6000000],
        'total_equity': [2000000, 1600000, 2400000],
        'pe_ratio': [15.5, 18.2, 12.8],
        'pb_ratio': [1.2, 1.5, 1.1]
    })


@pytest.fixture
def sample_trade_calendar():
    """生成样本交易日历"""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    # 排除周末
    trade_dates = [d for d in dates if d.weekday() < 5]
    
    return pd.DataFrame({
        'cal_date': trade_dates,
        'is_open': [1] * len(trade_dates),
        'pretrade_date': [None] * len(trade_dates)
    })


@pytest.fixture
def sample_industry_data():
    """生成样本行业数据"""
    return pd.DataFrame({
        'index_code': ['801010.SI', '801020.SI', '801030.SI'],
        'industry_name': ['石油石化', '煤炭', '有色金属'],
        'level': ['L1', 'L1', 'L1'],
        'parent_code': ['', '', ''],
        'src': ['SW2021', 'SW2021', 'SW2021']
    })


@pytest.fixture
def sample_sentiment_data():
    """生成样本情绪数据"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
        'news_date': ['2024-01-01', '2024-01-01', '2024-01-01'],
        'sentiment_score': [0.6, -0.3, 0.2],
        'positive_count': [10, 5, 8],
        'negative_count': [3, 12, 6],
        'neutral_count': [7, 8, 10],
        'news_volume': [20, 25, 24]
    })


@pytest.fixture
def temp_database():
    """创建临时数据库"""
    import sqlite3
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_file.close()
    
    # 创建数据库连接
    conn = sqlite3.connect(temp_file.name)
    
    # 创建基础表结构
    conn.execute('''
        CREATE TABLE IF NOT EXISTS stock_daily (
            ts_code TEXT,
            trade_date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ts_code, trade_date)
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sync_status (
            data_source TEXT,
            data_type TEXT,
            last_sync_date DATE,
            sync_status TEXT,
            PRIMARY KEY (data_source, data_type)
        )
    ''')
    
    conn.commit()
    
    yield temp_file.name
    
    # 清理
    conn.close()
    os.unlink(temp_file.name)


@pytest.fixture
def mock_tushare_api():
    """模拟Tushare API"""
    api = Mock()
    
    # 模拟常用API方法
    api.stock_basic.return_value = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'symbol': ['000001', '000002'],
        'name': ['平安银行', '万科A'],
        'area': ['深圳', '深圳'],
        'industry': ['银行', '房地产'],
        'list_date': ['19910403', '19910129']
    })
    
    api.daily.return_value = pd.DataFrame({
        'ts_code': ['000001.SZ'] * 5,
        'trade_date': ['20240101', '20240102', '20240103', '20240104', '20240105'],
        'open': [10.0, 10.1, 10.2, 10.3, 10.4],
        'high': [10.2, 10.3, 10.4, 10.5, 10.6],
        'low': [9.8, 9.9, 10.0, 10.1, 10.2],
        'close': [10.1, 10.2, 10.3, 10.4, 10.5],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    return api


@pytest.fixture
def mock_akshare_api():
    """模拟AkShare API"""
    api = Mock()
    
    # 模拟情绪数据API
    api.stock_news_em.return_value = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'news_date': ['2024-01-01', '2024-01-01'],
        'sentiment_score': [0.6, -0.3],
        'news_volume': [20, 25]
    })
    
    # 模拟热度排行API
    api.stock_hot_rank_em.return_value = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'ranking_date': ['2024-01-01', '2024-01-01'],
        'rank_position': [1, 2],
        'popularity_score': [95.8, 88.2]
    })
    
    return api


@pytest.fixture
def performance_monitor():
    """性能监控fixture"""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
        def stop(self):
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                'execution_time': end_time - self.start_time,
                'memory_usage': end_memory - self.start_memory,
                'peak_memory': end_memory
            }
    
    return PerformanceMonitor()


@pytest.fixture
def data_quality_samples():
    """数据质量测试样本"""
    return {
        'clean_data': pd.DataFrame({
            'ts_code': ['000001.SZ'] * 10,
            'close': np.random.randn(10) + 100,
            'volume': np.random.randint(1000, 5000, 10),
            'trade_date': pd.date_range('2024-01-01', periods=10)
        }),
        
        'dirty_data': pd.DataFrame({
            'ts_code': ['000001.SZ'] * 10,
            'close': [100, np.nan, 102, -50, 104, 1000, np.nan, 107, 108, 109],  # 包含异常值和缺失值
            'volume': [1000, 1100, -500, 1300, 1400, 1500, 1600, 1700, 1800, 1900],  # 包含负数
            'trade_date': pd.date_range('2024-01-01', periods=10)
        }),
        
        'incomplete_data': pd.DataFrame({
            'ts_code': ['000001.SZ'] * 5,
            'close': [100, 101, np.nan, np.nan, 104],  # 40%缺失
            'volume': [1000, np.nan, 1200, 1300, np.nan],  # 40%缺失
            'trade_date': pd.date_range('2024-01-01', periods=5)
        })
    }


# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "performance: 性能测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
    config.addinivalue_line("markers", "database: 需要数据库的测试")
    config.addinivalue_line("markers", "network: 需要网络连接的测试")


# 测试跳过条件
def pytest_collection_modifyitems(config, items):
    """根据条件跳过测试"""
    # 如果没有数据库连接，跳过数据库测试
    skip_database = pytest.mark.skip(reason="数据库连接不可用")
    
    # 如果没有网络连接，跳过网络测试
    skip_network = pytest.mark.skip(reason="网络连接不可用")
    
    for item in items:
        if "database" in item.keywords:
            # 这里可以添加数据库连接检查逻辑
            pass
            
        if "network" in item.keywords:
            # 这里可以添加网络连接检查逻辑
            pass