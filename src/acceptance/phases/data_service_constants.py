"""
数据服务验收测试常量配置
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DataServiceConstants:
    """数据服务测试常量"""
    
    # 测试股票
    TEST_STOCK_CODE = '000001.SZ'
    
    # 数据质量阈值
    INVALID_CODE_THRESHOLD = 0.01  # 1%
    TRADING_RATIO_MIN = 0.5        # 50%
    TRADING_RATIO_MAX = 0.8        # 80%
    MIN_QUALITY_SCORE = 60.0       # 最低质量分数
    
    # 必需的数据库表
    REQUIRED_TABLES = [
        'stock_basic',
        'stock_daily', 
        'trade_calendar',
        'financial_data',
        'sync_status'
    ]
    
    # 股票基础信息必需列
    STOCK_BASIC_REQUIRED_COLUMNS = [
        'ts_code', 'symbol', 'name', 'market', 'list_status'
    ]
    
    # 日线数据必需列
    DAILY_DATA_REQUIRED_COLUMNS = [
        'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol'
    ]
    
    # 交易日历必需列
    TRADE_CAL_REQUIRED_COLUMNS = ['cal_date', 'is_open']
    
    # 财务报表类型
    FINANCIAL_REPORT_TYPES = ['income', 'balancesheet', 'cashflow']
    
    # 股票代码正则模式
    STOCK_CODE_PATTERN = r'^\d{6}\.(SZ|SH)$'


@dataclass 
class TestConfig:
    """单个测试配置"""
    name: str
    method_name: str
    description: str
    timeout: int = 60
    retry_count: int = 1


# 测试配置映射
TEST_CONFIGS = {
    'tushare_connection': TestConfig(
        name='tushare_connection_test',
        method_name='_test_tushare_connection',
        description='测试Tushare API连接',
        timeout=30
    ),
    'data_sources_health': TestConfig(
        name='data_sources_health_check',
        method_name='_test_data_sources_health', 
        description='检查数据源健康状态',
        timeout=60
    ),
    'stock_basic_sync': TestConfig(
        name='stock_basic_sync_test',
        method_name='_test_stock_basic_sync',
        description='测试股票基础信息同步',
        timeout=120
    ),
    'daily_data_sync': TestConfig(
        name='daily_data_sync_test', 
        method_name='_test_daily_data_sync',
        description='测试日线数据同步',
        timeout=120
    ),
    'trade_calendar_sync': TestConfig(
        name='trade_calendar_sync_test',
        method_name='_test_trade_calendar_sync', 
        description='测试交易日历同步',
        timeout=60
    ),
    'financial_data_sync': TestConfig(
        name='financial_data_sync_test',
        method_name='_test_financial_data_sync',
        description='测试财务数据同步', 
        timeout=180
    ),
    'data_quality_check': TestConfig(
        name='data_quality_check_test',
        method_name='_test_data_quality_check',
        description='测试数据质量检查',
        timeout=120
    ),
    'database_structure_validation': TestConfig(
        name='database_structure_validation_test',
        method_name='_test_database_structure_validation',
        description='测试数据库结构验证',
        timeout=60
    )
}