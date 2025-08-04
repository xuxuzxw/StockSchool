#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置文件
提供全局的测试fixture和配置
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock


@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        'default_seed': 42,
        'default_periods': 30,
        'performance_threshold': 1.0,  # 秒
        'memory_threshold': 50,  # MB
        'test_stock_code': '000001.SZ'
    }


@pytest.fixture
def mock_db_engine():
    """模拟数据库引擎"""
    engine = Mock()
    engine.connect.return_value.__enter__ = Mock(return_value=Mock())
    engine.connect.return_value.__exit__ = Mock(return_value=None)
    return engine


@pytest.fixture
def sample_stock_codes():
    """样本股票代码列表"""
    return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']


@pytest.fixture(autouse=True)
def setup_numpy_seed():
    """自动设置numpy随机种子，确保测试可重现"""
    np.random.seed(42)
    yield
    # 测试后清理（如果需要）


class TestDataValidator:
    """测试数据验证器"""
    
    @staticmethod
    def validate_factor_series(series: pd.Series, expected_length: int, 
                             value_range: tuple = None, name: str = "Factor"):
        """验证因子序列的基本属性"""
        assert isinstance(series, pd.Series), f"{name} 应该是 pd.Series 类型"
        assert len(series) == expected_length, f"{name} 长度应该是 {expected_length}"
        
        if value_range:
            valid_values = series.dropna()
            if not valid_values.empty:
                min_val, max_val = value_range
                assert all(min_val <= val <= max_val for val in valid_values), \
                    f"{name} 的值应该在 {value_range} 范围内"
    
    @staticmethod
    def validate_calculation_results(results: dict, expected_factors: list, 
                                   data_length: int):
        """验证计算结果的完整性"""
        assert isinstance(results, dict), "计算结果应该是字典类型"
        
        for factor in expected_factors:
            assert factor in results, f"应该包含因子 {factor}"
            TestDataValidator.validate_factor_series(
                results[factor], data_length, name=factor
            )


# 性能测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "performance: 标记性能测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记集成测试"
    )
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试"
    )