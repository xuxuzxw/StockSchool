import os
import sys
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from src.compute.fundamental_factor_engine import ProfitabilityFactorCalculator

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试基本面因子计算引擎
验证ROE、ROA等盈利能力因子的计算准确性
"""


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestProfitabilityFactorCalculator(unittest.TestCase):
    """测试盈利能力因子计算器"""

    def setUp(self):
        """测试初始化"""
        self.calculator = ProfitabilityFactorCalculator()

        # 创建测试数据
        self.test_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "net_profit": [1000000, 2000000, -500000, 0],
                "total_hldr_eqy_exc_min_int": [5000000, 10000000, 2000000, 0],
                "total_assets": [10000000, 20000000, 5000000, 0],
            }
        )

    def test_calculate_roe_normal(self):
        """测试正常情况下的ROE计算"""
        roe = self.calculator.calculate_roe(self.test_data)

        # 验证计算结果
        expected_roe = [20.0, 20.0, -25.0, np.nan]  # 百分比形式
        np.testing.assert_array_almost_equal(roe.values, expected_roe, decimal=2)

    def test_calculate_roa_normal(self):
        """测试正常情况下的ROA计算"""
        roa = self.calculator.calculate_roa(self.test_data)

        # 验证计算结果
        expected_roa = [10.0, 10.0, -10.0, np.nan]  # 百分比形式
        np.testing.assert_array_almost_equal(roa.values, expected_roa, decimal=2)

    def test_calculate_roe_missing_columns(self):
        """测试缺少必要列时的ROE计算"""
        incomplete_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "net_profit": [1000000],
                # 缺少total_hldr_eqy_exc_min_int列
            }
        )

        roe = self.calculator.calculate_roe(incomplete_data)
        self.assertTrue(roe.empty or roe.isna().all())

    def test_calculate_roa_missing_columns(self):
        """测试缺少必要列时的ROA计算"""
        incomplete_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "net_profit": [1000000],
                # 缺少total_assets列
            }
        )

        roa = self.calculator.calculate_roa(incomplete_data)
        self.assertTrue(roa.empty or roa.isna().all())

    def test_calculate_roe_zero_division(self):
        """测试除零情况下的ROE计算"""
        zero_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "net_profit": [1000000, 2000000],
                "total_hldr_eqy_exc_min_int": [0, 1000000],
            }
        )

        roe = self.calculator.calculate_roe(zero_data)
        expected_roe = [np.nan, 200.0]
        np.testing.assert_array_almost_equal(roe.values, expected_roe, decimal=2)

    def test_calculate_roa_zero_division(self):
        """测试除零情况下的ROA计算"""
        zero_data = pd.DataFrame(
            {"ts_code": ["000001.SZ", "000002.SZ"], "net_profit": [1000000, 2000000], "total_assets": [0, 1000000]}
        )

        roa = self.calculator.calculate_roa(zero_data)
        expected_roa = [np.nan, 200.0]
        np.testing.assert_array_almost_equal(roa.values, expected_roa, decimal=2)

    def test_calculate_roe_outlier_filtering(self):
        """测试ROE异常值过滤"""
        outlier_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "net_profit": [1000000, 1000000, 1000000],
                "total_hldr_eqy_exc_min_int": [1000000, 10000, 1000000],  # 第二个会产生10000%的异常值
            }
        )

        roe = self.calculator.calculate_roe(outlier_data)
        expected_roe = [100.0, 10000.0, 100.0]
        np.testing.assert_array_almost_equal(roe.values, expected_roe, decimal=2)

    def test_calculate_roa_outlier_filtering(self):
        """测试ROA异常值过滤"""
        outlier_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "net_profit": [1000000, 1000000, 1000000],
                "total_assets": [1000000, 10000, 1000000],  # 第二个会产生10000%的异常值
            }
        )

        roa = self.calculator.calculate_roa(outlier_data)
        expected_roa = [100.0, 10000.0, 100.0]
        np.testing.assert_array_almost_equal(roa.values, expected_roa, decimal=2)


if __name__ == "__main__":
    unittest.main()
