import logging
import os
import sys
from datetime import date, timedelta
from typing import Any, Dict, List

import pytest

from src.compute.fundamental_engine import FundamentalFactorEngine
from src.compute.sentiment_engine import SentimentFactorEngine
from src.compute.technical_engine import TechnicalFactorEngine
from src.utils.db import get_db_engine
from tests.utils.test_data_generator import TestDataGenerator

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的因子计算性能测试
使用改进的架构和设计模式
"""


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from .performance_config import performance_config

# 导入重构的组件
from .performance_test_base import BasePerformanceTest, PerformanceMetrics, performance_test
from .test_strategies import (
    ConcurrentTestStrategy,
    MultipleFactorTestStrategy,
    PerformanceTestContext,
    SingleFactorTestStrategy,
)

logger = logging.getLogger(__name__)


class TechnicalFactorPerformanceTest(BasePerformanceTest):
    """技术面因子性能测试"""

    def __init__(self, engine, data_generator):
        """方法描述"""
        self.technical_engine = TechnicalFactorEngine(engine)

    def prepare_test_data(self, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """准备技术面因子测试数据"""
        stock_count = test_params.get("stock_count", 10)
        test_stocks = performance_config.get_test_stocks(stock_count)

        return {
            "ts_codes": test_stocks,
            "start_date": date.today() - timedelta(days=30),
            "end_date": date.today() - timedelta(days=1),
            "engine": self.technical_engine,
        }

    def execute_test(self, test_data: Dict[str, Any], test_params: Dict[str, Any]) -> Any:
        """执行技术面因子测试"""
        strategy_name = test_params.get("strategy", "single_factor")

        # 使用策略模式
        if strategy_name == "single_factor":
            strategy = SingleFactorTestStrategy()
        elif strategy_name == "multi_factor":
            strategy = MultipleFactorTestStrategy()
        elif strategy_name == "concurrent":
            strategy = ConcurrentTestStrategy()
        else:
            raise ValueError(f"不支持的测试策略: {strategy_name}")

        context = PerformanceTestContext(strategy)
        return context.execute_performance_test(
            test_data["engine"],
            {
                "ts_codes": test_data["ts_codes"],
                "factor_name": test_params.get("factor_name", "sma_5"),
                "window": test_params.get("window", 5),
                "factor_configs": performance_config.get_factor_configs("technical"),
                "max_workers": test_params.get("max_workers", 4),
            },
        )


class TestSingleFactorPerformance:
    """单因子性能测试类"""

    @pytest.fixture(scope="class")
    def performance_tester(self):
        """性能测试器fixture"""
        engine = get_db_engine()
        data_generator = TestDataGenerator()
        return TechnicalFactorPerformanceTest(engine, data_generator)

    @performance_test(test_type="single_factor")
    def test_sma_calculation_performance(self, performance_tester):
        """测试SMA计算性能"""
        metrics = performance_tester.run_performance_test(
            test_name="SMA计算性能测试",
            test_params={
                "strategy": "single_factor",
                "factor_name": "sma_5",
                "window": 5,
                "stock_count": 10,
                "test_type": "single_factor",
            },
        )

        # 验证结果
        assert metrics.success, f"SMA计算失败: {metrics.error_message}"
        assert metrics.data_count > 0, "SMA计算应该产生数据"

        logger.info(f"SMA性能指标: 执行时间={metrics.execution_time:.2f}s, " f"吞吐量={metrics.throughput:.2f}记录/秒")

    @performance_test(test_type="single_factor")
    def test_rsi_calculation_performance(self, performance_tester):
        """测试RSI计算性能"""
        metrics = performance_tester.run_performance_test(
            test_name="RSI计算性能测试",
            test_params={
                "strategy": "single_factor",
                "factor_name": "rsi_14",
                "window": 14,
                "stock_count": 10,
                "test_type": "single_factor",
            },
        )

        assert metrics.success, f"RSI计算失败: {metrics.error_message}"
        assert metrics.data_count > 0, "RSI计算应该产生数据"


class TestMultipleFactorPerformance:
    """多因子性能测试类"""

    @pytest.fixture(scope="class")
    def performance_tester(self):
        """性能测试器fixture"""
        engine = get_db_engine()
        data_generator = TestDataGenerator()
        return TechnicalFactorPerformanceTest(engine, data_generator)

    @performance_test(test_type="multi_factor")
    def test_multiple_technical_factors_performance(self, performance_tester):
        """测试多个技术面因子计算性能"""
        metrics = performance_tester.run_performance_test(
            test_name="多技术面因子性能测试",
            test_params={"strategy": "multi_factor", "stock_count": 5, "test_type": "multi_factor"},
        )

        assert metrics.success, f"多因子计算失败: {metrics.error_message}"
        assert metrics.data_count > 0, "多因子计算应该产生数据"

        logger.info(f"多因子性能指标: 执行时间={metrics.execution_time:.2f}s, " f"总数据量={metrics.data_count}")


class TestConcurrentPerformance:
    """并发性能测试类"""

    @pytest.fixture(scope="class")
    def performance_tester(self):
        """性能测试器fixture"""
        engine = get_db_engine()
        data_generator = TestDataGenerator()
        return TechnicalFactorPerformanceTest(engine, data_generator)

    @performance_test(test_type="concurrent")
    def test_concurrent_calculation_performance(self, performance_tester):
        """测试并发计算性能"""
        # 串行测试
        serial_metrics = performance_tester.run_performance_test(
            test_name="串行计算性能测试",
            test_params={
                "strategy": "single_factor",
                "factor_name": "sma_5",
                "stock_count": 8,
                "test_type": "single_factor",
            },
        )

        # 并发测试
        concurrent_metrics = performance_tester.run_performance_test(
            test_name="并发计算性能测试",
            test_params={
                "strategy": "concurrent",
                "factor_name": "sma_5",
                "stock_count": 8,
                "max_workers": 4,
                "test_type": "concurrent",
            },
        )

        # 比较性能
        if serial_metrics.success and concurrent_metrics.success:
            speedup = serial_metrics.execution_time / concurrent_metrics.execution_time
            logger.info(f"并发加速比: {speedup:.2f}x")

            # 验证并发效果
            assert speedup > 1.2, f"并发加速效果不明显: {speedup:.2f}x"


class TestMemoryPerformance:
    """内存性能测试类"""

    @pytest.fixture(scope="class")
    def performance_tester(self):
        """性能测试器fixture"""
        engine = get_db_engine()
        data_generator = TestDataGenerator()
        return TechnicalFactorPerformanceTest(engine, data_generator)

    def test_memory_usage_stability(self, performance_tester):
        """测试内存使用稳定性"""
        initial_metrics = None
        memory_growth_list = []

        # 执行多轮测试
        for round_num in range(5):
            metrics = performance_tester.run_performance_test(
                test_name=f"内存稳定性测试-第{round_num+1}轮",
                test_params={
                    "strategy": "single_factor",
                    "factor_name": "sma_5",
                    "stock_count": 5,
                    "test_type": "single_factor",
                },
            )

            if initial_metrics is None:
                initial_metrics = metrics
            else:
                memory_growth = metrics.memory_usage_mb - initial_metrics.memory_usage_mb
                memory_growth_list.append(memory_growth)

                logger.info(f"第{round_num+1}轮内存增长: {memory_growth:.1f}MB")

        # 验证内存稳定性
        if memory_growth_list:
            avg_growth = sum(memory_growth_list) / len(memory_growth_list)
            max_growth = max(memory_growth_list)

            logger.info(f"平均内存增长: {avg_growth:.1f}MB, 最大增长: {max_growth:.1f}MB")

            # 内存泄漏检查
            assert max_growth < 50, f"可能存在内存泄漏: 最大增长{max_growth:.1f}MB"
            assert avg_growth < 20, f"平均内存增长过大: {avg_growth:.1f}MB"


class TestLargeDatasetPerformance:
    """大数据集性能测试类"""

    @pytest.fixture(scope="class")
    def performance_tester(self):
        """性能测试器fixture"""
        engine = get_db_engine()
        data_generator = TestDataGenerator()
        return TechnicalFactorPerformanceTest(engine, data_generator)

    @performance_test(test_type="large_dataset")
    def test_large_stock_list_performance(self, performance_tester):
        """测试大股票列表性能"""
        large_stock_count = 50

        metrics = performance_tester.run_performance_test(
            test_name="大股票列表性能测试",
            test_params={
                "strategy": "single_factor",
                "factor_name": "sma_5",
                "stock_count": large_stock_count,
                "test_type": "large_dataset",
            },
        )

        assert metrics.success, f"大数据集计算失败: {metrics.error_message}"

        # 验证扩展性
        avg_time_per_stock = metrics.execution_time / large_stock_count
        logger.info(
            f"大数据集性能: 单股票平均时间={avg_time_per_stock:.3f}s, " f"总吞吐量={metrics.throughput:.2f}记录/秒"
        )

        assert avg_time_per_stock < 1.0, f"单股票处理时间过长: {avg_time_per_stock:.3f}s"


if __name__ == "__main__":
    # 运行重构后的性能测试
    pytest.main([__file__, "-v", "-s", "--tb=short"])
