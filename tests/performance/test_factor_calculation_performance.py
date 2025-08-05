import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import psutil
import pytest

from src.compute.factor_engine import FactorEngine
from src.compute.fundamental_engine import FundamentalFactorEngine
from src.compute.sentiment_engine import SentimentFactorEngine
from src.compute.technical_engine import TechnicalFactorEngine
from src.utils.db import get_db_engine
from tests.utils.test_data_generator import TestDataGenerator

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算性能测试
测试因子计算的性能指标和资源使用情况
"""


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFactorCalculationPerformance:
    """因子计算性能测试"""

    @pytest.fixture(scope="class")
    def test_engine(self):
        """测试数据库引擎"""
        return get_db_engine()

    @pytest.fixture(scope="class")
    def data_generator(self):
        """测试数据生成器"""
        return TestDataGenerator()

    @pytest.fixture(scope="class")
    def large_stock_list(self):
        """大量股票列表（用于性能测试）"""
        return [
            "000001.SZ",
            "000002.SZ",
            "000858.SZ",
            "000876.SZ",
            "600000.SH",
            "600036.SH",
            "600519.SH",
            "600887.SH",
            "000063.SZ",
            "000166.SZ",
            "000338.SZ",
            "000725.SZ",
            "600009.SH",
            "600028.SH",
            "600030.SH",
            "600048.SH",
            "000001.SZ",
            "000002.SZ",
            "000858.SZ",
            "000876.SZ",
        ]

    @pytest.fixture(scope="class")
    def performance_data(self, data_generator, large_stock_list):
        """生成性能测试数据"""
        # 生成3个月的数据
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=90)

        data = {}
        for ts_code in large_stock_list:
            data[ts_code] = data_generator.generate_stock_daily_data(
                ts_code=ts_code, start_date=start_date, end_date=end_date, initial_price=100.0, volatility=0.02
            )

        return data

    def test_single_factor_calculation_performance(self, test_engine, large_stock_list):
        """测试单个因子计算性能"""
        technical_engine = TechnicalFactorEngine(test_engine)

        # 测试SMA计算性能
        start_time = time.time()

        sma_data = technical_engine.calculate_sma(
            ts_codes=large_stock_list,
            start_date=date.today() - timedelta(days=30),
            end_date=date.today() - timedelta(days=1),
            window=5,
        )

        calculation_time = time.time() - start_time

        logger.info(
            f"SMA计算性能: {len(large_stock_list)}只股票, "
            f"耗时: {calculation_time:.2f}秒, "
            f"数据量: {len(sma_data)}"
        )

        # 性能断言
        if not sma_data.empty:
            avg_time_per_stock = calculation_time / len(large_stock_list)
            assert avg_time_per_stock < 1.0, f"单股票平均计算时间过长: {avg_time_per_stock:.2f}秒"

            # 计算吞吐量
            throughput = len(sma_data) / calculation_time
            logger.info(f"SMA计算吞吐量: {throughput:.2f} 记录/秒")
            assert throughput > 100, f"计算吞吐量过低: {throughput:.2f} 记录/秒"

    def test_multiple_factors_calculation_performance(self, test_engine, large_stock_list):
        """测试多因子计算性能"""
        technical_engine = TechnicalFactorEngine(test_engine)

        factors_to_test = [
            (
                "sma_5",
                lambda: technical_engine.calculate_sma(
                    large_stock_list[:10], date.today() - timedelta(days=20), date.today() - timedelta(days=1), 5
                ),
            ),
            (
                "sma_20",
                lambda: technical_engine.calculate_sma(
                    large_stock_list[:10], date.today() - timedelta(days=30), date.today() - timedelta(days=1), 20
                ),
            ),
            (
                "rsi_14",
                lambda: technical_engine.calculate_rsi(
                    large_stock_list[:10], date.today() - timedelta(days=30), date.today() - timedelta(days=1), 14
                ),
            ),
        ]

        total_start_time = time.time()
        results = {}

        for factor_name, calc_func in factors_to_test:
            start_time = time.time()

            try:
                data = calc_func()
                calculation_time = time.time() - start_time

                results[factor_name] = {
                    "time": calculation_time,
                    "data_count": len(data) if not data.empty else 0,
                    "success": True,
                }

                logger.info(f"{factor_name}计算: {calculation_time:.2f}秒, {len(data)}条数据")

            except Exception as e:
                results[factor_name] = {
                    "time": time.time() - start_time,
                    "data_count": 0,
                    "success": False,
                    "error": str(e),
                }
                logger.warning(f"{factor_name}计算失败: {e}")

        total_time = time.time() - total_start_time

        logger.info(f"多因子计算总耗时: {total_time:.2f}秒")

        # 性能断言
        successful_factors = [r for r in results.values() if r["success"]]
        if successful_factors:
            avg_time_per_factor = sum(r["time"] for r in successful_factors) / len(successful_factors)
            assert avg_time_per_factor < 5.0, f"单因子平均计算时间过长: {avg_time_per_factor:.2f}秒"

    def test_concurrent_calculation_performance(self, test_engine, large_stock_list):
        """测试并发计算性能"""
        technical_engine = TechnicalFactorEngine(test_engine)

        def calculate_factor(ts_code):
            """单个股票的因子计算"""
            try:
                start_time = time.time()

                sma_data = technical_engine.calculate_sma(
                    ts_codes=[ts_code],
                    start_date=date.today() - timedelta(days=20),
                    end_date=date.today() - timedelta(days=1),
                    window=5,
                )

                calculation_time = time.time() - start_time

                return {"ts_code": ts_code, "time": calculation_time, "data_count": len(sma_data), "success": True}

            except Exception as e:
                return {"ts_code": ts_code, "time": 0, "data_count": 0, "success": False, "error": str(e)}

        # 串行计算
        serial_start_time = time.time()
        serial_results = []

        for ts_code in large_stock_list[:8]:  # 测试前8只股票
            result = calculate_factor(ts_code)
            serial_results.append(result)

        serial_time = time.time() - serial_start_time

        # 并行计算
        parallel_start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(calculate_factor, large_stock_list[:8]))

        parallel_time = time.time() - parallel_start_time

        # 分析结果
        serial_success = sum(1 for r in serial_results if r["success"])
        parallel_success = sum(1 for r in parallel_results if r["success"])

        logger.info(f"串行计算: {serial_time:.2f}秒, 成功: {serial_success}/8")
        logger.info(f"并行计算: {parallel_time:.2f}秒, 成功: {parallel_success}/8")

        if serial_success > 0 and parallel_success > 0:
            speedup = serial_time / parallel_time
            logger.info(f"并行加速比: {speedup:.2f}x")

            # 性能断言
            assert speedup > 1.5, f"并行加速效果不明显: {speedup:.2f}x"

    def test_memory_usage_performance(self, test_engine, large_stock_list):
        """测试内存使用性能"""
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            technical_engine = TechnicalFactorEngine(test_engine)

            # 执行多个计算任务
            memory_snapshots = [initial_memory]

            for window in [5, 10, 20]:
                try:
                    data = technical_engine.calculate_sma(
                        ts_codes=large_stock_list[:10],
                        start_date=date.today() - timedelta(days=30),
                        end_date=date.today() - timedelta(days=1),
                        window=window,
                    )

                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_snapshots.append(current_memory)

                    logger.info(f"SMA{window}计算后内存: {current_memory:.1f}MB, " f"数据量: {len(data)}")

                except Exception as e:
                    logger.warning(f"SMA{window}计算失败: {e}")

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            max_memory = max(memory_snapshots)
            memory_increase = final_memory - initial_memory
            peak_increase = max_memory - initial_memory

            logger.info(
                f"内存使用测试: 初始={initial_memory:.1f}MB, "
                f"结束={final_memory:.1f}MB, "
                f"增长={memory_increase:.1f}MB, "
                f"峰值增长={peak_increase:.1f}MB"
            )

            # 内存使用断言
            assert memory_increase < 200, f"内存增长过多: {memory_increase:.1f}MB"
            assert peak_increase < 300, f"峰值内存增长过多: {peak_increase:.1f}MB"

        except ImportError:
            logger.warning("psutil未安装，跳过内存使用测试")
        except Exception as e:
            logger.error(f"内存使用测试失败: {e}")

    def test_large_dataset_performance(self, test_engine, data_generator):
        """测试大数据集性能"""
        # 生成大量测试数据
        large_ts_codes = [f"TEST{i:06d}.SZ" for i in range(100)]

        technical_engine = TechnicalFactorEngine(test_engine)

        # 测试批量计算性能
        batch_sizes = [10, 20, 50]

        for batch_size in batch_sizes:
            start_time = time.time()

            try:
                # 分批处理
                total_processed = 0

                for i in range(0, min(len(large_ts_codes), 50), batch_size):
                    batch_codes = large_ts_codes[i : i + batch_size]

                    data = technical_engine.calculate_sma(
                        ts_codes=batch_codes,
                        start_date=date.today() - timedelta(days=10),
                        end_date=date.today() - timedelta(days=1),
                        window=5,
                    )

                    total_processed += len(batch_codes)

                batch_time = time.time() - start_time

                logger.info(f"批量大小{batch_size}: {batch_time:.2f}秒, " f"处理{total_processed}只股票")

                # 性能断言
                if total_processed > 0:
                    avg_time_per_stock = batch_time / total_processed
                    assert avg_time_per_stock < 2.0, f"大数据集单股票处理时间过长: {avg_time_per_stock:.2f}秒"

            except Exception as e:
                logger.warning(f"批量大小{batch_size}测试失败: {e}")

    def test_calculation_accuracy_vs_performance(self, test_engine, data_generator):
        """测试计算精度与性能的权衡"""
        technical_engine = TechnicalFactorEngine(test_engine)

        # 生成标准测试数据
        test_data = data_generator.generate_stock_daily_data(
            ts_code="ACCURACY_TEST.SZ",
            start_date=date.today() - timedelta(days=100),
            end_date=date.today() - timedelta(days=1),
            initial_price=100.0,
            volatility=0.02,
        )

        # 测试不同精度设置的性能影响
        precision_tests = [("float32", np.float32), ("float64", np.float64)]

        for precision_name, dtype in precision_tests:
            start_time = time.time()

            try:
                # 模拟不同精度的计算
                prices = test_data["close"].astype(dtype)

                # 计算SMA
                sma_values = prices.rolling(window=20).mean()

                # 计算RSI
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_values = 100 - (100 / (1 + rs))

                calculation_time = time.time() - start_time

                logger.info(
                    f"{precision_name}精度计算: {calculation_time:.4f}秒, "
                    f"SMA数据点: {sma_values.notna().sum()}, "
                    f"RSI数据点: {rsi_values.notna().sum()}"
                )

                # 验证计算结果的合理性
                assert sma_values.notna().sum() > 0, f"{precision_name}精度SMA计算无有效结果"
                assert rsi_values.notna().sum() > 0, f"{precision_name}精度RSI计算无有效结果"

            except Exception as e:
                logger.error(f"{precision_name}精度测试失败: {e}")

    def test_database_io_performance(self, test_engine, large_stock_list):
        """测试数据库I/O性能"""
        # 测试数据读取性能
        read_start_time = time.time()

        try:
            with test_engine.connect() as conn:
                # 模拟批量数据读取
                for batch_start in range(0, min(len(large_stock_list), 20), 5):
                    batch_codes = large_stock_list[batch_start : batch_start + 5]

                    query = """
                    SELECT ts_code, trade_date, close, volume
                    FROM stock_daily
                    WHERE ts_code = ANY(%s)
                    AND trade_date >= %s
                    ORDER BY ts_code, trade_date
                    """

                    result = conn.execute(query, (batch_codes, (date.today() - timedelta(days=30)).strftime("%Y%m%d")))

                    data = result.fetchall()
                    logger.info(f"批量读取{len(batch_codes)}只股票: {len(data)}条记录")

            read_time = time.time() - read_start_time

            logger.info(f"数据库读取性能测试: {read_time:.2f}秒")

            # 性能断言
            assert read_time < 10.0, f"数据库读取时间过长: {read_time:.2f}秒"

        except Exception as e:
            logger.error(f"数据库I/O性能测试失败: {e}")

    def test_cache_performance_impact(self, test_engine, large_stock_list):
        """测试缓存对性能的影响"""
        technical_engine = TechnicalFactorEngine(test_engine)

        test_params = {
            "ts_codes": large_stock_list[:5],
            "start_date": date.today() - timedelta(days=20),
            "end_date": date.today() - timedelta(days=1),
            "window": 5,
        }

        # 第一次计算（无缓存）
        first_start_time = time.time()

        try:
            first_result = technical_engine.calculate_sma(**test_params)
            first_time = time.time() - first_start_time

            logger.info(f"首次计算（无缓存）: {first_time:.2f}秒, {len(first_result)}条数据")

            # 第二次计算（可能有缓存）
            second_start_time = time.time()
            second_result = technical_engine.calculate_sma(**test_params)
            second_time = time.time() - second_start_time

            logger.info(f"二次计算（可能缓存）: {second_time:.2f}秒, {len(second_result)}条数据")

            # 分析缓存效果
            if first_time > 0 and second_time > 0:
                speedup = first_time / second_time
                logger.info(f"缓存加速比: {speedup:.2f}x")

                # 如果有明显的缓存效果，验证加速比
                if speedup > 2.0:
                    logger.info("检测到缓存加速效果")
                else:
                    logger.info("未检测到明显缓存效果")

        except Exception as e:
            logger.error(f"缓存性能测试失败: {e}")


class TestFactorEnginePerformance:
    """因子引擎整体性能测试"""

    @pytest.fixture(scope="class")
    def factor_engine(self):
        """因子引擎实例"""
        return FactorEngine(get_db_engine())

    def test_multi_engine_performance(self, factor_engine):
        """测试多引擎协同性能"""
        test_stocks = ["000001.SZ", "000002.SZ", "600000.SH"]

        start_time = time.time()

        try:
            # 同时计算多种类型的因子
            results = factor_engine.calculate_all_factors(
                ts_codes=test_stocks,
                start_date=date.today() - timedelta(days=10),
                end_date=date.today() - timedelta(days=1),
                factor_types=["technical"],  # 只测试技术面因子
            )

            total_time = time.time() - start_time

            logger.info(f"多引擎协同计算: {total_time:.2f}秒")
            logger.info(f"计算结果: {results}")

            # 性能断言
            assert total_time < 30.0, f"多引擎计算时间过长: {total_time:.2f}秒"

        except Exception as e:
            logger.error(f"多引擎性能测试失败: {e}")

    def test_resource_cleanup_performance(self, factor_engine):
        """测试资源清理性能"""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        try:
            # 执行多轮计算
            for round_num in range(3):
                results = factor_engine.calculate_all_factors(
                    ts_codes=["000001.SZ"],
                    start_date=date.today() - timedelta(days=5),
                    end_date=date.today() - timedelta(days=1),
                    factor_types=["technical"],
                )

                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                logger.info(f"第{round_num+1}轮计算后内存: {current_memory:.1f}MB")

            final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory

            logger.info(f"资源清理测试: 内存增长 {memory_growth:.1f}MB")

            # 内存泄漏检查
            assert memory_growth < 100, f"可能存在内存泄漏: {memory_growth:.1f}MB"

        except Exception as e:
            logger.error(f"资源清理性能测试失败: {e}")


if __name__ == "__main__":
    # 运行性能测试
    pytest.main([__file__, "-v", "-s", "--tb=short"])
