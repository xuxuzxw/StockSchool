import logging
import multiprocessing
import os
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import psutil
import pytest

from src.compute.factor_engine import FactorEngine
from src.compute.technical_engine import TechnicalFactorEngine
from src.utils.db import get_db_engine
from tests.utils.test_data_generator import TestDataGenerator

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算系统压力测试
测试系统在高负载下的稳定性和性能表现
"""


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSystemStress:
    """系统压力测试"""

    @pytest.fixture(scope="class")
    def test_engine(self):
        """测试数据库引擎"""
        return get_db_engine()

    @pytest.fixture(scope="class")
    def stress_stock_list(self):
        """压力测试股票列表"""
        # 生成更多的测试股票代码
        stock_codes = []

        # A股主板
        for i in range(1, 101):
            stock_codes.append(f"{i:06d}.SZ")
            stock_codes.append(f"{i+600000:06d}.SH")

        return stock_codes[:200]  # 限制为200只股票

    def test_high_concurrency_calculation(self, test_engine, stress_stock_list):
        """高并发计算压力测试"""
        technical_engine = TechnicalFactorEngine(test_engine)

        def concurrent_calculation(thread_id, stock_batch):
            """并发计算任务"""
            results = []
            start_time = time.time()

            try:
                for ts_code in stock_batch:
                    calc_start = time.time()

                    # 随机选择计算类型
                    calc_type = random.choice(["sma", "rsi"])

                    if calc_type == "sma":
                        data = technical_engine.calculate_sma(
                            ts_codes=[ts_code],
                            start_date=date.today() - timedelta(days=20),
                            end_date=date.today() - timedelta(days=1),
                            window=random.choice([5, 10, 20]),
                        )
                    else:
                        data = technical_engine.calculate_rsi(
                            ts_codes=[ts_code],
                            start_date=date.today() - timedelta(days=30),
                            end_date=date.today() - timedelta(days=1),
                            window=14,
                        )

                    calc_time = time.time() - calc_start

                    results.append(
                        {
                            "thread_id": thread_id,
                            "ts_code": ts_code,
                            "calc_type": calc_type,
                            "calc_time": calc_time,
                            "data_count": len(data) if not data.empty else 0,
                            "success": True,
                        }
                    )

            except Exception as e:
                results.append(
                    {
                        "thread_id": thread_id,
                        "ts_code": ts_code if "ts_code" in locals() else "unknown",
                        "calc_type": calc_type if "calc_type" in locals() else "unknown",
                        "calc_time": 0,
                        "data_count": 0,
                        "success": False,
                        "error": str(e),
                    }
                )

            total_time = time.time() - start_time
            logger.info(f"线程{thread_id}完成: {total_time:.2f}秒, {len(results)}个任务")

            return results

        # 分配任务到多个线程
        num_threads = 8
        batch_size = len(stress_stock_list[:40]) // num_threads  # 只测试前40只股票

        thread_batches = []
        for i in range(num_threads):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, 40)
            thread_batches.append(stress_stock_list[start_idx:end_idx])

        # 执行并发测试
        stress_start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(concurrent_calculation, i, batch) for i, batch in enumerate(thread_batches)]

            all_results = []
            for future in futures:
                try:
                    results = future.result(timeout=60)  # 60秒超时
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"线程执行失败: {e}")

        total_stress_time = time.time() - stress_start_time

        # 分析结果
        successful_tasks = [r for r in all_results if r["success"]]
        failed_tasks = [r for r in all_results if not r["success"]]

        success_rate = len(successful_tasks) / len(all_results) if all_results else 0
        avg_calc_time = np.mean([r["calc_time"] for r in successful_tasks]) if successful_tasks else 0

        logger.info(f"高并发压力测试结果:")
        logger.info(f"  总耗时: {total_stress_time:.2f}秒")
        logger.info(f"  总任务数: {len(all_results)}")
        logger.info(f"  成功率: {success_rate:.2%}")
        logger.info(f"  平均计算时间: {avg_calc_time:.3f}秒")
        logger.info(f"  失败任务数: {len(failed_tasks)}")

        # 压力测试断言
        assert success_rate >= 0.8, f"高并发成功率过低: {success_rate:.2%}"
        assert avg_calc_time < 2.0, f"高并发平均计算时间过长: {avg_calc_time:.3f}秒"

    def test_memory_pressure(self, test_engine, stress_stock_list):
        """内存压力测试"""
        technical_engine = TechnicalFactorEngine(test_engine)

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_snapshots = [initial_memory]

        logger.info(f"内存压力测试开始，初始内存: {initial_memory:.1f}MB")

        try:
            # 执行大量计算任务，不立即释放结果
            calculation_results = []

            for batch_num in range(5):  # 5批次
                batch_start_time = time.time()
                batch_results = []

                # 每批次计算多个因子
                for window in [5, 10, 20]:
                    try:
                        data = technical_engine.calculate_sma(
                            ts_codes=stress_stock_list[:20],  # 每批次20只股票
                            start_date=date.today() - timedelta(days=30),
                            end_date=date.today() - timedelta(days=1),
                            window=window,
                        )

                        batch_results.append({"batch": batch_num, "window": window, "data": data, "size": len(data)})

                    except Exception as e:
                        logger.warning(f"批次{batch_num}窗口{window}计算失败: {e}")

                calculation_results.extend(batch_results)

                # 记录内存使用
                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                memory_snapshots.append(current_memory)

                batch_time = time.time() - batch_start_time
                logger.info(
                    f"批次{batch_num}完成: {batch_time:.2f}秒, "
                    f"内存: {current_memory:.1f}MB, "
                    f"结果数: {len(batch_results)}"
                )

            # 分析内存使用
            final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            max_memory = max(memory_snapshots)
            total_memory_increase = final_memory - initial_memory
            peak_memory_increase = max_memory - initial_memory

            logger.info(f"内存压力测试结果:")
            logger.info(f"  初始内存: {initial_memory:.1f}MB")
            logger.info(f"  最终内存: {final_memory:.1f}MB")
            logger.info(f"  峰值内存: {max_memory:.1f}MB")
            logger.info(f"  总增长: {total_memory_increase:.1f}MB")
            logger.info(f"  峰值增长: {peak_memory_increase:.1f}MB")
            logger.info(f"  计算结果数: {len(calculation_results)}")

            # 内存压力断言
            assert peak_memory_increase < 500, f"峰值内存增长过多: {peak_memory_increase:.1f}MB"
            assert total_memory_increase < 300, f"总内存增长过多: {total_memory_increase:.1f}MB"

        except Exception as e:
            logger.error(f"内存压力测试失败: {e}")
        finally:
            # 清理大型对象
            if "calculation_results" in locals():
                del calculation_results

    def test_sustained_load(self, test_engine, stress_stock_list):
        """持续负载测试"""
        technical_engine = TechnicalFactorEngine(test_engine)

        test_duration = 60  # 测试持续60秒
        start_time = time.time()

        task_results = []
        task_count = 0

        logger.info(f"持续负载测试开始，持续时间: {test_duration}秒")

        while time.time() - start_time < test_duration:
            task_start_time = time.time()
            task_count += 1

            try:
                # 随机选择股票和参数
                selected_stocks = random.sample(stress_stock_list[:50], 3)
                window = random.choice([5, 10, 20])

                data = technical_engine.calculate_sma(
                    ts_codes=selected_stocks,
                    start_date=date.today() - timedelta(days=20),
                    end_date=date.today() - timedelta(days=1),
                    window=window,
                )

                task_time = time.time() - task_start_time

                task_results.append(
                    {"task_id": task_count, "task_time": task_time, "data_count": len(data), "success": True}
                )

                if task_count % 10 == 0:
                    logger.info(f"已完成{task_count}个任务，当前任务耗时: {task_time:.3f}秒")

            except Exception as e:
                task_time = time.time() - task_start_time
                task_results.append(
                    {"task_id": task_count, "task_time": task_time, "data_count": 0, "success": False, "error": str(e)}
                )

                logger.warning(f"任务{task_count}失败: {e}")

            # 短暂休息，模拟真实负载
            time.sleep(0.1)

        total_duration = time.time() - start_time

        # 分析持续负载结果
        successful_tasks = [r for r in task_results if r["success"]]
        failed_tasks = [r for r in task_results if not r["success"]]

        success_rate = len(successful_tasks) / len(task_results) if task_results else 0
        avg_task_time = np.mean([r["task_time"] for r in successful_tasks]) if successful_tasks else 0
        throughput = len(successful_tasks) / total_duration

        logger.info(f"持续负载测试结果:")
        logger.info(f"  测试时长: {total_duration:.2f}秒")
        logger.info(f"  总任务数: {len(task_results)}")
        logger.info(f"  成功任务数: {len(successful_tasks)}")
        logger.info(f"  失败任务数: {len(failed_tasks)}")
        logger.info(f"  成功率: {success_rate:.2%}")
        logger.info(f"  平均任务时间: {avg_task_time:.3f}秒")
        logger.info(f"  吞吐量: {throughput:.2f} 任务/秒")

        # 持续负载断言
        assert success_rate >= 0.9, f"持续负载成功率过低: {success_rate:.2%}"
        assert throughput >= 5.0, f"持续负载吞吐量过低: {throughput:.2f} 任务/秒"

    def test_database_connection_stress(self, test_engine, stress_stock_list):
        """数据库连接压力测试"""

        def database_task(task_id):
            """数据库任务"""
            try:
                with test_engine.connect() as conn:
                    # 模拟数据库查询
                    query = """
                    SELECT COUNT(*) as count
                    FROM stock_basic
                    WHERE ts_code = ANY(%s)
                    """

                    selected_stocks = random.sample(stress_stock_list[:20], 5)
                    result = conn.execute(query, (selected_stocks,))
                    count = result.fetchone().count

                    return {"task_id": task_id, "count": count, "success": True}

            except Exception as e:
                return {"task_id": task_id, "count": 0, "success": False, "error": str(e)}

        # 并发数据库连接测试
        num_concurrent = 20

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(database_task, i) for i in range(50)]

            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"数据库任务失败: {e}")
                    results.append({"task_id": -1, "count": 0, "success": False, "error": str(e)})

        total_time = time.time() - start_time

        # 分析数据库连接压力结果
        successful_tasks = [r for r in results if r["success"]]
        failed_tasks = [r for r in results if not r["success"]]

        success_rate = len(successful_tasks) / len(results) if results else 0

        logger.info(f"数据库连接压力测试结果:")
        logger.info(f"  总耗时: {total_time:.2f}秒")
        logger.info(f"  并发数: {num_concurrent}")
        logger.info(f"  总任务数: {len(results)}")
        logger.info(f"  成功率: {success_rate:.2%}")
        logger.info(f"  失败任务数: {len(failed_tasks)}")

        # 数据库连接压力断言
        assert success_rate >= 0.95, f"数据库连接成功率过低: {success_rate:.2%}"

    def test_error_recovery_stress(self, test_engine, stress_stock_list):
        """错误恢复压力测试"""
        technical_engine = TechnicalFactorEngine(test_engine)

        def error_prone_task(task_id):
            """容易出错的任务"""
            try:
                # 随机引入错误
                if random.random() < 0.3:  # 30%的错误率
                    if random.random() < 0.5:
                        # 无效股票代码
                        invalid_code = f"INVALID{task_id}.XX"
                        data = technical_engine.calculate_sma(
                            ts_codes=[invalid_code],
                            start_date=date.today() - timedelta(days=10),
                            end_date=date.today() - timedelta(days=1),
                            window=5,
                        )
                    else:
                        # 无效日期范围
                        data = technical_engine.calculate_sma(
                            ts_codes=[stress_stock_list[0]],
                            start_date=date.today(),
                            end_date=date.today() - timedelta(days=10),  # 错误的日期范围
                            window=5,
                        )
                else:
                    # 正常任务
                    selected_stock = random.choice(stress_stock_list[:10])
                    data = technical_engine.calculate_sma(
                        ts_codes=[selected_stock],
                        start_date=date.today() - timedelta(days=15),
                        end_date=date.today() - timedelta(days=1),
                        window=5,
                    )

                return {"task_id": task_id, "data_count": len(data) if not data.empty else 0, "success": True}

            except Exception as e:
                return {"task_id": task_id, "data_count": 0, "success": False, "error": str(e)}

        # 执行错误恢复测试
        num_tasks = 50

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(error_prone_task, i) for i in range(num_tasks)]

            results = []
            for future in futures:
                try:
                    result = future.result(timeout=15)
                    results.append(result)
                except Exception as e:
                    logger.error(f"错误恢复任务超时: {e}")
                    results.append({"task_id": -1, "data_count": 0, "success": False, "error": "timeout"})

        total_time = time.time() - start_time

        # 分析错误恢复结果
        successful_tasks = [r for r in results if r["success"]]
        failed_tasks = [r for r in results if not r["success"]]

        success_rate = len(successful_tasks) / len(results) if results else 0

        logger.info(f"错误恢复压力测试结果:")
        logger.info(f"  总耗时: {total_time:.2f}秒")
        logger.info(f"  总任务数: {len(results)}")
        logger.info(f"  成功任务数: {len(successful_tasks)}")
        logger.info(f"  失败任务数: {len(failed_tasks)}")
        logger.info(f"  成功率: {success_rate:.2%}")

        # 错误恢复断言
        assert success_rate >= 0.6, f"错误恢复成功率过低: {success_rate:.2%}"  # 考虑到故意引入的错误
        assert total_time < 120, f"错误恢复测试时间过长: {total_time:.2f}秒"


class TestSystemLimits:
    """系统极限测试"""

    @pytest.fixture(scope="class")
    def test_engine(self):
        """测试数据库引擎"""
        return get_db_engine()

    def test_maximum_concurrent_connections(self, test_engine):
        """测试最大并发连接数"""

        def connection_task(task_id):
            """连接任务"""
            try:
                with test_engine.connect() as conn:
                    result = conn.execute("SELECT 1 as test")
                    value = result.fetchone().test
                    time.sleep(1)  # 保持连接1秒

                    return {"task_id": task_id, "value": value, "success": True}

            except Exception as e:
                return {"task_id": task_id, "value": None, "success": False, "error": str(e)}

        # 逐步增加并发连接数
        max_connections_tested = 0

        for concurrent_count in [10, 20, 30, 40, 50]:
            logger.info(f"测试{concurrent_count}个并发连接...")

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=concurrent_count) as executor:
                futures = [executor.submit(connection_task, i) for i in range(concurrent_count)]

                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=10)
                        results.append(result)
                    except Exception as e:
                        results.append({"task_id": -1, "value": None, "success": False, "error": str(e)})

            test_time = time.time() - start_time
            successful_connections = len([r for r in results if r["success"]])
            success_rate = successful_connections / len(results)

            logger.info(
                f"  {concurrent_count}并发连接: {successful_connections}/{len(results)} 成功, "
                f"成功率: {success_rate:.2%}, 耗时: {test_time:.2f}秒"
            )

            if success_rate >= 0.9:
                max_connections_tested = concurrent_count
            else:
                logger.warning(f"  {concurrent_count}并发连接成功率过低，停止测试")
                break

        logger.info(f"最大并发连接测试完成，支持的最大并发数: {max_connections_tested}")

        # 极限测试断言
        assert max_connections_tested >= 20, f"支持的最大并发连接数过低: {max_connections_tested}"

    def test_large_data_processing_limit(self, test_engine):
        """测试大数据处理极限"""
        technical_engine = TechnicalFactorEngine(test_engine)

        # 测试不同数据规模
        data_scales = [(10, "小规模"), (50, "中规模"), (100, "大规模"), (200, "超大规模")]

        max_scale_processed = 0

        for stock_count, scale_name in data_scales:
            logger.info(f"测试{scale_name}数据处理: {stock_count}只股票...")

            # 生成测试股票列表
            test_stocks = [f"TEST{i:06d}.SZ" for i in range(stock_count)]

            start_time = time.time()

            try:
                data = technical_engine.calculate_sma(
                    ts_codes=test_stocks,
                    start_date=date.today() - timedelta(days=10),
                    end_date=date.today() - timedelta(days=1),
                    window=5,
                )

                processing_time = time.time() - start_time

                logger.info(f"  {scale_name}处理完成: {processing_time:.2f}秒, " f"结果数量: {len(data)}")

                # 性能检查
                if processing_time < 60:  # 1分钟内完成
                    max_scale_processed = stock_count
                else:
                    logger.warning(f"  {scale_name}处理时间过长: {processing_time:.2f}秒")
                    break

            except Exception as e:
                logger.error(f"  {scale_name}处理失败: {e}")
                break

        logger.info(f"大数据处理极限测试完成，最大处理规模: {max_scale_processed}只股票")

        # 极限测试断言
        assert max_scale_processed >= 50, f"大数据处理能力不足: {max_scale_processed}只股票"


if __name__ == "__main__":
    # 运行压力测试
    pytest.main([__file__, "-v", "-s", "--tb=short", "-k", "not test_maximum_concurrent_connections"])
