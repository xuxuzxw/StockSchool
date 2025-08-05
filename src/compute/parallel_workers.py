import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from typing import Any, Dict, List, Tuple

from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行工作进程执行器
"""


from .engine_factory import FactorEngineFactory
from .factor_models import CalculationStatus, FactorResult, FactorType
from .parallel_config import ParallelCalculationConfig


class WorkerTaskResult:
    """工作任务结果封装"""

    def __init__(self, ts_code: str, success: bool, result: FactorResult = None, error: str = None):
        self.ts_code = ts_code
        self.success = success
        self.result = result
        self.error = error


def create_error_result(ts_code: str, factor_type: str, error_message: str) -> FactorResult:
    """创建错误结果的工厂函数"""
    # 转换字符串类型到枚举
    factor_type_enum = {
        "technical": FactorType.TECHNICAL,
        "fundamental": FactorType.FUNDAMENTAL,
        "sentiment": FactorType.SENTIMENT,
    }.get(factor_type, FactorType.TECHNICAL)

    return FactorResult(
        ts_code=ts_code,
        factor_type=factor_type_enum,
        status=CalculationStatus.FAILED,
        error_message=error_message,
        execution_time=timedelta(0),
        data_points=0,
        factors={},
    )


def execute_single_stock_calculation(args: Tuple[str, str, str, Dict[str, Any]]) -> FactorResult:
    """
    单股票计算工作函数

    Args:
        args: (ts_code, factor_type, database_url, calculation_params)
    """
    ts_code, factor_type, database_url, calculation_params = args

    try:
        # 使用工厂创建引擎
        engine = FactorEngineFactory.create_for_worker(database_url, factor_type)

        # 执行计算
        result = engine.calculate_factors(
            ts_code=ts_code,
            start_date=calculation_params.get("start_date"),
            end_date=calculation_params.get("end_date"),
            factor_names=calculation_params.get("factor_names"),
        )

        return result

    except Exception as e:
        logger.error(f"股票 {ts_code} 计算失败: {e}")
        return create_error_result(ts_code, factor_type, str(e))


def execute_batch_calculation(args: Tuple[List[str], str, str, Dict[str, Any]]) -> List[FactorResult]:
    """
    批量计算工作函数

    Args:
        args: (ts_codes, factor_type, database_url, calculation_params)
    """
    ts_codes, factor_type, database_url, calculation_params = args
    results = []

    try:
        # 使用工厂创建引擎
        engine = FactorEngineFactory.create_for_worker(database_url, factor_type)

        # 批量计算
        for ts_code in ts_codes:
            try:
                result = engine.calculate_factors(
                    ts_code=ts_code,
                    start_date=calculation_params.get("start_date"),
                    end_date=calculation_params.get("end_date"),
                    factor_names=calculation_params.get("factor_names"),
                )
                results.append(result)

            except Exception as e:
                logger.error(f"股票 {ts_code} 计算失败: {e}")
                results.append(create_error_result(ts_code, factor_type, str(e)))

    except Exception as e:
        logger.error(f"批量计算失败: {e}")
        # 为所有股票创建错误结果
        for ts_code in ts_codes:
            results.append(create_error_result(ts_code, factor_type, str(e)))

    finally:
        # 强制垃圾回收
        gc.collect()

    return results


class BaseWorkerExecutor:
    """工作执行器基类"""

    def __init__(self, max_workers: int, config: ParallelCalculationConfig = None):
        """方法描述"""
        self.config = config or ParallelCalculationget_config()

    def _handle_future_result(self, future, context_info: str) -> Any:
        """处理Future结果的通用方法"""
        try:
            return future.result(timeout=self.config.worker_config.default_timeout_seconds)
        except Exception as e:
            logger.error(f"{context_info} 执行失败: {e}")
            raise


class IndividualWorkerExecutor(BaseWorkerExecutor):
    """单个任务执行器"""

    def execute_individual(self, tasks: List[Tuple], **kwargs) -> List[FactorResult]:
        """执行单个任务并行计算"""
        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_task = {executor.submit(execute_single_stock_calculation, task): task for task in tasks}

            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                ts_code = task[0]  # 第一个参数是ts_code

                try:
                    result = self._handle_future_result(future, f"股票 {ts_code}")
                    results.append(result)
                except Exception as e:
                    # 创建错误结果
                    error_result = create_error_result(ts_code, task[1], str(e))
                    results.append(error_result)

        return results


class BatchWorkerExecutor(BaseWorkerExecutor):
    """批量任务执行器"""

    def execute_batch(self, tasks: List[Tuple], batch_size: int = None, **kwargs) -> List[FactorResult]:
        """执行批量并行计算"""
        if batch_size is None:
            batch_size = self.config.batch_config.default_batch_size

        # 将任务分组
        task_batches = self._create_batches(tasks, batch_size)
        all_results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交批量任务
            future_to_batch = {executor.submit(execute_batch_calculation, batch): batch for batch in task_batches}

            # 收集结果
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]

                try:
                    batch_results = self._handle_future_result(future, f"批次 {len(batch)} 个任务")
                    all_results.extend(batch_results)
                except Exception as e:
                    # 为批次中的所有任务创建错误结果
                    for task in batch:
                        ts_codes = task[0]  # 批量任务的第一个参数是ts_codes列表
                        factor_type = task[1]
                        for ts_code in ts_codes:
                            error_result = create_error_result(ts_code, factor_type, str(e))
                            all_results.append(error_result)

        return all_results

    def _create_batches(self, tasks: List[Tuple], batch_size: int) -> List[Tuple]:
        """创建任务批次"""
        # 这里需要根据实际的任务结构来重新组织
        # 假设tasks是单个股票的任务列表，需要重新组织为批量任务
        batches = []

        # 按因子类型分组
        type_groups = {}
        for task in tasks:
            ts_code, factor_type, database_url, calc_params = task
            key = (factor_type, database_url, str(calc_params))

            if key not in type_groups:
                type_groups[key] = []
            type_groups[key].append(ts_code)

        # 为每个分组创建批次
        for (factor_type, database_url, calc_params_str), ts_codes in type_groups.items():
            # 恢复calc_params（这里简化处理）
            calc_params = eval(calc_params_str) if calc_params_str != "{}" else {}

            # 分批处理
            for i in range(0, len(ts_codes), batch_size):
                batch_codes = ts_codes[i : i + batch_size]
                batches.append((batch_codes, factor_type, database_url, calc_params))

        return batches
