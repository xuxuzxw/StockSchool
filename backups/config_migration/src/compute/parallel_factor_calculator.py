#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行因子计算器
实现多进程并行因子计算和性能优化
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from loguru import logger
from sqlalchemy import text
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import gc
import time
from functools import partial
import pickle
import os

from .factor_models import FactorType, FactorCategory, FactorResult, CalculationStatus
from .technical_factor_engine import TechnicalFactorEngine
from .fundamental_factor_engine import FundamentalFactorEngine
from .sentiment_factor_engine import SentimentFactorEngine


class ResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        """初始化资源监控器"""
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
    
    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存使用量(MB)
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存使用量(MB)
            'percent': self.process.memory_percent(),  # 内存使用百分比
            'available_mb': system_memory.available / 1024 / 1024,  # 可用内存(MB)
            'total_mb': system_memory.total / 1024 / 1024  # 总内存(MB)
        }
    
    def get_disk_usage(self, path: str = '/') -> Dict[str, float]:
        """获取磁盘使用情况"""
        disk_usage = psutil.disk_usage(path)
        
        return {
            'total_gb': disk_usage.total / 1024 / 1024 / 1024,
            'used_gb': disk_usage.used / 1024 / 1024 / 1024,
            'free_gb': disk_usage.free / 1024 / 1024 / 1024,
            'percent': (disk_usage.used / disk_usage.total) * 100
        }
    
    def check_resource_availability(self, min_memory_mb: float = 1000,
                                  max_cpu_percent: float = 80) -> Dict[str, bool]:
        """检查资源可用性"""
        memory_info = self.get_memory_usage()
        cpu_usage = self.get_cpu_usage()
        
        return {
            'memory_available': memory_info['available_mb'] > min_memory_mb,
            'cpu_available': cpu_usage < max_cpu_percent,
            'overall_available': (memory_info['available_mb'] > min_memory_mb and 
                                cpu_usage < max_cpu_percent)
        }


class TaskLoadBalancer:
    """任务负载均衡器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        """初始化负载均衡器"""
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.resource_monitor = ResourceMonitor()
    
    def calculate_optimal_workers(self, task_count: int, 
                                task_complexity: str = 'medium') -> int:
        """计算最优工作进程数"""
        # 基础工作进程数
        cpu_count = os.cpu_count() or 1
        
        # 根据任务复杂度调整
        complexity_factors = {
            'low': 2.0,      # 简单任务可以更多并行
            'medium': 1.5,   # 中等复杂度
            'high': 1.0,     # 复杂任务减少并行
            'very_high': 0.5 # 非常复杂的任务
        }
        
        factor = complexity_factors.get(task_complexity, 1.5)
        optimal_workers = int(cpu_count * factor)
        
        # 考虑任务数量
        optimal_workers = min(optimal_workers, task_count, self.max_workers)
        
        # 检查系统资源
        resource_status = self.resource_monitor.check_resource_availability()
        if not resource_status['overall_available']:
            optimal_workers = max(1, optimal_workers // 2)
        
        return max(1, optimal_workers)
    
    def split_tasks(self, tasks: List[Any], chunk_size: Optional[int] = None) -> List[List[Any]]:
        """将任务分割为适合并行处理的块"""
        if chunk_size is None:
            # 自动计算块大小
            total_tasks = len(tasks)
            optimal_workers = self.calculate_optimal_workers(total_tasks)
            chunk_size = max(1, total_tasks // optimal_workers)
        
        chunks = []
        for i in range(0, len(tasks), chunk_size):
            chunks.append(tasks[i:i + chunk_size])
        
        return chunks
    
    def balance_workload(self, tasks: List[Any], 
                        worker_capacities: Optional[List[float]] = None) -> List[List[Any]]:
        """根据工作进程能力平衡工作负载"""
        if not worker_capacities:
            # 默认所有工作进程能力相同
            num_workers = self.calculate_optimal_workers(len(tasks))
            worker_capacities = [1.0] * num_workers
        
        # 按能力分配任务
        total_capacity = sum(worker_capacities)
        task_assignments = [[] for _ in worker_capacities]
        
        for i, task in enumerate(tasks):
            # 简单的轮询分配，可以根据需要改进
            worker_idx = i % len(worker_capacities)
            task_assignments[worker_idx].append(task)
        
        return task_assignments


def calculate_stock_factors_worker(args: Tuple[str, str, Any, Dict[str, Any]]) -> FactorResult:
    """
    工作进程函数：计算单个股票的因子
    
    Args:
        args: (ts_code, factor_type, engine_config, calculation_params)
        
    Returns:
        因子计算结果
    """
    ts_code, factor_type, engine_config, calculation_params = args
    
    try:
        # 重新创建数据库引擎（避免跨进程共享连接）
        from sqlalchemy import create_engine
        engine = create_engine(engine_config['database_url'])
        
        # 根据因子类型创建相应的计算引擎
        if factor_type == 'technical':
            calculator = TechnicalFactorEngine(engine)
        elif factor_type == 'fundamental':
            calculator = FundamentalFactorEngine(engine)
        elif factor_type == 'sentiment':
            calculator = SentimentFactorEngine(engine)
        else:
            raise ValueError(f"不支持的因子类型: {factor_type}")
        
        # 执行因子计算
        result = calculator.calculate_factors(
            ts_code=ts_code,
            start_date=calculation_params.get('start_date'),
            end_date=calculation_params.get('end_date'),
            factor_names=calculation_params.get('factor_names')
        )
        
        # 清理资源
        engine.dispose()
        
        return result
        
    except Exception as e:
        logger.error(f"计算股票 {ts_code} 的 {factor_type} 因子失败: {e}")
        return FactorResult(
            ts_code=ts_code,
            factor_type=FactorType.TECHNICAL,  # 默认类型
            status=CalculationStatus.FAILED,
            error_message=str(e),
            execution_time=timedelta(0),
            data_points=0,
            factors={}
        )


def calculate_batch_factors_worker(args: Tuple[List[str], str, Any, Dict[str, Any]]) -> List[FactorResult]:
    """
    工作进程函数：批量计算多个股票的因子
    
    Args:
        args: (ts_codes, factor_type, engine_config, calculation_params)
        
    Returns:
        因子计算结果列表
    """
    ts_codes, factor_type, engine_config, calculation_params = args
    
    results = []
    engine = None
    
    try:
        # 创建数据库引擎
        from sqlalchemy import create_engine
        engine = create_engine(engine_config['database_url'])
        
        # 根据因子类型创建相应的计算引擎
        if factor_type == 'technical':
            calculator = TechnicalFactorEngine(engine)
        elif factor_type == 'fundamental':
            calculator = FundamentalFactorEngine(engine)
        elif factor_type == 'sentiment':
            calculator = SentimentFactorEngine(engine)
        else:
            raise ValueError(f"不支持的因子类型: {factor_type}")
        
        # 批量计算因子
        for ts_code in ts_codes:
            try:
                result = calculator.calculate_factors(
                    ts_code=ts_code,
                    start_date=calculation_params.get('start_date'),
                    end_date=calculation_params.get('end_date'),
                    factor_names=calculation_params.get('factor_names')
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"计算股票 {ts_code} 的 {factor_type} 因子失败: {e}")
                results.append(FactorResult(
                    ts_code=ts_code,
                    factor_type=FactorType.TECHNICAL,
                    status=CalculationStatus.FAILED,
                    error_message=str(e),
                    execution_time=timedelta(0),
                    data_points=0,
                    factors={}
                ))
        
    except Exception as e:
        logger.error(f"批量计算因子失败: {e}")
        # 为所有股票返回失败结果
        for ts_code in ts_codes:
            results.append(FactorResult(
                ts_code=ts_code,
                factor_type=FactorType.TECHNICAL,
                status=CalculationStatus.FAILED,
                error_message=str(e),
                execution_time=timedelta(0),
                data_points=0,
                factors={}
            ))
    
    finally:
        # 清理资源
        if engine:
            engine.dispose()
        gc.collect()  # 强制垃圾回收
    
    return results


class ParallelFactorCalculator:
    """并行因子计算器 - 重构版本"""
    
    def __init__(self, engine, config: Optional['ParallelCalculationConfig'] = None):
        """初始化并行因子计算器"""
        from .parallel_config import ParallelCalculationConfig
        from .engine_factory import EngineConnectionManager
        from .performance_monitor import PerformanceMonitor
        
        self.config = config or ParallelCalculationConfig()
        
        # 组件初始化
        self.connection_manager = EngineConnectionManager(
            str(engine.url), 
            self.config.database_pool_size
        )
        self.load_balancer = TaskLoadBalancer(self.config.worker_config.max_workers)
        self.resource_monitor = ResourceMonitor()
        self.performance_monitor = PerformanceMonitor()
    
    def calculate_factors_parallel(self, ts_codes: List[str],
                                 factor_types: List[str],
                                 start_date: Optional[date] = None,
                                 end_date: Optional[date] = None,
                                 factor_names: Optional[List[str]] = None,
                                 batch_mode: bool = True,
                                 max_workers: Optional[int] = None) -> Dict[str, List[FactorResult]]:
        """
        并行计算多个股票的多种因子
        
        Args:
            ts_codes: 股票代码列表
            factor_types: 因子类型列表
            start_date: 开始日期
            end_date: 结束日期
            factor_names: 指定因子名称列表
            batch_mode: 是否使用批量模式
            max_workers: 最大工作进程数
            
        Returns:
            按因子类型分组的计算结果
        """
        start_time = datetime.now()
        logger.info(f"开始并行计算因子: {len(ts_codes)}只股票, {len(factor_types)}种因子类型")
        
        # 准备计算参数
        calculation_params = {
            'start_date': start_date,
            'end_date': end_date,
            'factor_names': factor_names
        }
        
        engine_config = {
            'database_url': self.database_url
        }
        
        all_results = {}
        
        try:
            for factor_type in factor_types:
                logger.info(f"开始计算 {factor_type} 类型因子")
                
                if batch_mode:
                    # 批量模式：将股票分组后并行处理
                    results = self._calculate_batch_parallel(
                        ts_codes, factor_type, engine_config, 
                        calculation_params, max_workers
                    )
                else:
                    # 单股票模式：每个股票单独并行处理
                    results = self._calculate_individual_parallel(
                        ts_codes, factor_type, engine_config,
                        calculation_params, max_workers
                    )
                
                all_results[factor_type] = results
                
                # 更新性能统计
                self._update_performance_stats(results)
                
                # 强制垃圾回收
                gc.collect()
        
        except Exception as e:
            logger.error(f"并行计算因子失败: {e}")
            raise
        
        finally:
            # 记录总执行时间
            total_time = datetime.now() - start_time
            logger.info(f"并行计算完成，总耗时: {total_time}")
            
            # 输出性能统计
            self._log_performance_stats()
        
        return all_results
    
    def _calculate_batch_parallel(self, ts_codes: List[str], factor_type: str,
                                engine_config: Dict[str, Any], 
                                calculation_params: Dict[str, Any],
                                max_workers: Optional[int] = None) -> List[FactorResult]:
        """批量并行计算"""
        # 计算最优工作进程数
        if max_workers is None:
            max_workers = self.load_balancer.calculate_optimal_workers(
                len(ts_codes), task_complexity='medium'
            )
        
        # 将股票分组
        stock_chunks = self.load_balancer.split_tasks(ts_codes)
        
        logger.info(f"使用 {max_workers} 个进程，分 {len(stock_chunks)} 个批次处理")
        
        all_results = []
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_chunk = {}
            for chunk in stock_chunks:
                if chunk:  # 确保块不为空
                    future = executor.submit(
                        calculate_batch_factors_worker,
                        (chunk, factor_type, engine_config, calculation_params)
                    )
                    future_to_chunk[future] = chunk
            
            # 收集结果
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result(timeout=300)  # 5分钟超时
                    all_results.extend(chunk_results)
                    logger.info(f"完成 {len(chunk)} 只股票的计算")
                    
                except Exception as e:
                    logger.error(f"批次计算失败: {e}")
                    # 为失败的批次创建失败结果
                    for ts_code in chunk:
                        all_results.append(FactorResult(
                            ts_code=ts_code,
                            factor_type=FactorType.TECHNICAL,
                            status=CalculationStatus.FAILED,
                            error_message=str(e),
                            execution_time=timedelta(0),
                            data_points=0,
                            factors={}
                        ))
        
        return all_results
    
    def _calculate_individual_parallel(self, ts_codes: List[str], factor_type: str,
                                     engine_config: Dict[str, Any],
                                     calculation_params: Dict[str, Any],
                                     max_workers: Optional[int] = None) -> List[FactorResult]:
        """单股票并行计算"""
        # 计算最优工作进程数
        if max_workers is None:
            max_workers = self.load_balancer.calculate_optimal_workers(
                len(ts_codes), task_complexity='low'
            )
        
        logger.info(f"使用 {max_workers} 个进程并行处理 {len(ts_codes)} 只股票")
        
        all_results = []
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_stock = {}
            for ts_code in ts_codes:
                future = executor.submit(
                    calculate_stock_factors_worker,
                    (ts_code, factor_type, engine_config, calculation_params)
                )
                future_to_stock[future] = ts_code
            
            # 收集结果
            for future in as_completed(future_to_stock):
                ts_code = future_to_stock[future]
                try:
                    result = future.result(timeout=60)  # 1分钟超时
                    all_results.append(result)
                    
                    if result.status == CalculationStatus.SUCCESS:
                        logger.debug(f"成功计算股票 {ts_code} 的因子")
                    else:
                        logger.warning(f"股票 {ts_code} 计算失败: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"股票 {ts_code} 计算异常: {e}")
                    all_results.append(FactorResult(
                        ts_code=ts_code,
                        factor_type=FactorType.TECHNICAL,
                        status=CalculationStatus.FAILED,
                        error_message=str(e),
                        execution_time=timedelta(0),
                        data_points=0,
                        factors={}
                    ))
        
        return all_results
    
    def _update_performance_stats(self, results: List[FactorResult]):
        """更新性能统计"""
        self.performance_stats['total_calculations'] += len(results)
        
        for result in results:
            if result.status == CalculationStatus.SUCCESS:
                self.performance_stats['successful_calculations'] += 1
            else:
                self.performance_stats['failed_calculations'] += 1
            
            self.performance_stats['total_execution_time'] += result.execution_time
        
        # 计算平均执行时间
        if self.performance_stats['total_calculations'] > 0:
            self.performance_stats['average_execution_time'] = (
                self.performance_stats['total_execution_time'] / 
                self.performance_stats['total_calculations']
            )
    
    def _log_performance_stats(self):
        """记录性能统计"""
        stats = self.performance_stats
        
        success_rate = (stats['successful_calculations'] / 
                       max(1, stats['total_calculations'])) * 100
        
        logger.info("=== 并行计算性能统计 ===")
        logger.info(f"总计算任务数: {stats['total_calculations']}")
        logger.info(f"成功任务数: {stats['successful_calculations']}")
        logger.info(f"失败任务数: {stats['failed_calculations']}")
        logger.info(f"成功率: {success_rate:.2f}%")
        logger.info(f"总执行时间: {stats['total_execution_time']}")
        logger.info(f"平均执行时间: {stats['average_execution_time']}")
        
        # 系统资源使用情况
        memory_info = self.resource_monitor.get_memory_usage()
        logger.info(f"内存使用: {memory_info['rss_mb']:.2f}MB")
        logger.info("========================")
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        # 强制垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收释放了 {collected} 个对象")
        
        # 获取内存使用情况
        memory_info = self.resource_monitor.get_memory_usage()
        logger.info(f"当前内存使用: {memory_info['rss_mb']:.2f}MB")
        
        # 如果内存使用过高，建议减少并行度
        if memory_info['percent'] > 80:
            logger.warning("内存使用率过高，建议减少并行工作进程数")
            return False
        
        return True
    
    def get_optimal_batch_size(self, total_stocks: int, 
                             available_memory_mb: float = None) -> int:
        """计算最优批次大小"""
        if available_memory_mb is None:
            memory_info = self.resource_monitor.get_memory_usage()
            available_memory_mb = memory_info['available_mb']
        
        # 估算每个股票计算需要的内存（MB）
        memory_per_stock = 10  # 经验值，可根据实际情况调整
        
        # 计算最大批次大小
        max_batch_size = int(available_memory_mb / memory_per_stock)
        
        # 考虑CPU核心数
        cpu_count = os.cpu_count() or 1
        optimal_batch_size = min(max_batch_size, total_stocks // cpu_count, 100)
        
        return max(1, optimal_batch_size)
    
    def calculate_with_resource_monitoring(self, ts_codes: List[str],
                                         factor_types: List[str],
                                         **kwargs) -> Dict[str, List[FactorResult]]:
        """带资源监控的因子计算"""
        # 检查初始资源状态
        initial_resources = self.resource_monitor.check_resource_availability()
        if not initial_resources['overall_available']:
            logger.warning("系统资源不足，可能影响计算性能")
        
        # 动态调整批次大小
        optimal_batch_size = self.get_optimal_batch_size(len(ts_codes))
        logger.info(f"推荐批次大小: {optimal_batch_size}")
        
        try:
            # 执行计算
            results = self.calculate_factors_parallel(ts_codes, factor_types, **kwargs)
            
            # 计算完成后优化内存
            self.optimize_memory_usage()
            
            return results
            
        except Exception as e:
            logger.error(f"资源监控计算失败: {e}")
            raise