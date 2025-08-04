#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控器
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, field
from loguru import logger

from .factor_models import FactorResult, CalculationStatus


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_calculations: int = 0
    successful_calculations: int = 0
    failed_calculations: int = 0
    total_execution_time: timedelta = field(default_factory=lambda: timedelta(0))
    average_execution_time: timedelta = field(default_factory=lambda: timedelta(0))
    peak_memory_mb: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_calculations == 0:
            return 0.0
        return (self.successful_calculations / self.total_calculations) * 100
    
    @property
    def throughput(self) -> float:
        """吞吐量（任务/秒）"""
        total_seconds = (datetime.now() - self.start_time).total_seconds()
        if total_seconds == 0:
            return 0.0
        return self.total_calculations / total_seconds


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.calculation_history: List[Dict[str, Any]] = []
        self.resource_snapshots: List[Dict[str, Any]] = []
    
    def record_calculation_start(self, task_info: Dict[str, Any]):
        """记录计算开始"""
        task_info['start_time'] = datetime.now()
        self.calculation_history.append(task_info)
    
    def record_calculation_end(self, task_info: Dict[str, Any], result: FactorResult):
        """记录计算结束"""
        task_info['end_time'] = datetime.now()
        task_info['execution_time'] = result.execution_time
        task_info['status'] = result.status
        task_info['error_message'] = result.error_message
        
        # 更新总体指标
        self.update_metrics([result])
    
    def update_metrics(self, results: List[FactorResult]):
        """更新性能指标"""
        self.metrics.total_calculations += len(results)
        
        for result in results:
            if result.status == CalculationStatus.SUCCESS:
                self.metrics.successful_calculations += 1
            else:
                self.metrics.failed_calculations += 1
            
            self.metrics.total_execution_time += result.execution_time
        
        # 计算平均执行时间
        if self.metrics.total_calculations > 0:
            self.metrics.average_execution_time = (
                self.metrics.total_execution_time / self.metrics.total_calculations
            )
    
    def record_resource_snapshot(self, resource_info: Dict[str, Any]):
        """记录资源快照"""
        resource_info['timestamp'] = datetime.now()
        self.resource_snapshots.append(resource_info)
        
        # 更新峰值内存
        memory_mb = resource_info.get('memory_mb', 0)
        if memory_mb > self.metrics.peak_memory_mb:
            self.metrics.peak_memory_mb = memory_mb
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'metrics': {
                'total_calculations': self.metrics.total_calculations,
                'successful_calculations': self.metrics.successful_calculations,
                'failed_calculations': self.metrics.failed_calculations,
                'success_rate': self.metrics.success_rate,
                'throughput': self.metrics.throughput,
                'average_execution_time_seconds': self.metrics.average_execution_time.total_seconds(),
                'peak_memory_mb': self.metrics.peak_memory_mb
            },
            'calculation_count': len(self.calculation_history),
            'resource_snapshots_count': len(self.resource_snapshots)
        }
    
    def log_performance_summary(self):
        """记录性能摘要日志"""
        summary = self.get_performance_summary()
        metrics = summary['metrics']
        
        logger.info("=== 并行计算性能统计 ===")
        logger.info(f"总计算任务数: {metrics['total_calculations']}")
        logger.info(f"成功任务数: {metrics['successful_calculations']}")
        logger.info(f"失败任务数: {metrics['failed_calculations']}")
        logger.info(f"成功率: {metrics['success_rate']:.2f}%")
        logger.info(f"吞吐量: {metrics['throughput']:.2f} 任务/秒")
        logger.info(f"平均执行时间: {metrics['average_execution_time_seconds']:.3f}秒")
        logger.info(f"峰值内存使用: {metrics['peak_memory_mb']:.2f}MB")
        logger.info("========================")
    
    def reset_metrics(self):
        """重置性能指标"""
        self.metrics = PerformanceMetrics()
        self.calculation_history.clear()
        self.resource_snapshots.clear()
    
    def export_detailed_report(self) -> Dict[str, Any]:
        """导出详细报告"""
        return {
            'summary': self.get_performance_summary(),
            'calculation_history': self.calculation_history,
            'resource_snapshots': self.resource_snapshots,
            'export_time': datetime.now().isoformat()
        }