#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试工具类和辅助函数
"""

import time
import psutil
import os
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceReport:
    """性能测试报告"""
    test_name: str
    test_type: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    data_count: int
    throughput: float
    success: bool
    timestamp: str
    error_message: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save_to_file(self, file_path: str):
        """保存到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存性能报告失败: {e}")

class PerformanceReportManager:
    """性能报告管理器"""
    
    def __init__(self, report_dir: str = "performance_reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        self.reports: List[PerformanceReport] = []
    
    def add_report(self, report: PerformanceReport):
        """添加性能报告"""
        self.reports.append(report)
        
        # 保存单个报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.test_name}_{timestamp}.json"
        report_path = self.report_dir / filename
        report.save_to_file(str(report_path))
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """生成汇总报告"""
        if not self.reports:
            return {"message": "没有性能测试报告"}
        
        # 按测试类型分组
        by_type = {}
        for report in self.reports:
            test_type = report.test_type
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(report)
        
        summary = {
            "total_tests": len(self.reports),
            "successful_tests": len([r for r in self.reports if r.success]),
            "failed_tests": len([r for r in self.reports if not r.success]),
            "by_type": {}
        }
        
        for test_type, type_reports in by_type.items():
            successful_reports = [r for r in type_reports if r.success]
            
            if successful_reports:
                avg_execution_time = sum(r.execution_time for r in successful_reports) / len(successful_reports)
                avg_memory_usage = sum(r.memory_usage_mb for r in successful_reports) / len(successful_reports)
                avg_throughput = sum(r.throughput for r in successful_reports) / len(successful_reports)
                
                summary["by_type"][test_type] = {
                    "total_tests": len(type_reports),
                    "successful_tests": len(successful_reports),
                    "avg_execution_time": avg_execution_time,
                    "avg_memory_usage_mb": avg_memory_usage,
                    "avg_throughput": avg_throughput,
                    "best_throughput": max(r.throughput for r in successful_reports),
                    "worst_execution_time": max(r.execution_time for r in successful_reports)
                }
        
        return summary
    
    def save_summary_report(self):
        """保存汇总报告"""
        summary = self.generate_summary_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.report_dir / f"performance_summary_{timestamp}.json"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"性能汇总报告已保存: {summary_path}")
        except Exception as e:
            logger.error(f"保存汇总报告失败: {e}")

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_snapshots = []
        self.process = psutil.Process(os.getpid())
    
    def start_profiling(self):
        """开始性能分析"""
        self.start_time = time.time()
        self.memory_snapshots = [self.process.memory_info().rss / 1024 / 1024]
    
    def take_memory_snapshot(self, label: str = ""):
        """记录内存快照"""
        if self.start_time is None:
            raise RuntimeError("请先调用 start_profiling()")
        
        current_memory = self.process.memory_info().rss / 1024 / 1024
        elapsed_time = time.time() - self.start_time
        
        self.memory_snapshots.append({
            'timestamp': elapsed_time,
            'memory_mb': current_memory,
            'label': label
        })
    
    def end_profiling(self) -> Dict[str, Any]:
        """结束性能分析并返回结果"""
        if self.start_time is None:
            raise RuntimeError("请先调用 start_profiling()")
        
        self.end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 计算统计信息
        execution_time = self.end_time - self.start_time
        initial_memory = self.memory_snapshots[0] if isinstance(self.memory_snapshots[0], (int, float)) else self.memory_snapshots[0]['memory_mb']
        memory_usage = final_memory - initial_memory
        peak_memory = max(
            snapshot['memory_mb'] if isinstance(snapshot, dict) else snapshot 
            for snapshot in self.memory_snapshots
        )
        
        return {
            'execution_time': execution_time,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_usage_mb': memory_usage,
            'peak_memory_mb': peak_memory,
            'memory_snapshots': self.memory_snapshots
        }

def benchmark_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """基准测试函数装饰器"""
    profiler = PerformanceProfiler()
    
    try:
        profiler.start_profiling()
        result = func(*args, **kwargs)
        profiler.take_memory_snapshot("function_completed")
        
        profile_data = profiler.end_profiling()
        profile_data['success'] = True
        profile_data['result'] = result
        
        return profile_data
        
    except Exception as e:
        profile_data = profiler.end_profiling() if profiler.start_time else {}
        profile_data.update({
            'success': False,
            'error': str(e),
            'result': None
        })
        return profile_data

class PerformanceComparator:
    """性能比较器"""
    
    @staticmethod
    def compare_reports(report1: PerformanceReport, report2: PerformanceReport) -> Dict[str, Any]:
        """比较两个性能报告"""
        if not (report1.success and report2.success):
            return {"error": "只能比较成功的测试报告"}
        
        time_improvement = (report1.execution_time - report2.execution_time) / report1.execution_time * 100
        memory_improvement = (report1.memory_usage_mb - report2.memory_usage_mb) / report1.memory_usage_mb * 100
        throughput_improvement = (report2.throughput - report1.throughput) / report1.throughput * 100
        
        return {
            "baseline": report1.test_name,
            "comparison": report2.test_name,
            "time_improvement_percent": time_improvement,
            "memory_improvement_percent": memory_improvement,
            "throughput_improvement_percent": throughput_improvement,
            "summary": {
                "faster": time_improvement > 0,
                "less_memory": memory_improvement > 0,
                "higher_throughput": throughput_improvement > 0
            }
        }
    
    @staticmethod
    def find_performance_regressions(reports: List[PerformanceReport], 
                                   threshold_percent: float = 10.0) -> List[Dict[str, Any]]:
        """查找性能回归"""
        regressions = []
        
        # 按测试名称分组
        by_name = {}
        for report in reports:
            if report.test_name not in by_name:
                by_name[report.test_name] = []
            by_name[report.test_name].append(report)
        
        # 检查每个测试的性能趋势
        for test_name, test_reports in by_name.items():
            if len(test_reports) < 2:
                continue
            
            # 按时间排序
            test_reports.sort(key=lambda r: r.timestamp)
            
            # 比较最新和之前的报告
            latest = test_reports[-1]
            previous = test_reports[-2]
            
            if latest.success and previous.success:
                time_regression = (latest.execution_time - previous.execution_time) / previous.execution_time * 100
                memory_regression = (latest.memory_usage_mb - previous.memory_usage_mb) / previous.memory_usage_mb * 100
                
                if time_regression > threshold_percent or memory_regression > threshold_percent:
                    regressions.append({
                        "test_name": test_name,
                        "time_regression_percent": time_regression,
                        "memory_regression_percent": memory_regression,
                        "previous_execution_time": previous.execution_time,
                        "latest_execution_time": latest.execution_time,
                        "previous_memory_mb": previous.memory_usage_mb,
                        "latest_memory_mb": latest.memory_usage_mb
                    })
        
        return regressions

# 全局性能报告管理器
performance_report_manager = PerformanceReportManager()