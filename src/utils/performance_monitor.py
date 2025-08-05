import functools
import gc
import json
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控工具
提供因子计算过程中的性能监控和资源使用统计功能
"""


@dataclass
class PerformanceMetrics:
    """性能指标"""

    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    cpu_usage_start: float = 0.0
    cpu_usage_end: float = 0.0
    cpu_usage_avg: float = 0.0
    memory_usage_start: float = 0.0
    memory_usage_end: float = 0.0
    memory_usage_peak: float = 0.0
    memory_allocated: float = 0.0
    disk_io_read: float = 0.0
    disk_io_write: float = 0.0
    network_io_sent: float = 0.0
    network_io_recv: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "cpu_usage_start": self.cpu_usage_start,
            "cpu_usage_end": self.cpu_usage_end,
            "cpu_usage_avg": self.cpu_usage_avg,
            "memory_usage_start": self.memory_usage_start,
            "memory_usage_end": self.memory_usage_end,
            "memory_usage_peak": self.memory_usage_peak,
            "memory_allocated": self.memory_allocated,
            "disk_io_read": self.disk_io_read,
            "disk_io_write": self.disk_io_write,
            "network_io_sent": self.network_io_sent,
            "network_io_recv": self.network_io_recv,
            "custom_metrics": self.custom_metrics,
        }


class ResourceMonitor:
    """资源监控器"""

    def __init__(self, interval: float = 1.0):
        """方法描述"""
        self.monitoring = False
        self.monitor_thread = None
        self.cpu_samples = deque(maxlen=1000)
        self.memory_samples = deque(maxlen=1000)
        self.disk_io_samples = deque(maxlen=1000)
        self.network_io_samples = deque(maxlen=1000)
        self.process = psutil.Process()

    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                timestamp = time.time()

                # CPU使用率
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append((timestamp, cpu_percent))

                # 内存使用
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_samples.append((timestamp, memory_mb))

                # 磁盘IO
                disk_io = self.process.io_counters()
                self.disk_io_samples.append((timestamp, disk_io.read_bytes, disk_io.write_bytes))

                # 网络IO（系统级别）
                try:
                    net_io = psutil.net_io_counters()
                    self.network_io_samples.append((timestamp, net_io.bytes_sent, net_io.bytes_recv))
                except:
                    pass

                time.sleep(self.interval)

            except Exception:
                # 忽略监控过程中的错误
                time.sleep(self.interval)

    def get_current_stats(self) -> Dict[str, float]:
        """获取当前统计信息"""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            disk_io = self.process.io_counters()

            stats = {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "disk_read_mb": disk_io.read_bytes / 1024 / 1024,
                "disk_write_mb": disk_io.write_bytes / 1024 / 1024,
            }

            try:
                net_io = psutil.net_io_counters()
                stats.update(
                    {
                        "network_sent_mb": net_io.bytes_sent / 1024 / 1024,
                        "network_recv_mb": net_io.bytes_recv / 1024 / 1024,
                    }
                )
            except:
                pass

            return stats

        except Exception:
            return {}

    def get_average_stats(self, duration_seconds: int = 60) -> Dict[str, float]:
        """获取指定时间内的平均统计信息"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds

        # CPU平均值
        cpu_values = [cpu for ts, cpu in self.cpu_samples if ts >= cutoff_time]
        avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0

        # 内存平均值
        memory_values = [mem for ts, mem in self.memory_samples if ts >= cutoff_time]
        avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
        peak_memory = max(memory_values) if memory_values else 0

        return {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory,
            "peak_memory_mb": peak_memory,
            "sample_count": len(cpu_values),
        }


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, enable_memory_tracking: bool = True):
        """方法描述"""
        self.resource_monitor = ResourceMonitor()
        self.active_sessions = {}
        self.completed_sessions = []
        self.function_stats = defaultdict(list)

        if enable_memory_tracking:
            tracemalloc.start()

    def start_session(self, session_name: str) -> str:
        """开始性能监控会话"""
        session_id = f"{session_name}_{int(time.time() * 1000)}"

        # 获取初始状态
        initial_stats = self.resource_monitor.get_current_stats()

        metrics = PerformanceMetrics(
            start_time=datetime.now(),
            cpu_usage_start=initial_stats.get("cpu_percent", 0),
            memory_usage_start=initial_stats.get("memory_mb", 0),
            disk_io_read=initial_stats.get("disk_read_mb", 0),
            disk_io_write=initial_stats.get("disk_write_mb", 0),
            network_io_sent=initial_stats.get("network_sent_mb", 0),
            network_io_recv=initial_stats.get("network_recv_mb", 0),
        )

        self.active_sessions[session_id] = metrics

        # 开始资源监控
        if not self.resource_monitor.monitoring:
            self.resource_monitor.start_monitoring()

        return session_id

    def end_session(self, session_id: str) -> PerformanceMetrics:
        """结束性能监控会话"""
        if session_id not in self.active_sessions:
            raise ValueError(f"会话 {session_id} 不存在")

        metrics = self.active_sessions[session_id]

        # 获取结束状态
        final_stats = self.resource_monitor.get_current_stats()

        metrics.end_time = datetime.now()
        metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.cpu_usage_end = final_stats.get("cpu_percent", 0)
        metrics.memory_usage_end = final_stats.get("memory_mb", 0)

        # 计算差值
        metrics.disk_io_read = final_stats.get("disk_read_mb", 0) - metrics.disk_io_read
        metrics.disk_io_write = final_stats.get("disk_write_mb", 0) - metrics.disk_io_write
        metrics.network_io_sent = final_stats.get("network_sent_mb", 0) - metrics.network_io_sent
        metrics.network_io_recv = final_stats.get("network_recv_mb", 0) - metrics.network_io_recv

        # 获取平均CPU使用率
        avg_stats = self.resource_monitor.get_average_stats(int(metrics.duration) + 1)
        metrics.cpu_usage_avg = avg_stats.get("avg_cpu_percent", 0)
        metrics.memory_usage_peak = avg_stats.get("peak_memory_mb", metrics.memory_usage_end)

        # 内存分配统计
        if self.enable_memory_tracking:
            try:
                current, peak = tracemalloc.get_traced_memory()
                metrics.memory_allocated = peak / 1024 / 1024  # MB
            except:
                pass

        # 移动到已完成会话
        del self.active_sessions[session_id]
        self.completed_sessions.append(metrics)

        return metrics

    def add_custom_metric(self, session_id: str, key: str, value: Any):
        """添加自定义指标"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].custom_metrics[key] = value

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """上下文管理器，用于监控操作"""
        session_id = self.start_session(operation_name)
        try:
            yield session_id
        finally:
            metrics = self.end_session(session_id)
            self.function_stats[operation_name].append(metrics)

    def get_function_statistics(self, function_name: str) -> Dict[str, Any]:
        """获取函数统计信息"""
        if function_name not in self.function_stats:
            return {}

        metrics_list = self.function_stats[function_name]
        if not metrics_list:
            return {}

        durations = [m.duration for m in metrics_list]
        cpu_usages = [m.cpu_usage_avg for m in metrics_list]
        memory_peaks = [m.memory_usage_peak for m in metrics_list]

        return {
            "call_count": len(metrics_list),
            "total_duration": sum(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_cpu_usage": sum(cpu_usages) / len(cpu_usages),
            "avg_memory_peak": sum(memory_peaks) / len(memory_peaks),
            "last_call": metrics_list[-1].start_time.isoformat(),
        }

    def get_all_statistics(self) -> Dict[str, Any]:
        """获取所有统计信息"""
        stats = {
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.completed_sessions),
            "function_stats": {},
        }

        for func_name in self.function_stats:
            stats["function_stats"][func_name] = self.get_function_statistics(func_name)

        return stats

    def export_statistics(self, file_path: str):
        """导出统计信息到文件"""
        stats = self.get_all_statistics()

        # 添加已完成会话的详细信息
        stats["completed_sessions_detail"] = [
            metrics.to_dict() for metrics in self.completed_sessions[-100:]  # 最近100个
        ]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def clear_statistics(self):
        """清除统计信息"""
        self.completed_sessions.clear()
        self.function_stats.clear()

        if self.enable_memory_tracking:
            try:
                tracemalloc.clear_traces()
            except:
                pass

    def cleanup(self):
        """清理资源"""
        self.resource_monitor.stop_monitoring()

        if self.enable_memory_tracking:
            try:
                tracemalloc.stop()
            except:
                pass


def performance_monitor(monitor: PerformanceMonitor = None, operation_name: str = None):
    """性能监控装饰器"""

    def decorator(func):
        """方法描述"""

        def wrapper(*args, **kwargs):
            """方法描述"""

            if monitor is None:
                monitor = PerformanceMonitor()

            if operation_name is None:
                operation_name = func.__name__

            with monitor.monitor_operation(operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class FactorPerformanceMonitor(PerformanceMonitor):
    """因子计算专用性能监控器"""

    def __init__(self):
        """方法描述"""
        self.factor_stats = defaultdict(dict)
        self.stock_processing_stats = defaultdict(list)

    def record_factor_calculation(
        self, factor_name: str, stock_count: int, calculation_time: float, data_size_mb: float
    ):
        """记录因子计算统计"""
        self.factor_stats[factor_name] = {
            "stock_count": stock_count,
            "calculation_time": calculation_time,
            "data_size_mb": data_size_mb,
            "throughput_stocks_per_sec": stock_count / calculation_time if calculation_time > 0 else 0,
            "throughput_mb_per_sec": data_size_mb / calculation_time if calculation_time > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def record_stock_processing(self, stock_code: str, processing_time: float, factor_count: int):
        """记录股票处理统计"""
        self.stock_processing_stats[stock_code].append(
            {"processing_time": processing_time, "factor_count": factor_count, "timestamp": datetime.now().isoformat()}
        )

    def get_factor_performance_summary(self) -> Dict[str, Any]:
        """获取因子性能摘要"""
        if not self.factor_stats:
            return {}

        total_stocks = sum(stats["stock_count"] for stats in self.factor_stats.values())
        total_time = sum(stats["calculation_time"] for stats in self.factor_stats.values())
        total_data = sum(stats["data_size_mb"] for stats in self.factor_stats.values())

        return {
            "total_factors": len(self.factor_stats),
            "total_stocks_processed": total_stocks,
            "total_calculation_time": total_time,
            "total_data_processed_mb": total_data,
            "avg_throughput_stocks_per_sec": total_stocks / total_time if total_time > 0 else 0,
            "avg_throughput_mb_per_sec": total_data / total_time if total_time > 0 else 0,
            "factor_details": dict(self.factor_stats),
        }

    def get_slowest_factors(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """获取最慢的因子"""
        sorted_factors = sorted(self.factor_stats.items(), key=lambda x: x[1]["calculation_time"], reverse=True)

        return [{"factor_name": name, **stats} for name, stats in sorted_factors[:top_n]]

    def get_stock_processing_summary(self) -> Dict[str, Any]:
        """获取股票处理摘要"""
        if not self.stock_processing_stats:
            return {}

        all_times = []
        for stock_stats in self.stock_processing_stats.values():
            all_times.extend([stat["processing_time"] for stat in stock_stats])

        if not all_times:
            return {}

        return {
            "total_stocks": len(self.stock_processing_stats),
            "total_processing_records": len(all_times),
            "avg_processing_time": sum(all_times) / len(all_times),
            "min_processing_time": min(all_times),
            "max_processing_time": max(all_times),
            "total_processing_time": sum(all_times),
        }


# 全局性能监控器实例
_global_monitor = None


def get_global_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def cleanup_global_monitor():
    """清理全局性能监控器"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.cleanup()
        _global_monitor = None


# 便捷装饰器
def monitor_performance(operation_name: str = None):
    """便捷的性能监控装饰器"""
    return performance_monitor(get_global_monitor(), operation_name)


# 示例用法
if __name__ == "__main__":
    import random

    # 创建性能监控器
    monitor = FactorPerformanceMonitor()

    # 使用上下文管理器监控操作
    with monitor.monitor_operation("测试计算") as session_id:
        # 模拟计算
        time.sleep(1)

        # 添加自定义指标
        monitor.add_custom_metric(session_id, "processed_records", 1000)
        monitor.add_custom_metric(session_id, "cache_hits", 850)

    # 记录因子计算统计
    monitor.record_factor_calculation("RSI", 100, 2.5, 10.5)
    monitor.record_factor_calculation("MACD", 100, 3.2, 15.2)

    # 使用装饰器
    @monitor_performance("装饰器测试")
    def test_function():
        """方法描述"""
        return "完成"

    result = test_function()

    # 获取统计信息
    print("所有统计信息:")
    stats = monitor.get_all_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    print("\n因子性能摘要:")
    factor_summary = monitor.get_factor_performance_summary()
    print(json.dumps(factor_summary, indent=2, ensure_ascii=False))

    # 导出统计信息
    monitor.export_statistics("performance_stats.json")

    # 清理
    monitor.cleanup()
