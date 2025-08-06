"""
性能监控工具 - 用于监控测试执行过程中的性能指标
"""
import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0


@dataclass
class PerformanceReport:
    """性能报告数据类"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    metrics: List[PerformanceMetrics] = field(default_factory=list)
    peak_cpu_percent: float = 0.0
    peak_memory_percent: float = 0.0
    peak_memory_used_mb: float = 0.0
    average_cpu_percent: float = 0.0
    average_memory_percent: float = 0.0
    
    def __post_init__(self):
        """计算统计指标"""
        if self.metrics:
            self.peak_cpu_percent = max(m.cpu_percent for m in self.metrics)
            self.peak_memory_percent = max(m.memory_percent for m in self.metrics)
            self.peak_memory_used_mb = max(m.memory_used_mb for m in self.metrics)
            self.average_cpu_percent = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)
            self.average_memory_percent = sum(m.memory_percent for m in self.metrics) / len(self.metrics)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics: List[PerformanceMetrics] = []
        self.logger = logging.getLogger(__name__)
        
        # 初始化基准值
        self._initial_disk_io = psutil.disk_io_counters()
        self._initial_network_io = psutil.net_io_counters()
    
    def start_monitoring(self, test_name: str = "unknown"):
        """开始性能监控"""
        if self.is_monitoring:
            self.logger.warning("性能监控已在运行中")
            return
        
        self.test_name = test_name
        self.start_time = datetime.now()
        self.metrics.clear()
        self.is_monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"开始监控测试 '{test_name}' 的性能")
    
    def stop_monitoring(self) -> PerformanceReport:
        """停止性能监控并返回报告"""
        if not self.is_monitoring:
            self.logger.warning("性能监控未在运行")
            return None
        
        self.is_monitoring = False
        self.end_time = datetime.now()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        report = PerformanceReport(
            test_name=getattr(self, 'test_name', 'unknown'),
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=duration,
            metrics=self.metrics.copy()
        )
        
        self.logger.info(f"性能监控完成，持续时间: {duration:.2f}秒")
        return report
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"性能监控异常: {e}")
                break
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        # CPU和内存
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # 磁盘I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = 0.0
        disk_write_mb = 0.0
        
        if disk_io and self._initial_disk_io:
            disk_read_mb = (disk_io.read_bytes - self._initial_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (disk_io.write_bytes - self._initial_disk_io.write_bytes) / (1024 * 1024)
        
        # 网络I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = 0.0
        network_recv_mb = 0.0
        
        if network_io and self._initial_network_io:
            network_sent_mb = (network_io.bytes_sent - self._initial_network_io.bytes_sent) / (1024 * 1024)
            network_recv_mb = (network_io.bytes_recv - self._initial_network_io.bytes_recv) / (1024 * 1024)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )


def performance_monitor(sampling_interval: float = 1.0):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor(sampling_interval)
            
            # 获取函数名作为测试名
            test_name = getattr(func, '__name__', 'unknown_function')
            
            monitor.start_monitoring(test_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                report = monitor.stop_monitoring()
                
                # 如果结果是字典，添加性能报告
                if isinstance(result, dict):
                    result['performance_report'] = {
                        'duration_seconds': report.duration_seconds,
                        'peak_cpu_percent': report.peak_cpu_percent,
                        'peak_memory_percent': report.peak_memory_percent,
                        'peak_memory_used_mb': report.peak_memory_used_mb,
                        'average_cpu_percent': report.average_cpu_percent,
                        'average_memory_percent': report.average_memory_percent
                    }
                
                # 记录性能摘要
                logger = logging.getLogger(__name__)
                logger.info(f"性能摘要 - {test_name}:")
                logger.info(f"  执行时间: {report.duration_seconds:.2f}秒")
                logger.info(f"  峰值CPU: {report.peak_cpu_percent:.1f}%")
                logger.info(f"  峰值内存: {report.peak_memory_percent:.1f}%")
                logger.info(f"  峰值内存使用: {report.peak_memory_used_mb:.1f}MB")
        
        return wrapper
    return decorator


class PerformanceThresholdChecker:
    """性能阈值检查器"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__)
    
    def check_thresholds(self, report: PerformanceReport) -> Dict[str, Any]:
        """检查性能阈值"""
        violations = []
        warnings = []
        
        # 检查执行时间
        max_duration = self.thresholds.get('max_duration_seconds', float('inf'))
        if report.duration_seconds > max_duration:
            violations.append(f"执行时间超限: {report.duration_seconds:.2f}s > {max_duration}s")
        
        # 检查CPU使用率
        max_cpu = self.thresholds.get('max_cpu_percent', 100.0)
        if report.peak_cpu_percent > max_cpu:
            violations.append(f"CPU使用率超限: {report.peak_cpu_percent:.1f}% > {max_cpu}%")
        
        # 检查内存使用率
        max_memory = self.thresholds.get('max_memory_percent', 90.0)
        if report.peak_memory_percent > max_memory:
            violations.append(f"内存使用率超限: {report.peak_memory_percent:.1f}% > {max_memory}%")
        
        # 检查内存使用量
        max_memory_mb = self.thresholds.get('max_memory_mb', float('inf'))
        if report.peak_memory_used_mb > max_memory_mb:
            violations.append(f"内存使用量超限: {report.peak_memory_used_mb:.1f}MB > {max_memory_mb}MB")
        
        # 生成警告
        warn_cpu = self.thresholds.get('warn_cpu_percent', 80.0)
        if report.average_cpu_percent > warn_cpu:
            warnings.append(f"平均CPU使用率较高: {report.average_cpu_percent:.1f}%")
        
        warn_memory = self.thresholds.get('warn_memory_percent', 70.0)
        if report.average_memory_percent > warn_memory:
            warnings.append(f"平均内存使用率较高: {report.average_memory_percent:.1f}%")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'score': max(0, 100 - len(violations) * 25 - len(warnings) * 5)
        }