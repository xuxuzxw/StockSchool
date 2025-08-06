"""
性能测试工具类
"""
import time
import psutil
import threading
import numpy as np
from typing import Dict, Any, List, Callable
from datetime import datetime
from .exceptions import SystemMetricsError


class SystemMonitor:
    """系统监控工具"""
    
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """获取系统指标"""
        try:
            memory = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            network_io = psutil.net_io_counters()
            
            return {
                'memory_usage_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'memory_usage_percent': memory.percent,
                'cpu_usage_percent': cpu_usage,
                'cpu_count': psutil.cpu_count(),
                'disk_usage_percent': (disk.used / disk.total) * 100,
                'disk_total_gb': disk.total / (1024**3),
                'network_bytes_sent': network_io.bytes_sent,
                'network_bytes_recv': network_io.bytes_recv,
                'process_count': len(psutil.pids()),
                'timestamp': datetime.now().isoformat()
            }
        except psutil.Error as e:
            raise SystemMetricsError(f"系统指标获取失败: {e}")
    
    @staticmethod
    def monitor_resources(
        duration: int, 
        interval: int = 1,
        callback: Callable[[Dict[str, Any]], None] = None
    ) -> List[Dict[str, Any]]:
        """监控系统资源使用情况"""
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                metrics = SystemMonitor.get_system_metrics()
                metrics['elapsed_time'] = time.time() - start_time
                samples.append(metrics)
                
                if callback:
                    callback(metrics)
                
                time.sleep(interval)
                
            except SystemMetricsError as e:
                # 记录错误但继续监控
                print(f"监控采样失败: {e}")
                continue
        
        return samples


class LoadGenerator:
    """负载生成器"""
    
    @staticmethod
    def generate_cpu_load(duration: int, intensity: float = 0.8):
        """生成CPU负载"""
        end_time = time.time() + duration
        work_time = intensity
        rest_time = 1 - intensity
        
        while time.time() < end_time:
            # 工作阶段
            work_end = time.time() + work_time
            while time.time() < work_end:
                # CPU密集型计算
                for _ in range(10000):
                    _ = sum(i * i for i in range(100))
            
            # 休息阶段
            time.sleep(rest_time)
    
    @staticmethod
    def generate_memory_load(duration: int, block_size_mb: int = 1):
        """生成内存负载"""
        memory_blocks = []
        end_time = time.time() + duration
        
        try:
            while time.time() < end_time:
                # 分配内存块
                block = bytearray(block_size_mb * 1024 * 1024)
                memory_blocks.append(block)
                time.sleep(0.1)
                
                # 限制内存使用，避免系统崩溃
                if len(memory_blocks) > 100:
                    memory_blocks.pop(0)
        finally:
            # 清理内存
            del memory_blocks


class PerformanceAnalyzer:
    """性能分析器"""
    
    @staticmethod
    def analyze_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析性能样本数据"""
        if not samples:
            return {}
        
        # 提取数值指标
        memory_usage = [s['memory_usage_gb'] for s in samples]
        cpu_usage = [s['cpu_usage_percent'] for s in samples]
        
        return {
            'sample_count': len(samples),
            'duration_seconds': samples[-1]['elapsed_time'] if samples else 0,
            'memory_stats': {
                'min': min(memory_usage),
                'max': max(memory_usage),
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'p95': np.percentile(memory_usage, 95)
            },
            'cpu_stats': {
                'min': min(cpu_usage),
                'max': max(cpu_usage),
                'mean': np.mean(cpu_usage),
                'std': np.std(cpu_usage),
                'p95': np.percentile(cpu_usage, 95)
            }
        }
    
    @staticmethod
    def check_thresholds(
        metrics: Dict[str, Any], 
        thresholds: Dict[str, float]
    ) -> List[str]:
        """检查性能阈值"""
        violations = []
        
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                actual_value = metrics[metric_name]
                if actual_value > threshold:
                    violations.append(
                        f"{metric_name}: {actual_value:.2f} > {threshold}"
                    )
        
        return violations


class ThreadPoolExecutor:
    """线程池执行器"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.threads = []
    
    def submit(self, func: Callable, *args, **kwargs):
        """提交任务到线程池"""
        if len(self.threads) >= self.max_workers:
            # 等待一个线程完成
            for thread in self.threads[:]:
                if not thread.is_alive():
                    self.threads.remove(thread)
                    break
        
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        self.threads.append(thread)
    
    def wait_all(self):
        """等待所有线程完成"""
        for thread in self.threads:
            thread.join()
        self.threads.clear()