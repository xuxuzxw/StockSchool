"""
性能测试策略模式实现
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import psutil
import time
import threading
import numpy as np
from datetime import datetime


class PerformanceTestStrategy(ABC):
    """性能测试策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None  # 由子类注入
    
    @abstractmethod
    def execute_test(self) -> Dict[str, Any]:
        """执行性能测试"""
        pass
    
    @abstractmethod
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """验证测试结果"""
        pass


class SystemBaselineStrategy(PerformanceTestStrategy):
    """系统基准性能测试策略"""
    
    def execute_test(self) -> Dict[str, Any]:
        """执行系统基准测试"""
        try:
            return self._measure_system_metrics()
        except psutil.Error as e:
            raise PerformanceTestError(f"系统指标获取失败: {e}")
        except Exception as e:
            raise PerformanceTestError(f"系统基准测试失败: {e}")
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """验证系统基准结果"""
        issues = []
        thresholds = self.config.get('thresholds', {})
        
        if results['memory_usage_gb'] > thresholds.get('memory_limit_gb', 16):
            issues.append(f"内存使用超限: {results['memory_usage_gb']:.2f}GB")
        
        if results['cpu_usage_percent'] > thresholds.get('cpu_usage_limit', 80):
            issues.append(f"CPU使用率超限: {results['cpu_usage_percent']:.1f}%")
        
        return {
            'issues': issues,
            'score': max(0, 100 - len(issues) * 20),
            'passed': len(issues) == 0
        }
    
    def _measure_system_metrics(self) -> Dict[str, Any]:
        """测量系统指标"""
        memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            'memory_usage_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_usage_percent': memory.percent,
            'cpu_usage_percent': cpu_usage,
            'cpu_count': psutil.cpu_count(),
            'disk_usage_percent': (disk.used / disk.total) * 100,
            'timestamp': datetime.now().isoformat()
        }


class LoadTestStrategy(PerformanceTestStrategy):
    """负载测试策略"""
    
    def execute_test(self) -> Dict[str, Any]:
        """执行负载测试"""
        test_config = self.config.get('load_test', {})
        duration = test_config.get('duration_seconds', 60)
        thread_count = test_config.get('thread_count', 4)
        
        try:
            return self._execute_load_test(duration, thread_count)
        except Exception as e:
            raise PerformanceTestError(f"负载测试执行失败: {e}")
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """验证负载测试结果"""
        issues = []
        thresholds = self.config.get('thresholds', {})
        
        if results['max_memory_usage_gb'] > thresholds.get('memory_limit_gb', 16):
            issues.append("负载测试内存使用超限")
        
        if results['avg_cpu_usage_percent'] > thresholds.get('cpu_usage_limit', 80):
            issues.append("负载测试CPU使用率超限")
        
        return {
            'issues': issues,
            'score': max(0, 100 - len(issues) * 25),
            'passed': len(issues) == 0
        }
    
    def _execute_load_test(self, duration: int, thread_count: int) -> Dict[str, Any]:
        """执行负载测试实现"""
        samples = []
        start_time = time.time()
        
        # 创建负载线程
        load_threads = []
        for i in range(thread_count):
            thread = threading.Thread(
                target=self._generate_cpu_load, 
                args=(duration,)
            )
            load_threads.append(thread)
            thread.start()
        
        # 监控系统资源
        sample_interval = self.config.get('load_test', {}).get('sample_interval', 5)
        while time.time() - start_time < duration:
            memory = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=1)
            
            samples.append({
                'timestamp': time.time() - start_time,
                'memory_usage_gb': memory.used / (1024**3),
                'cpu_usage_percent': cpu_usage
            })
            
            time.sleep(sample_interval)
        
        # 等待线程结束
        for thread in load_threads:
            thread.join()
        
        # 分析结果
        if samples:
            return {
                'test_duration_seconds': duration,
                'samples_collected': len(samples),
                'max_memory_usage_gb': max(s['memory_usage_gb'] for s in samples),
                'avg_memory_usage_gb': np.mean([s['memory_usage_gb'] for s in samples]),
                'max_cpu_usage_percent': max(s['cpu_usage_percent'] for s in samples),
                'avg_cpu_usage_percent': np.mean([s['cpu_usage_percent'] for s in samples]),
                'thread_count': thread_count,
                'test_completed': True
            }
        else:
            raise PerformanceTestError("负载测试未收集到有效样本")
    
    def _generate_cpu_load(self, duration: int):
        """生成CPU负载"""
        end_time = time.time() + duration
        while time.time() < end_time:
            # CPU密集型计算
            for _ in range(10000):
                _ = sum(i * i for i in range(100))
            time.sleep(0.01)  # 短暂休息避免100%占用


class PerformanceTestError(Exception):
    """性能测试异常"""
    pass