#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型解释器性能监控系统

监控API服务的性能指标和资源使用情况
"""

import time
import psutil
import threading
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
import os
from collections import deque

# 移除PerformanceManager导入，因为没有使用
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

class ExplainerPerformanceMonitor:
    """模型解释器性能监控器"""
    
    def __init__(self, history_size: int = 1000):
        """
        初始化性能监控器
        
        Args:
            history_size: 历史数据保留数量
        """
        # 移除未使用的performance_monitor属性
        self.history_size = history_size
        
        # 性能指标
        self.metrics = {
            'request_count': 0,
            'success_count': 0,
            'error_count': 0,
            'total_response_time': 0.0,
            'average_response_time': 0.0,
            'min_response_time': float('inf'),
            'max_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_requests': 0,
            'single_requests': 0
        }
        
        # 历史数据
        self.response_times = deque(maxlen=history_size)
        self.request_history = deque(maxlen=history_size)
        self.system_metrics_history = deque(maxlen=history_size)
        
        # 资源监控
        self.resource_monitor = ResourceMonitor()
        
        # 启动资源监控线程
        self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Explainer性能监控器初始化完成")
    
    def record_request(self, start_time: float, success: bool = True, 
                      is_batch: bool = False, cache_hit: bool = False):
        """
        记录请求性能
        
        Args:
            start_time: 请求开始时间戳
            success: 是否成功
            is_batch: 是否为批量请求
            cache_hit: 是否缓存命中
        """
        try:
            response_time = time.time() - start_time
            
            # 更新基本指标
            self.metrics['request_count'] += 1
            if success:
                self.metrics['success_count'] += 1
            else:
                self.metrics['error_count'] += 1
            
            if is_batch:
                self.metrics['batch_requests'] += 1
            else:
                self.metrics['single_requests'] += 1
            
            if cache_hit:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1
            
            # 更新响应时间统计
            self.metrics['total_response_time'] += response_time
            self.metrics['average_response_time'] = (
                self.metrics['total_response_time'] / self.metrics['request_count']
            )
            
            # 更新最值
            self.metrics['min_response_time'] = min(
                self.metrics['min_response_time'], response_time
            )
            self.metrics['max_response_time'] = max(
                self.metrics['max_response_time'], response_time
            )
            
            # 记录历史数据
            self.response_times.append({
                'timestamp': datetime.now().isoformat(),
                'response_time': response_time,
                'success': success,
                'is_batch': is_batch,
                'cache_hit': cache_hit
            })
            
            # 记录请求历史
            self.request_history.append({
                'timestamp': datetime.now().isoformat(),
                'response_time': response_time,
                'success': success
            })
            
            logger.debug(f"请求记录完成: 响应时间={response_time:.3f}秒, 成功={success}")
            
        except Exception as e:
            logger.error(f"记录请求性能失败: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        获取系统资源使用情况
        
        Returns:
            系统资源指标
        """
        try:
            # 获取系统指标
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available / (1024**3),  # GB
                'memory_used': memory.used / (1024**3),  # GB
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free': disk.free / (1024**3),  # GB
                'timestamp': datetime.now().isoformat()
            }
            
            # 记录历史数据
            self.system_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {}
    
    def _monitor_resources(self):
        """后台资源监控线程"""
        while True:
            try:
                # 每5秒获取一次系统指标
                time.sleep(5)
                self.get_system_metrics()
            except Exception as e:
                logger.error(f"资源监控线程错误: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计指标
        """
        try:
            # 计算成功率
            success_rate = 0.0
            if self.metrics['request_count'] > 0:
                success_rate = (
                    self.metrics['success_count'] / self.metrics['request_count'] * 100
                )
            
            # 计算缓存命中率
            cache_hit_rate = 0.0
            total_cache_ops = self.metrics['cache_hits'] + self.metrics['cache_misses']
            if total_cache_ops > 0:
                cache_hit_rate = (self.metrics['cache_hits'] / total_cache_ops * 100)
            
            # 计算最近的响应时间统计
            recent_times = [r['response_time'] for r in list(self.response_times)[-100:]]
            recent_avg_time = 0.0
            if recent_times:
                recent_avg_time = sum(recent_times) / len(recent_times)
            
            stats = {
                **self.metrics,
                'success_rate': success_rate,
                'cache_hit_rate': cache_hit_rate,
                'recent_average_response_time': recent_avg_time,
                'uptime': self._get_uptime()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取性能统计失败: {e}")
            return {}
    
    def _get_uptime(self) -> str:
        """获取运行时间"""
        # 简化实现，实际项目中可以从应用启动时间计算
        return "0 days, 0 hours, 0 minutes"
    
    def get_recent_performance(self, minutes: int = 5) -> Dict[str, Any]:
        """
        获取最近性能数据
        
        Args:
            minutes: 时间窗口（分钟）
            
        Returns:
            最近性能数据
        """
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # 过滤最近的请求
            recent_requests = [
                r for r in self.request_history
                if datetime.fromisoformat(r['timestamp']) > cutoff_time
            ]
            
            if not recent_requests:
                return {'request_count': 0, 'average_response_time': 0, 'success_rate': 0}
            
            # 计算统计信息
            response_times = [r['response_time'] for r in recent_requests]
            success_count = sum(1 for r in recent_requests if r['success'])
            
            return {
                'request_count': len(recent_requests),
                'average_response_time': sum(response_times) / len(response_times),
                'success_rate': (success_count / len(recent_requests)) * 100,
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            }
            
        except Exception as e:
            logger.error(f"获取最近性能数据失败: {e}")
            return {}
    
    def reset_metrics(self):
        """重置性能指标"""
        self.metrics = {
            'request_count': 0,
            'success_count': 0,
            'error_count': 0,
            'total_response_time': 0.0,
            'average_response_time': 0.0,
            'min_response_time': float('inf'),
            'max_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_requests': 0,
            'single_requests': 0
        }
        self.response_times.clear()
        self.request_history.clear()
        self.system_metrics_history.clear()
        logger.info("性能指标已重置")
    
    def export_metrics(self, filepath: str = None) -> str:
        """
        导出性能指标
        
        Args:
            filepath: 导出文件路径
            
        Returns:
            导出文件路径
        """
        try:
            export_data = {
                'metrics': self.metrics,
                'performance_stats': self.get_performance_stats(),
                'system_metrics_history': list(self.system_metrics_history),
                'response_times': list(self.response_times),
                'export_time': datetime.now().isoformat()
            }
            
            if filepath is None:
                filepath = f"performance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"性能指标已导出到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"导出性能指标失败: {e}")
            raise

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        """初始化资源监控器"""
        self.process = psutil.Process()
        logger.info("资源监控器初始化完成")
    
    def get_process_metrics(self) -> Dict[str, Any]:
        """
        获取进程资源使用情况
        
        Returns:
            进程资源指标
        """
        try:
            with self.process.oneshot():
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                num_threads = self.process.num_threads()
                num_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
                
            return {
                'cpu_percent': cpu_percent,
                'memory_rss': memory_info.rss / (1024**2),  # MB
                'memory_vms': memory_info.vms / (1024**2),  # MB
                'memory_percent': memory_percent,
                'num_threads': num_threads,
                'num_fds': num_fds,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取进程指标失败: {e}")
            return {}
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """
        获取GPU使用情况（如果可用）
        
        Returns:
            GPU指标
        """
        try:
            # 尝试获取GPU信息
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_metrics = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_metrics.append({
                    'gpu_id': i,
                    'memory_used': info.used / (1024**2),  # MB
                    'memory_total': info.total / (1024**2),  # MB
                    'memory_percent': (info.used / info.total) * 100,
                    'gpu_utilization': utilization.gpu,
                    'memory_utilization': utilization.memory
                })
            
            return {'gpus': gpu_metrics}
            
        except Exception as e:
            logger.debug(f"获取GPU指标失败（可能未安装GPU）: {e}")
            return {'gpus': []}

# 全局监控实例
explainer_monitor = ExplainerPerformanceMonitor()

def get_monitor() -> ExplainerPerformanceMonitor:
    """获取全局监控实例"""
    return explainer_monitor
