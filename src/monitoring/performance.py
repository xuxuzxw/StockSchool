import os
import sys
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path
import functools
import traceback
from contextlib import contextmanager
from loguru import logger
import pandas as pd
import numpy as np
from src.utils.config_loader import config

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class FunctionProfile:
    """函数性能分析"""
    function_name: str
    module_name: str
    call_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_called: datetime
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['last_called'] = self.last_called.isoformat()
        return data

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, collection_interval: int = None):
        """初始化系统监控器"""
        if collection_interval is None:
            collection_interval = config.get('monitoring_params.collection_interval', 60)
        self.collection_interval = collection_interval
        max_history = config.get('monitoring_params.metrics_max_history', 1000)
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.running = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def start(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"系统监控异常: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        timestamp = datetime.now()
        
        # CPU指标
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        self._record_metric("cpu_usage_percent", cpu_percent, "%", timestamp)
        self._record_metric("cpu_count", cpu_count, "cores", timestamp)
        if cpu_freq:
            self._record_metric("cpu_frequency_mhz", cpu_freq.current, "MHz", timestamp)
        
        # 内存指标
        memory = psutil.virtual_memory()
        self._record_metric("memory_total_gb", memory.total / (1024**3), "GB", timestamp)
        self._record_metric("memory_used_gb", memory.used / (1024**3), "GB", timestamp)
        self._record_metric("memory_available_gb", memory.available / (1024**3), "GB", timestamp)
        self._record_metric("memory_usage_percent", memory.percent, "%", timestamp)
        
        # 磁盘指标
        disk_usage = psutil.disk_usage('/')
        self._record_metric("disk_total_gb", disk_usage.total / (1024**3), "GB", timestamp)
        self._record_metric("disk_used_gb", disk_usage.used / (1024**3), "GB", timestamp)
        self._record_metric("disk_free_gb", disk_usage.free / (1024**3), "GB", timestamp)
        self._record_metric("disk_usage_percent", 
                          (disk_usage.used / disk_usage.total) * 100, "%", timestamp)
        
        # 网络指标
        try:
            net_io = psutil.net_io_counters()
            self._record_metric("network_bytes_sent", net_io.bytes_sent, "bytes", timestamp)
            self._record_metric("network_bytes_recv", net_io.bytes_recv, "bytes", timestamp)
            self._record_metric("network_packets_sent", net_io.packets_sent, "packets", timestamp)
            self._record_metric("network_packets_recv", net_io.packets_recv, "packets", timestamp)
        except Exception as e:
            logger.warning(f"网络指标收集失败: {e}")
        
        # 进程指标
        try:
            process = psutil.Process()
            self._record_metric("process_cpu_percent", process.cpu_percent(), "%", timestamp)
            self._record_metric("process_memory_mb", 
                              process.memory_info().rss / (1024**2), "MB", timestamp)
            self._record_metric("process_threads", process.num_threads(), "count", timestamp)
            # Windows系统没有num_fds方法
            if hasattr(process, 'num_fds'):
                self._record_metric("process_fds", process.num_fds(), "count", timestamp)
        except Exception as e:
            logger.warning(f"进程指标收集失败: {e}")
    
    def _record_metric(self, name: str, value: float, unit: str, timestamp: datetime):
        """记录指标"""
        metric = PerformanceMetric(name, value, unit, timestamp)
        
        with self.lock:
            self.metrics_history[name].append(metric)
    
    def get_metrics(self, metric_name: str = None, 
                   start_time: datetime = None, 
                   end_time: datetime = None) -> List[PerformanceMetric]:
        """获取指标"""
        with self.lock:
            if metric_name:
                metrics = list(self.metrics_history[metric_name])
            else:
                metrics = []
                for metric_list in self.metrics_history.values():
                    metrics.extend(metric_list)
        
        # 时间过滤
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered_metrics.append(metric)
            metrics = filtered_metrics
        
        return sorted(metrics, key=lambda x: x.timestamp)
    
    def get_metric_summary(self, metric_name: str, 
                          window_minutes: int = None) -> Dict[str, float]:
        """获取指标摘要"""
        if window_minutes is None:
            window_minutes = config.get('monitoring_params.metric_window_minutes', 60)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=window_minutes)
        
        metrics = self.get_metrics(metric_name, start_time, end_time)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'latest': values[-1] if values else 0
        }

class FunctionProfiler:
    """函数性能分析器"""
    
    def __init__(self):
        """初始化函数分析器"""
        self.profiles = {}
        self.lock = threading.Lock()
        self.enabled = True
    
    def enable(self):
        """启用分析器"""
        self.enabled = True
    
    def disable(self):
        """禁用分析器"""
        self.enabled = False
    
    def profile(self, func: Callable = None, *, name: str = None):
        """性能分析装饰器"""
        def decorator(f):
            function_name = name or f.__name__
            module_name = f.__module__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return f(*args, **kwargs)
                
                start_time = time.time()
                error_occurred = False
                
                try:
                    result = f(*args, **kwargs)
                    return result
                except Exception as e:
                    error_occurred = True
                    raise
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    self._record_execution(function_name, module_name, 
                                         execution_time, error_occurred)
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _record_execution(self, function_name: str, module_name: str, 
                         execution_time: float, error_occurred: bool):
        """记录执行信息"""
        with self.lock:
            key = f"{module_name}.{function_name}"
            
            if key not in self.profiles:
                self.profiles[key] = FunctionProfile(
                    function_name=function_name,
                    module_name=module_name,
                    call_count=0,
                    total_time=0.0,
                    avg_time=0.0,
                    min_time=float('inf'),
                    max_time=0.0,
                    last_called=datetime.now(),
                    error_count=0
                )
            
            profile = self.profiles[key]
            profile.call_count += 1
            profile.total_time += execution_time
            profile.avg_time = profile.total_time / profile.call_count
            profile.min_time = min(profile.min_time, execution_time)
            profile.max_time = max(profile.max_time, execution_time)
            profile.last_called = datetime.now()
            
            if error_occurred:
                profile.error_count += 1
    
    def get_profiles(self, sort_by: str = 'total_time') -> List[FunctionProfile]:
        """获取性能分析结果"""
        with self.lock:
            profiles = list(self.profiles.values())
        
        # 排序
        if sort_by == 'total_time':
            profiles.sort(key=lambda x: x.total_time, reverse=True)
        elif sort_by == 'avg_time':
            profiles.sort(key=lambda x: x.avg_time, reverse=True)
        elif sort_by == 'call_count':
            profiles.sort(key=lambda x: x.call_count, reverse=True)
        elif sort_by == 'error_count':
            profiles.sort(key=lambda x: x.error_count, reverse=True)
        
        return profiles
    
    def get_top_functions(self, limit: int = 10, sort_by: str = 'total_time') -> List[FunctionProfile]:
        """获取性能最差的函数"""
        profiles = self.get_profiles(sort_by)
        return profiles[:limit]
    
    def reset(self):
        """重置统计信息"""
        with self.lock:
            self.profiles.clear()
        logger.info("函数性能统计已重置")
    
    def generate_report(self) -> str:
        """生成性能报告"""
        profiles = self.get_profiles('total_time')
        
        if not profiles:
            return "暂无性能数据"
        
        report = ["\n=== 函数性能分析报告 ==="]
        report.append(f"总函数数: {len(profiles)}")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n排名 | 函数名 | 调用次数 | 总时间(s) | 平均时间(ms) | 最大时间(ms) | 错误次数")
        separator_length = config.get('monitoring_params.report_separator_length', 100)
        report.append("-" * separator_length)
        
        max_profiles = config.get('monitoring_params.slow_query_limit', 20)
        for i, profile in enumerate(profiles[:max_profiles], 1):
            report.append(
                f"{i:4d} | {profile.function_name[:20]:20s} | "
                f"{profile.call_count:8d} | {profile.total_time:9.3f} | "
                f"{profile.avg_time*1000:11.2f} | {profile.max_time*1000:11.2f} | "
                f"{profile.error_count:8d}"
            )
        
        return "\n".join(report)

class DatabaseProfiler:
    """数据库性能分析器"""
    
    def __init__(self):
        """初始化数据库分析器"""
        self.query_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0,
            'last_executed': None
        })
        max_queries = config.get('monitoring_params.metric_retention', 100)
        self.slow_queries = deque(maxlen=max_queries)
        self.lock = threading.Lock()
        self.slow_query_threshold = 1.0  # 慢查询阈值（秒）
    
    @contextmanager
    def profile_query(self, query: str, params: tuple = None):
        """查询性能分析上下文管理器"""
        # 简化查询字符串用作键
        query_key = self._normalize_query(query)
        start_time = time.time()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            raise
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            self._record_query(query_key, query, execution_time, error_occurred)
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询字符串"""
        # 移除多余空格和换行
        normalized = ' '.join(query.split())
        
        # 截取前100个字符作为键
        max_query_length = config.get('monitoring_params.metric_retention', 100)
        if len(normalized) > max_query_length:
            normalized = normalized[:max_query_length] + "..."
        
        return normalized
    
    def _record_query(self, query_key: str, full_query: str, 
                     execution_time: float, error_occurred: bool):
        """记录查询统计"""
        with self.lock:
            stats = self.query_stats[query_key]
            stats['count'] += 1
            stats['total_time'] += execution_time
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['min_time'] = min(stats['min_time'], execution_time)
            stats['max_time'] = max(stats['max_time'], execution_time)
            stats['last_executed'] = datetime.now()
            
            if error_occurred:
                stats['error_count'] += 1
            
            # 记录慢查询
            if execution_time >= self.slow_query_threshold:
                self.slow_queries.append({
                    'query': full_query,
                    'execution_time': execution_time,
                    'timestamp': datetime.now(),
                    'error': error_occurred
                })
    
    def get_query_stats(self, sort_by: str = 'total_time') -> List[Dict[str, Any]]:
        """获取查询统计"""
        with self.lock:
            stats_list = []
            for query_key, stats in self.query_stats.items():
                stats_copy = stats.copy()
                stats_copy['query'] = query_key
                if stats_copy['last_executed']:
                    stats_copy['last_executed'] = stats_copy['last_executed'].isoformat()
                stats_list.append(stats_copy)
        
        # 排序
        if sort_by in ['total_time', 'avg_time', 'max_time', 'count', 'error_count']:
            stats_list.sort(key=lambda x: x[sort_by], reverse=True)
        
        return stats_list
    
    def get_slow_queries(self, limit: int = None) -> List[Dict[str, Any]]:
        """获取慢查询"""
        if limit is None:
            limit = config.get('monitoring_params.slow_query_limit', 20)
        with self.lock:
            slow_queries = list(self.slow_queries)
        
        # 按执行时间排序
        slow_queries.sort(key=lambda x: x['execution_time'], reverse=True)
        
        # 转换时间戳
        for query in slow_queries[:limit]:
            if hasattr(query['timestamp'], 'isoformat'):
                query['timestamp'] = query['timestamp'].isoformat()
        
        return slow_queries[:limit]
    
    def set_slow_query_threshold(self, threshold: float):
        """设置慢查询阈值"""
        self.slow_query_threshold = threshold
        logger.info(f"慢查询阈值设置为: {threshold}秒")
    
    def reset(self):
        """重置统计信息"""
        with self.lock:
            self.query_stats.clear()
            self.slow_queries.clear()
        logger.info("数据库性能统计已重置")

class PerformanceStorage:
    """性能数据存储"""
    
    def __init__(self, db_path: str = "performance.db"):
        """初始化性能存储"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 系统指标表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            # 函数性能表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS function_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    function_name TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    call_count INTEGER NOT NULL,
                    total_time REAL NOT NULL,
                    avg_time REAL NOT NULL,
                    min_time REAL NOT NULL,
                    max_time REAL NOT NULL,
                    error_count INTEGER NOT NULL,
                    last_called TIMESTAMP NOT NULL,
                    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 数据库查询统计表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    error_occurred BOOLEAN NOT NULL
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON system_metrics(name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_profiles_function ON function_profiles(function_name, module_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_time ON query_stats(timestamp)")
    
    def save_metrics(self, metrics: List[PerformanceMetric]):
        """保存系统指标"""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT INTO system_metrics (name, value, unit, timestamp, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.name, metric.value, metric.unit, metric.timestamp.isoformat(),
                    json.dumps(metric.tags), json.dumps(metric.metadata)
                ))
    
    def save_function_profiles(self, profiles: List[FunctionProfile]):
        """保存函数性能数据"""
        with sqlite3.connect(self.db_path) as conn:
            for profile in profiles:
                conn.execute("""
                    INSERT INTO function_profiles 
                    (function_name, module_name, call_count, total_time, avg_time,
                     min_time, max_time, error_count, last_called)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.function_name, profile.module_name, profile.call_count,
                    profile.total_time, profile.avg_time, profile.min_time,
                    profile.max_time, profile.error_count, profile.last_called.isoformat()
                ))
    
    def cleanup_old_data(self, days: int = None):
        """清理旧数据"""
        if days is None:
            days = config.get('monitoring_params.data_retention_days', 30)
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # 清理旧的系统指标
            result = conn.execute("""
                DELETE FROM system_metrics WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            metrics_deleted = result.rowcount
            
            # 清理旧的查询统计
            result = conn.execute("""
                DELETE FROM query_stats WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            queries_deleted = result.rowcount
            
            logger.info(f"清理了 {metrics_deleted} 条系统指标和 {queries_deleted} 条查询记录")

class PerformanceManager:
    """性能管理器"""
    
    def __init__(self, storage_path: str = "performance.db"):
        """初始化性能管理器"""
        self.system_monitor = SystemMonitor()
        self.function_profiler = FunctionProfiler()
        self.db_profiler = DatabaseProfiler()
        self.storage = PerformanceStorage(storage_path)
        
        # 定期保存数据的线程
        self.save_interval = config.get('monitoring_params.save_interval', 300)  # 5分钟
        self.save_thread = None
        self.running = False
    
    def start(self):
        """启动性能监控"""
        self.system_monitor.start()
        self.running = True
        
        # 启动定期保存线程
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()
        
        logger.info("性能监控管理器已启动")
    
    def stop(self):
        """停止性能监控"""
        self.system_monitor.stop()
        self.running = False
        
        if self.save_thread:
            join_timeout = config.get('monitoring_params.thread_join_timeout', 5)
            self.save_thread.join(timeout=join_timeout)
        
        # 最后保存一次数据
        self._save_data()
        
        logger.info("性能监控管理器已停止")
    
    def _save_loop(self):
        """定期保存数据循环"""
        while self.running:
            try:
                time.sleep(self.save_interval)
                if self.running:
                    self._save_data()
            except Exception as e:
                logger.error(f"保存性能数据异常: {e}")
    
    def _save_data(self):
        """保存性能数据"""
        try:
            # 保存系统指标
            metrics = self.system_monitor.get_metrics()
            if metrics:
                retention_count = config.get('monitoring_params.metric_retention', 100)
                self.storage.save_metrics(metrics[-retention_count:])  # 只保存最近N条
            
            # 保存函数性能数据
            profiles = self.function_profiler.get_profiles()
            if profiles:
                self.storage.save_function_profiles(profiles)
            
            logger.debug("性能数据已保存")
        
        except Exception as e:
            logger.error(f"保存性能数据失败: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'issues': []
        }
        
        # 检查CPU使用率
        cpu_summary = self.system_monitor.get_metric_summary('cpu_usage_percent', 10)
        if cpu_summary and cpu_summary.get('latest', 0) > 80:
            health['issues'].append('CPU使用率过高')
            health['status'] = 'warning'
        
        # 检查内存使用率
        memory_summary = self.system_monitor.get_metric_summary('memory_usage_percent', 10)
        if memory_summary and memory_summary.get('latest', 0) > 85:
            health['issues'].append('内存使用率过高')
            health['status'] = 'critical' if health['status'] != 'critical' else 'critical'
        
        # 检查磁盘使用率
        disk_summary = self.system_monitor.get_metric_summary('disk_usage_percent', 10)
        if disk_summary and disk_summary.get('latest', 0) > 90:
            health['issues'].append('磁盘空间不足')
            health['status'] = 'critical'
        
        # 检查慢查询
        slow_query_check_limit = config.get('monitoring_params.slow_query_check_limit', 5)
        slow_query_threshold = config.get('monitoring_params.slow_query_count_threshold', 10)
        slow_queries = self.db_profiler.get_slow_queries(slow_query_check_limit)
        if len(slow_queries) > slow_query_threshold:
            health['issues'].append('存在大量慢查询')
            if health['status'] == 'healthy':
                health['status'] = 'warning'
        
        health['cpu'] = cpu_summary
        health['memory'] = memory_summary
        health['disk'] = disk_summary
        health['slow_queries_count'] = len(slow_queries)
        
        return health
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        report = ["\n=== 系统性能报告 ==="]
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 系统健康状态
        health = self.get_system_health()
        report.append(f"\n系统状态: {health['status'].upper()}")
        if health['issues']:
            report.append("发现问题:")
            for issue in health['issues']:
                report.append(f"  - {issue}")
        
        # 系统指标摘要
        report.append("\n=== 系统指标摘要 (最近1小时) ===")
        metrics = ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent']
        for metric in metrics:
            window_minutes = config.get('monitoring_params.evaluation_window', 60)
            summary = self.system_monitor.get_metric_summary(metric, window_minutes)
            if summary:
                report.append(f"{metric}: 当前={summary['latest']:.1f}%, "
                            f"平均={summary['mean']:.1f}%, 最大={summary['max']:.1f}%")
        
        # 函数性能TOP10
        report.append("\n=== 函数性能TOP10 ===")
        top_functions = self.function_profiler.get_top_functions(10)
        if top_functions:
            for i, func in enumerate(top_functions, 1):
                report.append(f"{i:2d}. {func.function_name}: "
                            f"调用{func.call_count}次, 总时间{func.total_time:.3f}s, "
                            f"平均{func.avg_time*1000:.2f}ms")
        else:
            report.append("暂无函数性能数据")
        
        # 慢查询TOP5
        report.append("\n=== 慢查询TOP5 ===")
        slow_queries = self.db_profiler.get_slow_queries(5)
        if slow_queries:
            for i, query in enumerate(slow_queries, 1):
                report.append(f"{i}. 执行时间: {query['execution_time']:.3f}s")
                max_query_display = config.get('monitoring_params.metric_retention', 100)
                report.append(f"   查询: {query['query'][:max_query_display]}...")
        else:
            report.append("暂无慢查询记录")
        
        return "\n".join(report)
    
    def cleanup_old_data(self, days: int = None):
        """清理旧数据"""
        if days is None:
            days = config.get('monitoring_params.data_retention_days', 30)
        self.storage.cleanup_old_data(days)

# 全局性能管理器实例
performance_manager = PerformanceManager()

# 装饰器快捷方式
profile = performance_manager.function_profiler.profile
profile_query = performance_manager.db_profiler.profile_query

if __name__ == '__main__':
    # 测试代码
    print("测试性能监控系统...")
    
    # 启动性能监控
    performance_manager.start()
    
    # 测试函数性能分析
    @profile
    def test_function_1():
        """测试函数1"""
        time.sleep(0.1)
        return "result1"
    
    @profile
    def test_function_2():
        """测试函数2"""
        sleep_interval = 0.05  # 短暂睡眠间隔，可以保持硬编码
        time.sleep(sleep_interval)
        return "result2"
    
    @profile
    def test_function_error():
        """测试错误函数"""
        time.sleep(0.02)
        raise ValueError("测试错误")
    
    # 执行测试函数
    print("\n执行测试函数...")
    for i in range(5):
        test_function_1()
        test_function_2()
        try:
            test_function_error()
        except ValueError:
            pass
    
    # 测试数据库查询分析
    print("\n测试数据库查询分析...")
    db_profiler = performance_manager.db_profiler
    
    # 模拟查询
    queries = [
        "SELECT * FROM stock_daily WHERE ts_code = '000001.SZ'",
        "SELECT COUNT(*) FROM stock_basic",
        "SELECT * FROM stock_daily WHERE trade_date > '20240101'"
    ]
    
    for query in queries:
        with db_profiler.profile_query(query):
            time.sleep(0.1)  # 模拟查询执行时间
    
    # 模拟慢查询
    with db_profiler.profile_query("SELECT * FROM large_table ORDER BY date"):
        time.sleep(1.5)  # 模拟慢查询
    
    # 等待收集一些系统指标
    print("\n等待收集系统指标...")
    time.sleep(3)
    
    # 生成性能报告
    print("\n=== 性能报告 ===")
    report = performance_manager.generate_performance_report()
    print(report)
    
    # 获取系统健康状态
    print("\n=== 系统健康状态 ===")
    health = performance_manager.get_system_health()
    print(f"状态: {health['status']}")
    if health['issues']:
        print("问题:")
        for issue in health['issues']:
            print(f"  - {issue}")
    
    # 查看函数性能统计
    print("\n=== 函数性能统计 ===")
    profiles = performance_manager.function_profiler.get_profiles()
    for profile in profiles:
        print(f"{profile.function_name}: 调用{profile.call_count}次, "
              f"总时间{profile.total_time:.3f}s, 平均{profile.avg_time*1000:.2f}ms, "
              f"错误{profile.error_count}次")
    
    # 查看数据库查询统计
    print("\n=== 数据库查询统计 ===")
    query_stats = db_profiler.get_query_stats()
    for stat in query_stats[:5]:
        print(f"查询: {stat['query'][:50]}...")
        print(f"  执行{stat['count']}次, 总时间{stat['total_time']:.3f}s, "
              f"平均{stat['avg_time']*1000:.2f}ms")
    
    # 查看慢查询
    print("\n=== 慢查询记录 ===")
    slow_queries = db_profiler.get_slow_queries()
    for query in slow_queries:
        print(f"执行时间: {query['execution_time']:.3f}s")
        print(f"查询: {query['query'][:80]}...")
        print()
    
    # 停止性能监控
    performance_manager.stop()
    
    print("\n性能监控系统测试完成!")