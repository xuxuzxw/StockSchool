# -*- coding: utf-8 -*-
"""
AI策略系统优化器

实现系统性能优化、资源管理、缓存策略、数据库优化等功能
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil
import gc
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.exc import SQLAlchemyError
import redis
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    database_connections: int
    cache_hit_rate: float
    response_time: float
    throughput: float
    error_rate: float
    active_users: int

@dataclass
class PerformanceAlert:
    """性能告警"""
    alert_id: str
    alert_type: str  # cpu, memory, disk, database, cache, response_time
    severity: str  # low, medium, high, critical
    metric_name: str
    current_value: float
    threshold_value: float
    description: str
    recommendations: List[str]
    created_at: datetime
    resolved_at: Optional[datetime] = None
    status: str = 'active'  # active, resolved, ignored

@dataclass
class OptimizationTask:
    """优化任务"""
    task_id: str
    task_type: str  # cache_cleanup, db_optimization, index_rebuild, data_archival
    priority: str  # low, medium, high, critical
    description: str
    parameters: Dict[str, Any]
    status: str = 'pending'  # pending, running, completed, failed
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = None

@dataclass
class CacheStrategy:
    """缓存策略"""
    strategy_name: str
    cache_type: str  # redis, memory, database
    ttl_seconds: int
    max_size: int
    eviction_policy: str  # lru, lfu, fifo, random
    compression_enabled: bool
    serialization_format: str  # json, pickle, msgpack
    hit_rate_threshold: float
    is_active: bool = True
    created_at: datetime = None

class SystemOptimizer:
    """系统优化器
    
    提供系统性能监控、优化建议、资源管理等功能
    """
    
    def __init__(self, database_url: str = None, redis_url: str = None):
        from ...utils.db import get_db_manager
        self.engine = get_db_manager().engine
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        self.metadata = MetaData()
        self._optimization_lock = threading.Lock()
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # 创建数据库表
        self._create_tables()
        
        # 初始化默认配置
        self._init_default_configs()
    
    def _create_tables(self):
        """创建数据库表"""
        try:
            # 系统指标表
            system_metrics = Table(
                'system_metrics', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('timestamp', DateTime, nullable=False),
                Column('cpu_usage', Float, nullable=False),
                Column('memory_usage', Float, nullable=False),
                Column('disk_usage', Float, nullable=False),
                Column('network_io', JSON, nullable=True),
                Column('database_connections', Integer, nullable=False),
                Column('cache_hit_rate', Float, nullable=False),
                Column('response_time', Float, nullable=False),
                Column('throughput', Float, nullable=False),
                Column('error_rate', Float, nullable=False),
                Column('active_users', Integer, nullable=False),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # 性能告警表
            performance_alerts = Table(
                'performance_alerts', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('alert_id', String(100), nullable=False, unique=True),
                Column('alert_type', String(50), nullable=False),
                Column('severity', String(20), nullable=False),
                Column('metric_name', String(100), nullable=False),
                Column('current_value', Float, nullable=False),
                Column('threshold_value', Float, nullable=False),
                Column('description', Text, nullable=False),
                Column('recommendations', JSON, nullable=True),
                Column('status', String(20), default='active'),
                Column('resolved_at', DateTime, nullable=True),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # 优化任务表
            optimization_tasks = Table(
                'optimization_tasks', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('task_id', String(100), nullable=False, unique=True),
                Column('task_type', String(50), nullable=False),
                Column('priority', String(20), nullable=False),
                Column('description', Text, nullable=False),
                Column('parameters', JSON, nullable=True),
                Column('status', String(20), default='pending'),
                Column('progress', Float, default=0.0),
                Column('start_time', DateTime, nullable=True),
                Column('end_time', DateTime, nullable=True),
                Column('result', JSON, nullable=True),
                Column('error_message', Text, nullable=True),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # 缓存策略表
            cache_strategies = Table(
                'cache_strategies', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('strategy_name', String(100), nullable=False, unique=True),
                Column('cache_type', String(50), nullable=False),
                Column('ttl_seconds', Integer, nullable=False),
                Column('max_size', Integer, nullable=False),
                Column('eviction_policy', String(20), nullable=False),
                Column('compression_enabled', Boolean, default=False),
                Column('serialization_format', String(20), default='json'),
                Column('hit_rate_threshold', Float, default=0.8),
                Column('is_active', Boolean, default=True),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # 系统配置表
            system_configs = Table(
                'system_configs', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('config_key', String(100), nullable=False, unique=True),
                Column('config_value', JSON, nullable=False),
                Column('description', Text, nullable=True),
                Column('is_active', Boolean, default=True),
                Column('updated_at', DateTime, default=datetime.now),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            # 性能基线表
            performance_baselines = Table(
                'performance_baselines', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('metric_name', String(100), nullable=False),
                Column('baseline_value', Float, nullable=False),
                Column('upper_threshold', Float, nullable=False),
                Column('lower_threshold', Float, nullable=False),
                Column('measurement_period', String(20), nullable=False),  # daily, weekly, monthly
                Column('is_active', Boolean, default=True),
                Column('updated_at', DateTime, default=datetime.now),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            self.metadata.create_all(self.engine)
            logger.info("系统优化器数据库表创建成功")
            
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise
    
    def _init_default_configs(self):
        """初始化默认配置"""
        try:
            default_configs = {
                'monitoring_interval': 60,  # 监控间隔（秒）
                'alert_thresholds': {
                    'cpu_usage': 80.0,
                    'memory_usage': 85.0,
                    'disk_usage': 90.0,
                    'response_time': 2.0,
                    'error_rate': 5.0,
                    'cache_hit_rate': 70.0
                },
                'optimization_schedule': {
                    'cache_cleanup': '0 2 * * *',  # 每天凌晨2点
                    'db_optimization': '0 3 * * 0',  # 每周日凌晨3点
                    'log_archival': '0 1 * * *'  # 每天凌晨1点
                },
                'resource_limits': {
                    'max_database_connections': 100,
                    'max_cache_size_mb': 1024,
                    'max_log_file_size_mb': 100
                }
            }
            
            for key, value in default_configs.items():
                self._save_config(key, value, f"默认{key}配置")
            
            # 初始化默认缓存策略
            self._init_default_cache_strategies()
            
            # 初始化性能基线
            self._init_performance_baselines()
            
        except Exception as e:
            logger.error(f"初始化默认配置失败: {e}")
    
    def _init_default_cache_strategies(self):
        """初始化默认缓存策略"""
        try:
            default_strategies = [
                CacheStrategy(
                    strategy_name='stock_data_cache',
                    cache_type='redis',
                    ttl_seconds=3600,  # 1小时
                    max_size=10000,
                    eviction_policy='lru',
                    compression_enabled=True,
                    serialization_format='json',
                    hit_rate_threshold=0.8
                ),
                CacheStrategy(
                    strategy_name='model_predictions_cache',
                    cache_type='redis',
                    ttl_seconds=1800,  # 30分钟
                    max_size=5000,
                    eviction_policy='lru',
                    compression_enabled=True,
                    serialization_format='pickle',
                    hit_rate_threshold=0.75
                ),
                CacheStrategy(
                    strategy_name='user_session_cache',
                    cache_type='memory',
                    ttl_seconds=7200,  # 2小时
                    max_size=1000,
                    eviction_policy='lru',
                    compression_enabled=False,
                    serialization_format='json',
                    hit_rate_threshold=0.9
                )
            ]
            
            for strategy in default_strategies:
                self.save_cache_strategy(strategy)
            
        except Exception as e:
            logger.error(f"初始化默认缓存策略失败: {e}")
    
    def _init_performance_baselines(self):
        """初始化性能基线"""
        try:
            baselines = [
                {'metric_name': 'cpu_usage', 'baseline_value': 50.0, 'upper_threshold': 80.0, 'lower_threshold': 10.0},
                {'metric_name': 'memory_usage', 'baseline_value': 60.0, 'upper_threshold': 85.0, 'lower_threshold': 20.0},
                {'metric_name': 'response_time', 'baseline_value': 0.5, 'upper_threshold': 2.0, 'lower_threshold': 0.1},
                {'metric_name': 'throughput', 'baseline_value': 1000.0, 'upper_threshold': 2000.0, 'lower_threshold': 100.0},
                {'metric_name': 'cache_hit_rate', 'baseline_value': 85.0, 'upper_threshold': 95.0, 'lower_threshold': 70.0}
            ]
            
            for baseline in baselines:
                self._save_performance_baseline(
                    baseline['metric_name'],
                    baseline['baseline_value'],
                    baseline['upper_threshold'],
                    baseline['lower_threshold'],
                    'daily'
                )
            
        except Exception as e:
            logger.error(f"初始化性能基线失败: {e}")
    
    def start_monitoring(self):
        """启动系统监控"""
        if self._monitoring_active:
            logger.warning("系统监控已在运行")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止系统监控"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("系统监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self._monitoring_active:
            try:
                # 收集系统指标
                metrics = self.collect_system_metrics()
                
                # 保存指标
                self.save_system_metrics(metrics)
                
                # 检查告警
                self.check_performance_alerts(metrics)
                
                # 执行自动优化
                self.run_auto_optimization()
                
                # 获取监控间隔
                interval = self._get_config('monitoring_interval', 60)
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(60)  # 出错时等待1分钟再继续
    
    def collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # 网络IO
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': float(network.bytes_sent),
                'bytes_recv': float(network.bytes_recv),
                'packets_sent': float(network.packets_sent),
                'packets_recv': float(network.packets_recv)
            }
            
            # 数据库连接数
            db_connections = self._get_database_connections()
            
            # 缓存命中率
            cache_hit_rate = self._get_cache_hit_rate()
            
            # 响应时间（模拟）
            response_time = self._measure_response_time()
            
            # 吞吐量（模拟）
            throughput = self._measure_throughput()
            
            # 错误率（模拟）
            error_rate = self._measure_error_rate()
            
            # 活跃用户数（模拟）
            active_users = self._get_active_users()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                database_connections=db_connections,
                cache_hit_rate=cache_hit_rate,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate,
                active_users=active_users
            )
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            raise
    
    def save_system_metrics(self, metrics: SystemMetrics):
        """保存系统指标"""
        try:
            insert_sql = """
            INSERT INTO system_metrics (
                timestamp, cpu_usage, memory_usage, disk_usage, network_io,
                database_connections, cache_hit_rate, response_time,
                throughput, error_rate, active_users
            ) VALUES (
                :timestamp, :cpu_usage, :memory_usage, :disk_usage, :network_io,
                :database_connections, :cache_hit_rate, :response_time,
                :throughput, :error_rate, :active_users
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'timestamp': metrics.timestamp,
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'disk_usage': metrics.disk_usage,
                    'network_io': json.dumps(metrics.network_io),
                    'database_connections': metrics.database_connections,
                    'cache_hit_rate': metrics.cache_hit_rate,
                    'response_time': metrics.response_time,
                    'throughput': metrics.throughput,
                    'error_rate': metrics.error_rate,
                    'active_users': metrics.active_users
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存系统指标失败: {e}")
    
    def check_performance_alerts(self, metrics: SystemMetrics):
        """检查性能告警"""
        try:
            thresholds = self._get_config('alert_thresholds', {})
            alerts = []
            
            # 检查CPU使用率
            if metrics.cpu_usage > thresholds.get('cpu_usage', 80):
                alert = self._create_alert(
                    'cpu',
                    'high' if metrics.cpu_usage < 90 else 'critical',
                    'cpu_usage',
                    metrics.cpu_usage,
                    thresholds.get('cpu_usage', 80),
                    f"CPU使用率过高: {metrics.cpu_usage:.1f}%",
                    ['检查CPU密集型进程', '考虑增加CPU资源', '优化算法复杂度']
                )
                alerts.append(alert)
            
            # 检查内存使用率
            if metrics.memory_usage > thresholds.get('memory_usage', 85):
                alert = self._create_alert(
                    'memory',
                    'high' if metrics.memory_usage < 95 else 'critical',
                    'memory_usage',
                    metrics.memory_usage,
                    thresholds.get('memory_usage', 85),
                    f"内存使用率过高: {metrics.memory_usage:.1f}%",
                    ['检查内存泄漏', '清理缓存', '增加内存资源', '优化数据结构']
                )
                alerts.append(alert)
            
            # 检查磁盘使用率
            if metrics.disk_usage > thresholds.get('disk_usage', 90):
                alert = self._create_alert(
                    'disk',
                    'high' if metrics.disk_usage < 95 else 'critical',
                    'disk_usage',
                    metrics.disk_usage,
                    thresholds.get('disk_usage', 90),
                    f"磁盘使用率过高: {metrics.disk_usage:.1f}%",
                    ['清理临时文件', '归档历史数据', '增加存储空间', '压缩数据']
                )
                alerts.append(alert)
            
            # 检查响应时间
            if metrics.response_time > thresholds.get('response_time', 2.0):
                alert = self._create_alert(
                    'response_time',
                    'medium' if metrics.response_time < 5.0 else 'high',
                    'response_time',
                    metrics.response_time,
                    thresholds.get('response_time', 2.0),
                    f"响应时间过长: {metrics.response_time:.2f}秒",
                    ['优化数据库查询', '增加缓存', '优化算法', '增加服务器资源']
                )
                alerts.append(alert)
            
            # 检查错误率
            if metrics.error_rate > thresholds.get('error_rate', 5.0):
                alert = self._create_alert(
                    'error_rate',
                    'medium' if metrics.error_rate < 10.0 else 'high',
                    'error_rate',
                    metrics.error_rate,
                    thresholds.get('error_rate', 5.0),
                    f"错误率过高: {metrics.error_rate:.1f}%",
                    ['检查错误日志', '修复bug', '增加异常处理', '提高代码质量']
                )
                alerts.append(alert)
            
            # 检查缓存命中率
            if metrics.cache_hit_rate < thresholds.get('cache_hit_rate', 70.0):
                alert = self._create_alert(
                    'cache',
                    'medium',
                    'cache_hit_rate',
                    metrics.cache_hit_rate,
                    thresholds.get('cache_hit_rate', 70.0),
                    f"缓存命中率过低: {metrics.cache_hit_rate:.1f}%",
                    ['优化缓存策略', '增加缓存容量', '调整TTL设置', '预热缓存']
                )
                alerts.append(alert)
            
            # 保存告警
            for alert in alerts:
                self.save_performance_alert(alert)
            
        except Exception as e:
            logger.error(f"检查性能告警失败: {e}")
    
    def _create_alert(self, alert_type: str, severity: str, metric_name: str,
                     current_value: float, threshold_value: float,
                     description: str, recommendations: List[str]) -> PerformanceAlert:
        """创建告警"""
        alert_id = f"{alert_type}_{metric_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return PerformanceAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            description=description,
            recommendations=recommendations,
            created_at=datetime.now()
        )
    
    def save_performance_alert(self, alert: PerformanceAlert):
        """保存性能告警"""
        try:
            # 检查是否已存在相同类型的活跃告警
            existing_alert = self._get_active_alert(alert.alert_type, alert.metric_name)
            if existing_alert:
                # 更新现有告警
                self._update_alert(existing_alert, alert)
                return
            
            insert_sql = """
            INSERT INTO performance_alerts (
                alert_id, alert_type, severity, metric_name, current_value,
                threshold_value, description, recommendations, status
            ) VALUES (
                :alert_id, :alert_type, :severity, :metric_name, :current_value,
                :threshold_value, :description, :recommendations, :status
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'description': alert.description,
                    'recommendations': json.dumps(alert.recommendations),
                    'status': alert.status
                })
                conn.commit()
            
            logger.info(f"创建性能告警: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"保存性能告警失败: {e}")
    
    def run_auto_optimization(self):
        """运行自动优化"""
        try:
            with self._optimization_lock:
                # 检查是否有待执行的优化任务
                pending_tasks = self.get_pending_optimization_tasks()
                
                for task in pending_tasks:
                    if task.priority in ['high', 'critical']:
                        self.execute_optimization_task(task)
                
                # 检查是否需要创建新的优化任务
                self._check_auto_optimization_triggers()
            
        except Exception as e:
            logger.error(f"自动优化执行失败: {e}")
    
    def _check_auto_optimization_triggers(self):
        """检查自动优化触发条件"""
        try:
            # 检查缓存清理需求
            if self._should_cleanup_cache():
                task = OptimizationTask(
                    task_id=f"CACHE_CLEANUP_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    task_type='cache_cleanup',
                    priority='medium',
                    description='自动缓存清理任务',
                    parameters={'cleanup_expired': True, 'cleanup_lru': True},
                    created_at=datetime.now()
                )
                self.create_optimization_task(task)
            
            # 检查数据库优化需求
            if self._should_optimize_database():
                task = OptimizationTask(
                    task_id=f"DB_OPTIMIZE_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    task_type='db_optimization',
                    priority='low',
                    description='自动数据库优化任务',
                    parameters={'analyze_tables': True, 'rebuild_indexes': False},
                    created_at=datetime.now()
                )
                self.create_optimization_task(task)
            
            # 检查日志归档需求
            if self._should_archive_logs():
                task = OptimizationTask(
                    task_id=f"LOG_ARCHIVE_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    task_type='log_archival',
                    priority='low',
                    description='自动日志归档任务',
                    parameters={'archive_days': 30, 'compress': True},
                    created_at=datetime.now()
                )
                self.create_optimization_task(task)
            
        except Exception as e:
            logger.error(f"检查自动优化触发条件失败: {e}")
    
    def create_optimization_task(self, task: OptimizationTask) -> str:
        """创建优化任务"""
        try:
            insert_sql = """
            INSERT INTO optimization_tasks (
                task_id, task_type, priority, description, parameters, status
            ) VALUES (
                :task_id, :task_type, :priority, :description, :parameters, :status
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'priority': task.priority,
                    'description': task.description,
                    'parameters': json.dumps(task.parameters),
                    'status': task.status
                })
                conn.commit()
            
            logger.info(f"创建优化任务: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"创建优化任务失败: {e}")
            raise
    
    def execute_optimization_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行优化任务"""
        try:
            # 更新任务状态为运行中
            self._update_task_status(task.task_id, 'running', start_time=datetime.now())
            
            result = {}
            
            if task.task_type == 'cache_cleanup':
                result = self._execute_cache_cleanup(task)
            elif task.task_type == 'db_optimization':
                result = self._execute_db_optimization(task)
            elif task.task_type == 'log_archival':
                result = self._execute_log_archival(task)
            elif task.task_type == 'index_rebuild':
                result = self._execute_index_rebuild(task)
            else:
                raise ValueError(f"未知的优化任务类型: {task.task_type}")
            
            # 更新任务状态为完成
            self._update_task_status(
                task.task_id, 'completed',
                end_time=datetime.now(),
                progress=100.0,
                result=result
            )
            
            logger.info(f"优化任务执行完成: {task.task_id}")
            return result
            
        except Exception as e:
            # 更新任务状态为失败
            self._update_task_status(
                task.task_id, 'failed',
                end_time=datetime.now(),
                error_message=str(e)
            )
            logger.error(f"优化任务执行失败: {task.task_id}, 错误: {e}")
            raise
    
    def get_optimization_recommendations(self, metrics: SystemMetrics = None) -> List[Dict[str, Any]]:
        """获取优化建议
        
        Args:
            metrics: 系统指标，如果为None则获取最新指标
            
        Returns:
            优化建议列表
        """
        try:
            if not metrics:
                metrics = self.get_latest_system_metrics()
            
            if not metrics:
                return []
            
            recommendations = []
            
            # CPU优化建议
            if metrics.cpu_usage > 70:
                recommendations.append({
                    'category': 'CPU优化',
                    'priority': 'high' if metrics.cpu_usage > 85 else 'medium',
                    'title': 'CPU使用率过高',
                    'description': f'当前CPU使用率为{metrics.cpu_usage:.1f}%，建议进行优化',
                    'actions': [
                        '检查并优化CPU密集型算法',
                        '考虑使用多进程或异步处理',
                        '增加CPU资源或横向扩展',
                        '优化数据库查询和索引'
                    ],
                    'estimated_impact': 'high'
                })
            
            # 内存优化建议
            if metrics.memory_usage > 75:
                recommendations.append({
                    'category': '内存优化',
                    'priority': 'high' if metrics.memory_usage > 90 else 'medium',
                    'title': '内存使用率过高',
                    'description': f'当前内存使用率为{metrics.memory_usage:.1f}%，建议进行优化',
                    'actions': [
                        '检查并修复内存泄漏',
                        '优化缓存策略和大小',
                        '使用更高效的数据结构',
                        '实施垃圾回收优化',
                        '考虑增加内存资源'
                    ],
                    'estimated_impact': 'high'
                })
            
            # 缓存优化建议
            if metrics.cache_hit_rate < 80:
                recommendations.append({
                    'category': '缓存优化',
                    'priority': 'medium',
                    'title': '缓存命中率偏低',
                    'description': f'当前缓存命中率为{metrics.cache_hit_rate:.1f}%，建议进行优化',
                    'actions': [
                        '分析缓存访问模式',
                        '调整缓存TTL设置',
                        '增加缓存容量',
                        '优化缓存键设计',
                        '实施缓存预热策略'
                    ],
                    'estimated_impact': 'medium'
                })
            
            # 响应时间优化建议
            if metrics.response_time > 1.0:
                recommendations.append({
                    'category': '性能优化',
                    'priority': 'high' if metrics.response_time > 3.0 else 'medium',
                    'title': '响应时间过长',
                    'description': f'当前平均响应时间为{metrics.response_time:.2f}秒，建议进行优化',
                    'actions': [
                        '优化数据库查询性能',
                        '增加适当的索引',
                        '实施查询结果缓存',
                        '优化API接口设计',
                        '考虑使用CDN加速'
                    ],
                    'estimated_impact': 'high'
                })
            
            # 数据库优化建议
            if metrics.database_connections > 80:
                recommendations.append({
                    'category': '数据库优化',
                    'priority': 'medium',
                    'title': '数据库连接数过多',
                    'description': f'当前数据库连接数为{metrics.database_connections}，建议进行优化',
                    'actions': [
                        '优化连接池配置',
                        '检查长时间运行的查询',
                        '实施连接复用',
                        '优化事务处理',
                        '考虑读写分离'
                    ],
                    'estimated_impact': 'medium'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"获取优化建议失败: {e}")
            return []
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """获取系统健康报告"""
        try:
            # 获取最新指标
            latest_metrics = self.get_latest_system_metrics()
            
            # 获取历史趋势
            trends = self.get_performance_trends(days=7)
            
            # 获取活跃告警
            active_alerts = self.get_active_alerts()
            
            # 获取优化建议
            recommendations = self.get_optimization_recommendations(latest_metrics)
            
            # 计算健康评分
            health_score = self._calculate_health_score(latest_metrics)
            
            # 获取资源使用统计
            resource_stats = self._get_resource_usage_stats()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'health_score': health_score,
                'status': self._get_health_status(health_score),
                'latest_metrics': asdict(latest_metrics) if latest_metrics else {},
                'trends': trends,
                'active_alerts': [asdict(alert) for alert in active_alerts],
                'recommendations': recommendations,
                'resource_stats': resource_stats,
                'summary': self._generate_health_summary(health_score, len(active_alerts), len(recommendations))
            }
            
        except Exception as e:
            logger.error(f"获取系统健康报告失败: {e}")
            return {}
    
    def save_cache_strategy(self, strategy: CacheStrategy):
        """保存缓存策略"""
        try:
            # 检查是否已存在
            existing = self._get_cache_strategy(strategy.strategy_name)
            if existing:
                # 更新现有策略
                update_sql = """
                UPDATE cache_strategies
                SET cache_type = :cache_type, ttl_seconds = :ttl_seconds,
                    max_size = :max_size, eviction_policy = :eviction_policy,
                    compression_enabled = :compression_enabled,
                    serialization_format = :serialization_format,
                    hit_rate_threshold = :hit_rate_threshold,
                    is_active = :is_active
                WHERE strategy_name = :strategy_name
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(update_sql), {
                        'strategy_name': strategy.strategy_name,
                        'cache_type': strategy.cache_type,
                        'ttl_seconds': strategy.ttl_seconds,
                        'max_size': strategy.max_size,
                        'eviction_policy': strategy.eviction_policy,
                        'compression_enabled': strategy.compression_enabled,
                        'serialization_format': strategy.serialization_format,
                        'hit_rate_threshold': strategy.hit_rate_threshold,
                        'is_active': strategy.is_active
                    })
                    conn.commit()
            else:
                # 插入新策略
                insert_sql = """
                INSERT INTO cache_strategies (
                    strategy_name, cache_type, ttl_seconds, max_size,
                    eviction_policy, compression_enabled, serialization_format,
                    hit_rate_threshold, is_active
                ) VALUES (
                    :strategy_name, :cache_type, :ttl_seconds, :max_size,
                    :eviction_policy, :compression_enabled, :serialization_format,
                    :hit_rate_threshold, :is_active
                )
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(insert_sql), {
                        'strategy_name': strategy.strategy_name,
                        'cache_type': strategy.cache_type,
                        'ttl_seconds': strategy.ttl_seconds,
                        'max_size': strategy.max_size,
                        'eviction_policy': strategy.eviction_policy,
                        'compression_enabled': strategy.compression_enabled,
                        'serialization_format': strategy.serialization_format,
                        'hit_rate_threshold': strategy.hit_rate_threshold,
                        'is_active': strategy.is_active
                    })
                    conn.commit()
            
        except Exception as e:
            logger.error(f"保存缓存策略失败: {e}")
            raise
    
    # 辅助方法
    def _get_database_connections(self) -> int:
        """获取数据库连接数"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'"))
                return result.scalar() or 0
        except:
            return 0
    
    def _get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                hits = info.get('keyspace_hits', 0)
                misses = info.get('keyspace_misses', 0)
                total = hits + misses
                return (hits / total * 100) if total > 0 else 0
            return 85.0  # 默认值
        except:
            return 85.0
    
    def _measure_response_time(self) -> float:
        """测量响应时间"""
        # 这里应该实现实际的响应时间测量
        # 可以通过健康检查接口或数据库查询来测量
        return 0.5  # 模拟值
    
    def _measure_throughput(self) -> float:
        """测量吞吐量"""
        # 这里应该实现实际的吞吐量测量
        return 1000.0  # 模拟值
    
    def _measure_error_rate(self) -> float:
        """测量错误率"""
        # 这里应该实现实际的错误率测量
        return 1.0  # 模拟值
    
    def _get_active_users(self) -> int:
        """获取活跃用户数"""
        # 这里应该实现实际的活跃用户统计
        return 100  # 模拟值
    
    def _get_config(self, key: str, default=None):
        """获取配置"""
        try:
            query_sql = """
            SELECT config_value
            FROM system_configs
            WHERE config_key = :key AND is_active = true
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'key': key})
                row = result.fetchone()
            
            if row:
                return json.loads(row[0])
            return default
            
        except Exception as e:
            logger.error(f"获取配置失败: {e}")
            return default
    
    def _save_config(self, key: str, value: Any, description: str = None):
        """保存配置"""
        try:
            # 使用 INSERT ... ON CONFLICT 来避免重复插入
            upsert_sql = """
            INSERT INTO system_configs (config_key, config_value, description, created_at, updated_at)
            VALUES (:key, :value, :description, :created_at, :updated_at)
            ON CONFLICT (config_key) 
            DO UPDATE SET 
                config_value = EXCLUDED.config_value,
                description = EXCLUDED.description,
                updated_at = EXCLUDED.updated_at
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(upsert_sql), {
                    'key': key,
                    'value': json.dumps(value),
                    'description': description,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """获取最新系统指标"""
        try:
            query_sql = """
            SELECT *
            FROM system_metrics
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql))
                row = result.fetchone()
            
            if row:
                return SystemMetrics(
                    timestamp=row[1],
                    cpu_usage=float(row[2]),
                    memory_usage=float(row[3]),
                    disk_usage=float(row[4]),
                    network_io=json.loads(row[5]) if row[5] else {},
                    database_connections=row[6],
                    cache_hit_rate=float(row[7]),
                    response_time=float(row[8]),
                    throughput=float(row[9]),
                    error_rate=float(row[10]),
                    active_users=row[11]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取最新系统指标失败: {e}")
            return None
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """获取性能趋势"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query_sql = """
            SELECT 
                DATE(timestamp) as date,
                AVG(cpu_usage) as avg_cpu,
                AVG(memory_usage) as avg_memory,
                AVG(response_time) as avg_response_time,
                AVG(throughput) as avg_throughput,
                AVG(error_rate) as avg_error_rate,
                AVG(cache_hit_rate) as avg_cache_hit_rate
            FROM system_metrics
            WHERE timestamp >= :start_date AND timestamp <= :end_date
            GROUP BY DATE(timestamp)
            ORDER BY date
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'start_date': start_date,
                    'end_date': end_date
                })
                rows = result.fetchall()
            
            trends = {
                'dates': [],
                'cpu_usage': [],
                'memory_usage': [],
                'response_time': [],
                'throughput': [],
                'error_rate': [],
                'cache_hit_rate': []
            }
            
            for row in rows:
                trends['dates'].append(row[0].isoformat())
                trends['cpu_usage'].append(float(row[1]) if row[1] else 0)
                trends['memory_usage'].append(float(row[2]) if row[2] else 0)
                trends['response_time'].append(float(row[3]) if row[3] else 0)
                trends['throughput'].append(float(row[4]) if row[4] else 0)
                trends['error_rate'].append(float(row[5]) if row[5] else 0)
                trends['cache_hit_rate'].append(float(row[6]) if row[6] else 0)
            
            return trends
            
        except Exception as e:
            logger.error(f"获取性能趋势失败: {e}")
            return {}
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """获取活跃告警"""
        try:
            query_sql = """
            SELECT *
            FROM performance_alerts
            WHERE status = 'active'
            ORDER BY created_at DESC
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql))
                rows = result.fetchall()
            
            alerts = []
            for row in rows:
                alert = PerformanceAlert(
                    alert_id=row[1],
                    alert_type=row[2],
                    severity=row[3],
                    metric_name=row[4],
                    current_value=float(row[5]),
                    threshold_value=float(row[6]),
                    description=row[7],
                    recommendations=json.loads(row[8]) if row[8] else [],
                    status=row[9],
                    resolved_at=row[10],
                    created_at=row[11]
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"获取活跃告警失败: {e}")
            return []
    
    def get_pending_optimization_tasks(self) -> List[OptimizationTask]:
        """获取待执行的优化任务"""
        try:
            query_sql = """
            SELECT *
            FROM optimization_tasks
            WHERE status = 'pending'
            ORDER BY 
                CASE priority 
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END,
                created_at
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql))
                rows = result.fetchall()
            
            tasks = []
            for row in rows:
                task = OptimizationTask(
                    task_id=row[1],
                    task_type=row[2],
                    priority=row[3],
                    description=row[4],
                    parameters=json.loads(row[5]) if row[5] else {},
                    status=row[6],
                    progress=float(row[7]),
                    start_time=row[8],
                    end_time=row[9],
                    result=json.loads(row[10]) if row[10] else None,
                    error_message=row[11],
                    created_at=row[12]
                )
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"获取待执行优化任务失败: {e}")
            return []
    
    # 更多辅助方法的实现
    def _should_cleanup_cache(self) -> bool:
        """检查是否需要清理缓存"""
        try:
            if not self.redis_client:
                return False
            
            # 检查缓存使用情况
            info = self.redis_client.info('memory')
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            if max_memory > 0:
                usage_ratio = used_memory / max_memory
                return usage_ratio > 0.8  # 使用率超过80%时清理
            
            return False
        except:
            return False
    
    def _should_optimize_database(self) -> bool:
        """检查是否需要优化数据库"""
        try:
            # 检查数据库性能指标
            with self.engine.connect() as conn:
                # 检查慢查询数量
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM pg_stat_statements 
                    WHERE mean_time > 1000
                """))
                slow_queries = result.scalar() or 0
                
                # 检查表膨胀情况
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM pg_stat_user_tables 
                    WHERE n_dead_tup > n_live_tup * 0.1
                """))
                bloated_tables = result.scalar() or 0
                
                return slow_queries > 10 or bloated_tables > 5
        except:
            return False
    
    def _should_archive_logs(self) -> bool:
        """检查是否需要归档日志"""
        try:
            # 检查日志表大小
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM system_metrics 
                    WHERE created_at < NOW() - INTERVAL '30 days'
                """))
                old_records = result.scalar() or 0
                
                return old_records > 10000
        except:
            return False
    
    def _execute_cache_cleanup(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行缓存清理"""
        try:
            result = {'cleaned_keys': 0, 'freed_memory': 0}
            
            if self.redis_client:
                # 清理过期键
                if task.parameters.get('cleanup_expired', True):
                    # Redis会自动清理过期键，这里记录统计
                    info_before = self.redis_client.info('memory')
                    
                    # 手动触发过期键清理
                    self.redis_client.execute_command('MEMORY', 'PURGE')
                    
                    info_after = self.redis_client.info('memory')
                    freed = info_before.get('used_memory', 0) - info_after.get('used_memory', 0)
                    result['freed_memory'] = max(0, freed)
                
                # 清理LRU键
                if task.parameters.get('cleanup_lru', False):
                    # 获取内存使用情况
                    info = self.redis_client.info('memory')
                    used_memory = info.get('used_memory', 0)
                    max_memory = info.get('maxmemory', 0)
                    
                    if max_memory > 0 and used_memory / max_memory > 0.9:
                        # 触发LRU清理
                        self.redis_client.execute_command('MEMORY', 'PURGE')
            
            return result
            
        except Exception as e:
            logger.error(f"缓存清理执行失败: {e}")
            raise
    
    def _execute_db_optimization(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行数据库优化"""
        try:
            result = {'analyzed_tables': 0, 'rebuilt_indexes': 0}
            
            with self.engine.connect() as conn:
                # 分析表统计信息
                if task.parameters.get('analyze_tables', True):
                    tables = ['system_metrics', 'performance_alerts', 'optimization_tasks']
                    for table in tables:
                        conn.execute(text(f"ANALYZE {table}"))
                        result['analyzed_tables'] += 1
                
                # 重建索引
                if task.parameters.get('rebuild_indexes', False):
                    # 获取需要重建的索引
                    index_result = conn.execute(text("""
                        SELECT indexname, tablename
                        FROM pg_indexes
                        WHERE schemaname = 'public'
                        AND indexname NOT LIKE 'pg_%'
                    """))
                    
                    for row in index_result:
                        try:
                            conn.execute(text(f"REINDEX INDEX {row[0]}"))
                            result['rebuilt_indexes'] += 1
                        except:
                            continue
                
                conn.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"数据库优化执行失败: {e}")
            raise
    
    def _execute_log_archival(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行日志归档"""
        try:
            result = {'archived_records': 0}
            
            archive_days = task.parameters.get('archive_days', 30)
            archive_date = datetime.now() - timedelta(days=archive_days)
            
            with self.engine.connect() as conn:
                # 归档旧的系统指标记录
                delete_result = conn.execute(text("""
                    DELETE FROM system_metrics
                    WHERE created_at < :archive_date
                """), {'archive_date': archive_date})
                
                result['archived_records'] = delete_result.rowcount
                
                # 归档旧的告警记录
                conn.execute(text("""
                    DELETE FROM performance_alerts
                    WHERE created_at < :archive_date AND status = 'resolved'
                """), {'archive_date': archive_date})
                
                conn.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"日志归档执行失败: {e}")
            raise
    
    def _execute_index_rebuild(self, task: OptimizationTask) -> Dict[str, Any]:
        """执行索引重建"""
        try:
            result = {'rebuilt_indexes': 0}
            
            tables = task.parameters.get('tables', [])
            
            with self.engine.connect() as conn:
                for table in tables:
                    try:
                        conn.execute(text(f"REINDEX TABLE {table}"))
                        result['rebuilt_indexes'] += 1
                    except Exception as e:
                        logger.warning(f"重建表{table}索引失败: {e}")
                        continue
                
                conn.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"索引重建执行失败: {e}")
            raise
    
    def _update_task_status(self, task_id: str, status: str, **kwargs):
        """更新任务状态"""
        try:
            update_fields = ['status = :status']
            params = {'task_id': task_id, 'status': status}
            
            if 'start_time' in kwargs:
                update_fields.append('start_time = :start_time')
                params['start_time'] = kwargs['start_time']
            
            if 'end_time' in kwargs:
                update_fields.append('end_time = :end_time')
                params['end_time'] = kwargs['end_time']
            
            if 'progress' in kwargs:
                update_fields.append('progress = :progress')
                params['progress'] = kwargs['progress']
            
            if 'result' in kwargs:
                update_fields.append('result = :result')
                params['result'] = json.dumps(kwargs['result'])
            
            if 'error_message' in kwargs:
                update_fields.append('error_message = :error_message')
                params['error_message'] = kwargs['error_message']
            
            update_sql = f"""
            UPDATE optimization_tasks
            SET {', '.join(update_fields)}
            WHERE task_id = :task_id
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(update_sql), params)
                conn.commit()
            
        except Exception as e:
            logger.error(f"更新任务状态失败: {e}")
    
    def _get_active_alert(self, alert_type: str, metric_name: str) -> Optional[str]:
        """获取活跃告警"""
        try:
            query_sql = """
            SELECT alert_id
            FROM performance_alerts
            WHERE alert_type = :alert_type AND metric_name = :metric_name AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'alert_type': alert_type,
                    'metric_name': metric_name
                })
                row = result.fetchone()
            
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"获取活跃告警失败: {e}")
            return None
    
    def _update_alert(self, alert_id: str, new_alert: PerformanceAlert):
        """更新告警"""
        try:
            update_sql = """
            UPDATE performance_alerts
            SET current_value = :current_value, severity = :severity,
                description = :description, recommendations = :recommendations
            WHERE alert_id = :alert_id
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(update_sql), {
                    'alert_id': alert_id,
                    'current_value': new_alert.current_value,
                    'severity': new_alert.severity,
                    'description': new_alert.description,
                    'recommendations': json.dumps(new_alert.recommendations)
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"更新告警失败: {e}")
    
    def _get_cache_strategy(self, strategy_name: str) -> Optional[CacheStrategy]:
        """获取缓存策略"""
        try:
            query_sql = """
            SELECT *
            FROM cache_strategies
            WHERE strategy_name = :strategy_name
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'strategy_name': strategy_name})
                row = result.fetchone()
            
            if row:
                return CacheStrategy(
                    strategy_name=row[1],
                    cache_type=row[2],
                    ttl_seconds=row[3],
                    max_size=row[4],
                    eviction_policy=row[5],
                    compression_enabled=row[6],
                    serialization_format=row[7],
                    hit_rate_threshold=float(row[8]),
                    is_active=row[9],
                    created_at=row[10]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取缓存策略失败: {e}")
            return None
    
    def _save_performance_baseline(self, metric_name: str, baseline_value: float,
                                 upper_threshold: float, lower_threshold: float,
                                 measurement_period: str):
        """保存性能基线"""
        try:
            # 检查是否已存在
            query_sql = """
            SELECT id FROM performance_baselines
            WHERE metric_name = :metric_name AND measurement_period = :measurement_period
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {
                    'metric_name': metric_name,
                    'measurement_period': measurement_period
                })
                existing = result.fetchone()
            
            if existing:
                # 更新现有基线
                update_sql = """
                UPDATE performance_baselines
                SET baseline_value = :baseline_value,
                    upper_threshold = :upper_threshold,
                    lower_threshold = :lower_threshold,
                    updated_at = :updated_at
                WHERE metric_name = :metric_name AND measurement_period = :measurement_period
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(update_sql), {
                        'metric_name': metric_name,
                        'baseline_value': baseline_value,
                        'upper_threshold': upper_threshold,
                        'lower_threshold': lower_threshold,
                        'measurement_period': measurement_period,
                        'updated_at': datetime.now()
                    })
                    conn.commit()
            else:
                # 插入新基线
                insert_sql = """
                INSERT INTO performance_baselines (
                    metric_name, baseline_value, upper_threshold,
                    lower_threshold, measurement_period
                ) VALUES (
                    :metric_name, :baseline_value, :upper_threshold,
                    :lower_threshold, :measurement_period
                )
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(insert_sql), {
                        'metric_name': metric_name,
                        'baseline_value': baseline_value,
                        'upper_threshold': upper_threshold,
                        'lower_threshold': lower_threshold,
                        'measurement_period': measurement_period
                    })
                    conn.commit()
            
        except Exception as e:
            logger.error(f"保存性能基线失败: {e}")
    
    def _get_resource_usage_stats(self) -> Dict[str, Any]:
        """获取资源使用统计"""
        try:
            stats = {}
            
            # CPU统计
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            stats['cpu'] = {
                'usage_percent': cpu_percent,
                'core_count': cpu_count,
                'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
            }
            
            # 内存统计
            memory = psutil.virtual_memory()
            stats['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'usage_percent': memory.percent
            }
            
            # 磁盘统计
            disk = psutil.disk_usage('/')
            stats['disk'] = {
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'usage_percent': round((disk.used / disk.total) * 100, 2)
            }
            
            # 网络统计
            network = psutil.net_io_counters()
            stats['network'] = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取资源使用统计失败: {e}")
            return {}
    
    def _generate_health_summary(self, health_score: float, alert_count: int, recommendation_count: int) -> str:
        """生成健康摘要"""
        status = self._get_health_status(health_score)
        
        if status == 'excellent':
            summary = f"系统运行状态优秀（评分: {health_score}），性能表现良好。"
        elif status == 'good':
            summary = f"系统运行状态良好（评分: {health_score}），整体性能稳定。"
        elif status == 'fair':
            summary = f"系统运行状态一般（评分: {health_score}），建议关注性能优化。"
        elif status == 'poor':
            summary = f"系统运行状态较差（评分: {health_score}），需要及时优化。"
        else:
            summary = f"系统运行状态危险（评分: {health_score}），需要立即处理。"
        
        if alert_count > 0:
            summary += f" 当前有{alert_count}个活跃告警需要处理。"
        
        if recommendation_count > 0:
            summary += f" 系统提供了{recommendation_count}条优化建议。"
        
        return summary
    
    def _calculate_health_score(self, metrics: SystemMetrics) -> float:
        """计算健康评分"""
        if not metrics:
            return 0.0
        
        # 基于各项指标计算综合健康评分
        cpu_score = max(0, 100 - metrics.cpu_usage)
        memory_score = max(0, 100 - metrics.memory_usage)
        response_score = max(0, 100 - min(metrics.response_time * 50, 100))
        cache_score = metrics.cache_hit_rate
        error_score = max(0, 100 - metrics.error_rate * 10)
        
        # 加权平均
        health_score = (
            cpu_score * 0.2 +
            memory_score * 0.2 +
            response_score * 0.25 +
            cache_score * 0.15 +
            error_score * 0.2
        )
        
        return round(health_score, 1)
    
    def _get_health_status(self, score: float) -> str:
        """根据健康评分获取状态"""
        if score >= 90:
            return 'excellent'
        elif score >= 80:
            return 'good'
        elif score >= 70:
            return 'fair'
        elif score >= 60:
            return 'poor'
        else:
            return 'critical'