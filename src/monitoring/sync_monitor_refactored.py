#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的同步监控器

应用了设计模式和最佳实践的版本
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
from collections import defaultdict, deque
from loguru import logger

from .models import SyncTaskInfo, SyncTaskStatus, SyncEvent, SyncEventType
from .repositories import SyncTaskRepository, SyncEventRepository
from .events import EventPublisher, DatabaseEventObserver, LoggingEventObserver, MetricsEventObserver
from .constants import MonitoringConstants
from .decorators import monitoring_operation, handle_exceptions, cache_result
from .services import DashboardService, PerformanceAnalyzer


class SyncMonitor:
    """
    重构后的同步监控器
    
    使用依赖注入和单一职责原则
    """
    
    def __init__(
        self,
        task_repository: SyncTaskRepository,
        event_repository: SyncEventRepository,
        constants: MonitoringConstants,
        event_publisher: Optional[EventPublisher] = None
    ):
        """
        初始化同步监控器
        
        Args:
            task_repository: 任务数据访问对象
            event_repository: 事件数据访问对象
            constants: 监控常量配置
            event_publisher: 事件发布器（可选）
        """
        self.task_repository = task_repository
        self.event_repository = event_repository
        self.constants = constants
        
        # 事件系统
        self.event_publisher = event_publisher or EventPublisher()
        self._setup_event_observers()
        
        # 内存状态管理
        self.active_tasks: Dict[str, SyncTaskInfo] = {}
        self.task_lock = threading.RLock()
        
        # 服务组件
        self.dashboard_service = DashboardService(self.active_tasks, self.constants)
        self.performance_analyzer = PerformanceAnalyzer(task_repository)
        
        logger.info("✅ 重构后的同步监控器初始化成功")
    
    def _setup_event_observers(self) -> None:
        """设置事件观察者"""
        # 数据库观察者
        db_observer = DatabaseEventObserver(self.event_repository)
        self.event_publisher.subscribe_all(db_observer)
        
        # 日志观察者
        log_observer = LoggingEventObserver()
        self.event_publisher.subscribe_all(log_observer)
        
        # 指标观察者
        self.metrics_observer = MetricsEventObserver()
        self.event_publisher.subscribe_all(self.metrics_observer)
    
    # ================================
    # 任务监控核心方法
    # ================================
    
    @monitoring_operation(max_retries=2, measure_time=True)
    def start_task_monitoring(self, task_info: SyncTaskInfo) -> bool:
        """
        开始监控同步任务
        
        Args:
            task_info: 任务信息
            
        Returns:
            是否成功开始监控
        """
        with self.task_lock:
            # 更新任务状态
            task_info.status = SyncTaskStatus.RUNNING
            task_info.start_time = datetime.now()
            task_info.updated_at = datetime.now()
            
            # 添加到活跃任务列表
            self.active_tasks[task_info.task_id] = task_info
        
        # 保存到数据库
        success = self.task_repository.save_task(task_info)
        if not success:
            return False
        
        # 发布事件
        event = SyncEvent(
            event_id="",  # 会在__post_init__中生成
            event_type=SyncEventType.TASK_STARTED,
            task_id=task_info.task_id,
            timestamp=datetime.now(),
            data={
                'data_source': task_info.data_source,
                'data_type': task_info.data_type,
                'target_date': task_info.target_date,
                'priority': task_info.priority
            }
        )
        self.event_publisher.publish(event)
        
        logger.info(f"开始监控同步任务: {task_info.task_id}")
        return True
    
    @handle_exceptions(default_return=False, log_error=True)
    def update_task_progress(
        self,
        task_id: str,
        progress: float,
        records_processed: int = 0,
        records_failed: int = 0,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新任务执行进度
        
        Args:
            task_id: 任务ID
            progress: 进度百分比 (0-100)
            records_processed: 已处理记录数
            records_failed: 失败记录数
            additional_data: 额外数据
            
        Returns:
            是否成功更新
        """
        # 验证参数
        progress = max(0.0, min(100.0, progress))
        
        with self.task_lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                task_info.progress = progress
                task_info.records_processed = records_processed
                task_info.records_failed = records_failed
                task_info.updated_at = datetime.now()
        
        # 更新数据库
        success = self.task_repository.update_progress(
            task_id, progress, records_processed, records_failed
        )
        
        if success:
            # 发布进度事件
            event_data = {
                'progress': progress,
                'records_processed': records_processed,
                'records_failed': records_failed
            }
            if additional_data:
                event_data.update(additional_data)
            
            event = SyncEvent(
                event_id="",
                event_type=SyncEventType.TASK_PROGRESS,
                task_id=task_id,
                timestamp=datetime.now(),
                data=event_data
            )
            self.event_publisher.publish(event)
        
        return success
    
    @monitoring_operation(max_retries=2, measure_time=True)
    def complete_task_monitoring(
        self,
        task_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        final_stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        完成任务监控
        
        Args:
            task_id: 任务ID
            success: 是否成功
            error_message: 错误信息
            final_stats: 最终统计信息
            
        Returns:
            是否成功完成监控
        """
        task_info = None
        
        with self.task_lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                
                # 更新任务状态
                task_info.status = SyncTaskStatus.COMPLETED if success else SyncTaskStatus.FAILED
                task_info.end_time = datetime.now()
                task_info.updated_at = datetime.now()
                
                if task_info.start_time:
                    task_info.duration = (task_info.end_time - task_info.start_time).total_seconds()
                
                if error_message:
                    task_info.error_message = error_message
                
                if not success:
                    task_info.retry_count += 1
                
                # 应用最终统计信息
                if final_stats:
                    if 'records_processed' in final_stats:
                        task_info.records_processed = final_stats['records_processed']
                    if 'records_failed' in final_stats:
                        task_info.records_failed = final_stats['records_failed']
                
                # 从活跃任务列表中移除
                del self.active_tasks[task_id]
        
        if not task_info:
            logger.warning(f"任务 {task_id} 不在活跃任务列表中")
            return False
        
        # 保存到数据库
        repo_success = self.task_repository.save_task(task_info)
        
        if repo_success:
            # 发布完成事件
            event_data = {
                'success': success,
                'duration': task_info.duration,
                'records_processed': task_info.records_processed,
                'records_failed': task_info.records_failed
            }
            if error_message:
                event_data['error_message'] = error_message
            if final_stats:
                event_data.update(final_stats)
            
            event_type = SyncEventType.TASK_COMPLETED if success else SyncEventType.TASK_FAILED
            event = SyncEvent(
                event_id="",
                event_type=event_type,
                task_id=task_id,
                timestamp=datetime.now(),
                data=event_data
            )
            self.event_publisher.publish(event)
            
            logger.info(f"任务监控完成: {task_id}, 成功: {success}")
        
        return repo_success
    
    # ================================
    # 查询方法
    # ================================
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """获取当前活跃的同步任务"""
        with self.task_lock:
            return [task.to_dict() for task in self.active_tasks.values()]
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取指定任务的状态"""
        # 先检查活跃任务
        with self.task_lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].to_dict()
        
        # 从数据库查询
        task_info = self.task_repository.get_task(task_id)
        return task_info.to_dict() if task_info else None
    
    @cache_result(ttl=60)  # 缓存1分钟
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """获取实时监控仪表板数据"""
        return self.dashboard_service.get_dashboard_data()
    
    # ================================
    # 事件系统方法
    # ================================
    
    def subscribe_to_events(self, event_type: SyncEventType, callback) -> None:
        """订阅事件"""
        from .events import CallbackEventObserver
        observer = CallbackEventObserver(callback)
        self.event_publisher.subscribe(event_type, observer)
    
    def get_metrics(self) -> Dict[str, int]:
        """获取监控指标"""
        return self.metrics_observer.get_metrics()
    
    # ================================
    # 性能分析方法
    # ================================
    
    def get_performance_analysis(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取性能分析报告"""
        return self.performance_analyzer.analyze_performance(start_date, end_date)


# 工厂函数
def create_sync_monitor(config: Dict[str, Any]) -> SyncMonitor:
    """
    创建同步监控器实例
    
    Args:
        config: 配置字典
        
    Returns:
        配置好的同步监控器实例
    """
    from .repositories import PostgresSyncTaskRepository, SyncEventRepository
    from src.utils.db import get_db_engine
    
    # 创建依赖
    engine = get_db_engine()
    constants = MonitoringConstants.from_config(config)
    task_repository = PostgresSyncTaskRepository(engine)
    event_repository = SyncEventRepository(engine)
    
    # 创建监控器
    return SyncMonitor(
        task_repository=task_repository,
        event_repository=event_repository,
        constants=constants
    )