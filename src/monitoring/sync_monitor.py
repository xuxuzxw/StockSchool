#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步监控仪表板

实现数据同步任务的实时状态监控、历史记录管理和性能分析功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
import threading
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from sqlalchemy import text, create_engine
from loguru import logger
import asyncio
import websockets
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import get_db_engine
from utils.config_loader import config


class SyncTaskStatus(Enum):
    """同步任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class SyncEventType(Enum):
    """同步事件类型枚举"""
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    SYNC_BATCH_STARTED = "sync_batch_started"
    SYNC_BATCH_COMPLETED = "sync_batch_completed"


@dataclass
class SyncTaskInfo:
    """同步任务信息"""
    task_id: str
    data_source: str
    data_type: str
    target_date: str
    status: SyncTaskStatus
    priority: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    progress: float = 0.0
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class SyncEvent:
    """同步事件"""
    event_id: str
    event_type: SyncEventType
    task_id: str
    timestamp: datetime
    data: Dict[str, Any]
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'task_id': self.task_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }


class SyncMonitor:
    """
    同步监控器
    
    提供数据同步任务的实时状态监控、历史记录管理和性能分析功能
    """
    
    def __init__(self):
        """初始化同步监控器"""
        self.engine = get_db_engine()
        self.config = config
        
        # 实时任务状态缓存
        self.active_tasks: Dict[str, SyncTaskInfo] = {}
        self.task_lock = threading.Lock()
        
        # 事件队列和通知
        self.event_queue = deque(maxlen=1000)
        self.event_subscribers = set()
        self.event_lock = threading.Lock()
        
        # 性能统计
        self.performance_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'avg_duration': 0.0,
            'throughput': 0.0,  # 任务/小时
            'error_rate': 0.0
        }
        
        # 监控配置
        self.monitoring_enabled = self.config.get('monitoring.sync_monitor.enabled', True)
        self.event_retention_hours = self.config.get('monitoring.sync_monitor.event_retention_hours', 24)
        self.performance_window_hours = self.config.get('monitoring.sync_monitor.performance_window_hours', 1)
        
        # 初始化数据库表
        self._init_database_tables()
        
        logger.info("✅ 同步监控器初始化成功")
    
    def _init_database_tables(self):
        """初始化数据库表"""
        try:
            with self.engine.connect() as conn:
                # 同步任务状态表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS sync_task_status (
                        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
                        task_id VARCHAR(100) UNIQUE NOT NULL,
                        data_source VARCHAR(50) NOT NULL,
                        data_type VARCHAR(50) NOT NULL,
                        target_date DATE NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        priority INTEGER DEFAULT 1,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        duration DECIMAL(10,3),
                        progress DECIMAL(5,2) DEFAULT 0.0,
                        records_processed INTEGER DEFAULT 0,
                        records_failed INTEGER DEFAULT 0,
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # 同步事件日志表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS sync_event_log (
                        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
                        event_id VARCHAR(100) UNIQUE NOT NULL,
                        event_type VARCHAR(50) NOT NULL,
                        task_id VARCHAR(100) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # 同步性能统计表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS sync_performance_stats (
                        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
                        stat_date DATE NOT NULL,
                        stat_hour INTEGER NOT NULL,
                        data_source VARCHAR(50),
                        data_type VARCHAR(50),
                        total_tasks INTEGER DEFAULT 0,
                        successful_tasks INTEGER DEFAULT 0,
                        failed_tasks INTEGER DEFAULT 0,
                        avg_duration DECIMAL(10,3) DEFAULT 0.0,
                        min_duration DECIMAL(10,3) DEFAULT 0.0,
                        max_duration DECIMAL(10,3) DEFAULT 0.0,
                        total_records INTEGER DEFAULT 0,
                        throughput DECIMAL(10,2) DEFAULT 0.0,
                        error_rate DECIMAL(5,4) DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(stat_date, stat_hour, data_source, data_type)
                    )
                """))
                
                # 创建索引
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sync_task_status_source_type 
                    ON sync_task_status(data_source, data_type)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sync_task_status_date 
                    ON sync_task_status(target_date)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sync_event_log_timestamp 
                    ON sync_event_log(timestamp)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sync_event_log_task_id 
                    ON sync_event_log(task_id)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sync_performance_stats_date_hour 
                    ON sync_performance_stats(stat_date, stat_hour)
                """))
                
                conn.commit()
                logger.info("✅ 同步监控数据库表初始化成功")
                
        except Exception as e:
            logger.error(f"❌ 初始化同步监控数据库表失败: {e}")
            raise
    
    # ================================
    # 实时状态监控功能 (Task 7.1)
    # ================================
    
    def start_task_monitoring(self, task_info: SyncTaskInfo):
        """
        开始监控同步任务
        
        Args:
            task_info: 任务信息
        """
        if not self.monitoring_enabled:
            return
        
        try:
            with self.task_lock:
                # 更新任务状态
                task_info.status = SyncTaskStatus.RUNNING
                task_info.start_time = datetime.now()
                task_info.updated_at = datetime.now()
                
                # 添加到活跃任务列表
                self.active_tasks[task_info.task_id] = task_info
            
            # 保存到数据库
            self._save_task_status(task_info)
            
            # 发送事件通知
            event = SyncEvent(
                event_id=str(uuid.uuid4()),
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
            self._emit_event(event)
            
            logger.info(f"开始监控同步任务: {task_info.task_id}")
            
        except Exception as e:
            logger.error(f"开始任务监控失败 {task_info.task_id}: {e}")
    
    def update_task_progress(self, task_id: str, progress: float, 
                           records_processed: int = 0, records_failed: int = 0,
                           additional_data: Dict[str, Any] = None):
        """
        更新任务执行进度
        
        Args:
            task_id: 任务ID
            progress: 进度百分比 (0-100)
            records_processed: 已处理记录数
            records_failed: 失败记录数
            additional_data: 额外数据
        """
        if not self.monitoring_enabled:
            return
        
        try:
            with self.task_lock:
                if task_id in self.active_tasks:
                    task_info = self.active_tasks[task_id]
                    task_info.progress = min(100.0, max(0.0, progress))
                    task_info.records_processed = records_processed
                    task_info.records_failed = records_failed
                    task_info.updated_at = datetime.now()
                    
                    # 更新数据库
                    self._update_task_progress_in_db(task_id, progress, 
                                                   records_processed, records_failed)
                    
                    # 发送进度事件
                    event_data = {
                        'progress': progress,
                        'records_processed': records_processed,
                        'records_failed': records_failed
                    }
                    if additional_data:
                        event_data.update(additional_data)
                    
                    event = SyncEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=SyncEventType.TASK_PROGRESS,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data=event_data
                    )
                    self._emit_event(event)
                    
        except Exception as e:
            logger.error(f"更新任务进度失败 {task_id}: {e}")
    
    def complete_task_monitoring(self, task_id: str, success: bool = True, 
                               error_message: str = None, 
                               final_stats: Dict[str, Any] = None):
        """
        完成任务监控
        
        Args:
            task_id: 任务ID
            success: 是否成功
            error_message: 错误信息
            final_stats: 最终统计信息
        """
        if not self.monitoring_enabled:
            return
        
        try:
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
                    
                    # 保存到数据库
                    self._save_task_status(task_info)
                    
                    # 发送完成事件
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
                        event_id=str(uuid.uuid4()),
                        event_type=event_type,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data=event_data
                    )
                    self._emit_event(event)
                    
                    # 从活跃任务列表中移除
                    del self.active_tasks[task_id]
                    
                    logger.info(f"任务监控完成: {task_id}, 成功: {success}")
                    
        except Exception as e:
            logger.error(f"完成任务监控失败 {task_id}: {e}")
    
    def cancel_task_monitoring(self, task_id: str, reason: str = None):
        """
        取消任务监控
        
        Args:
            task_id: 任务ID
            reason: 取消原因
        """
        if not self.monitoring_enabled:
            return
        
        try:
            with self.task_lock:
                if task_id in self.active_tasks:
                    task_info = self.active_tasks[task_id]
                    task_info.status = SyncTaskStatus.CANCELLED
                    task_info.end_time = datetime.now()
                    task_info.updated_at = datetime.now()
                    
                    if reason:
                        task_info.error_message = f"任务被取消: {reason}"
                    
                    if task_info.start_time:
                        task_info.duration = (task_info.end_time - task_info.start_time).total_seconds()
                    
                    # 保存到数据库
                    self._save_task_status(task_info)
                    
                    # 发送取消事件
                    event = SyncEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=SyncEventType.TASK_CANCELLED,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data={'reason': reason or '用户取消'}
                    )
                    self._emit_event(event)
                    
                    # 从活跃任务列表中移除
                    del self.active_tasks[task_id]
                    
                    logger.info(f"任务监控已取消: {task_id}")
                    
        except Exception as e:
            logger.error(f"取消任务监控失败 {task_id}: {e}")
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        获取当前活跃的同步任务
        
        Returns:
            活跃任务列表
        """
        try:
            with self.task_lock:
                return [task.to_dict() for task in self.active_tasks.values()]
        except Exception as e:
            logger.error(f"获取活跃任务失败: {e}")
            return []
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定任务的状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        try:
            # 先检查活跃任务
            with self.task_lock:
                if task_id in self.active_tasks:
                    return self.active_tasks[task_id].to_dict()
            
            # 从数据库查询
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT task_id, data_source, data_type, target_date, status,
                           priority, start_time, end_time, duration, progress,
                           records_processed, records_failed, error_message,
                           retry_count, max_retries, created_at, updated_at
                    FROM sync_task_status
                    WHERE task_id = :task_id
                """), {'task_id': task_id})
                
                row = result.fetchone()
                if row:
                    return {
                        'task_id': row[0],
                        'data_source': row[1],
                        'data_type': row[2],
                        'target_date': row[3].strftime('%Y-%m-%d') if row[3] else None,
                        'status': row[4],
                        'priority': row[5],
                        'start_time': row[6].isoformat() if row[6] else None,
                        'end_time': row[7].isoformat() if row[7] else None,
                        'duration': float(row[8]) if row[8] else None,
                        'progress': float(row[9]) if row[9] else 0.0,
                        'records_processed': row[10] or 0,
                        'records_failed': row[11] or 0,
                        'error_message': row[12],
                        'retry_count': row[13] or 0,
                        'max_retries': row[14] or 3,
                        'created_at': row[15].isoformat() if row[15] else None,
                        'updated_at': row[16].isoformat() if row[16] else None
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"获取任务状态失败 {task_id}: {e}")
            return None
    
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """
        获取实时监控仪表板数据
        
        Returns:
            仪表板数据
        """
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'active_tasks': self.get_active_tasks(),
                'summary': {
                    'total_active': len(self.active_tasks),
                    'running_tasks': 0,
                    'pending_tasks': 0,
                    'avg_progress': 0.0
                },
                'data_sources': defaultdict(int),
                'data_types': defaultdict(int),
                'recent_events': self._get_recent_events(limit=10)
            }
            
            # 计算摘要统计
            total_progress = 0.0
            for task in self.active_tasks.values():
                if task.status == SyncTaskStatus.RUNNING:
                    dashboard['summary']['running_tasks'] += 1
                elif task.status == SyncTaskStatus.PENDING:
                    dashboard['summary']['pending_tasks'] += 1
                
                total_progress += task.progress
                dashboard['data_sources'][task.data_source] += 1
                dashboard['data_types'][task.data_type] += 1
            
            if self.active_tasks:
                dashboard['summary']['avg_progress'] = total_progress / len(self.active_tasks)
            
            # 转换defaultdict为普通dict
            dashboard['data_sources'] = dict(dashboard['data_sources'])
            dashboard['data_types'] = dict(dashboard['data_types'])
            
            return dashboard
            
        except Exception as e:
            logger.error(f"获取实时仪表板数据失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'active_tasks': [],
                'summary': {'total_active': 0, 'running_tasks': 0, 'pending_tasks': 0, 'avg_progress': 0.0},
                'data_sources': {},
                'data_types': {},
                'recent_events': []
            }
    
    # ================================
    # 事件通知机制
    # ================================
    
    def _emit_event(self, event: SyncEvent):
        """
        发送事件通知
        
        Args:
            event: 同步事件
        """
        try:
            with self.event_lock:
                # 添加到事件队列
                self.event_queue.append(event)
                
                # 保存到数据库
                self._save_event_to_db(event)
                
                # 通知所有订阅者
                for subscriber in self.event_subscribers:
                    try:
                        subscriber(event)
                    except Exception as e:
                        logger.warning(f"事件通知失败: {e}")
                        
        except Exception as e:
            logger.error(f"发送事件通知失败: {e}")
    
    def subscribe_events(self, callback):
        """
        订阅事件通知
        
        Args:
            callback: 回调函数
        """
        with self.event_lock:
            self.event_subscribers.add(callback)
    
    def unsubscribe_events(self, callback):
        """
        取消订阅事件通知
        
        Args:
            callback: 回调函数
        """
        with self.event_lock:
            self.event_subscribers.discard(callback)
    
    def _get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取最近的事件
        
        Args:
            limit: 限制数量
            
        Returns:
            事件列表
        """
        try:
            with self.event_lock:
                recent_events = list(self.event_queue)[-limit:]
                return [event.to_dict() for event in reversed(recent_events)]
        except Exception as e:
            logger.error(f"获取最近事件失败: {e}")
            return []
    
    # ================================
    # 历史记录管理功能 (Task 7.2)
    # ================================
    
    def get_sync_history(self, start_date: str = None, end_date: str = None,
                        data_source: str = None, data_type: str = None,
                        status: str = None, limit: int = 100,
                        offset: int = 0) -> Dict[str, Any]:
        """
        获取同步历史记录
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            data_source: 数据源过滤
            data_type: 数据类型过滤
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            历史记录和统计信息
        """
        try:
            # 构建查询条件
            where_conditions = []
            params = {}
            
            if start_date:
                where_conditions.append("target_date >= :start_date")
                params['start_date'] = start_date
            
            if end_date:
                where_conditions.append("target_date <= :end_date")
                params['end_date'] = end_date
            
            if data_source:
                where_conditions.append("data_source = :data_source")
                params['data_source'] = data_source
            
            if data_type:
                where_conditions.append("data_type = :data_type")
                params['data_type'] = data_type
            
            if status:
                where_conditions.append("status = :status")
                params['status'] = status
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            with self.engine.connect() as conn:
                # 获取总数
                count_query = f"""
                    SELECT COUNT(*) FROM sync_task_status {where_clause}
                """
                total_count = conn.execute(text(count_query), params).scalar()
                
                # 获取历史记录
                params['limit'] = limit
                params['offset'] = offset
                
                history_query = f"""
                    SELECT task_id, data_source, data_type, target_date, status,
                           priority, start_time, end_time, duration, progress,
                           records_processed, records_failed, error_message,
                           retry_count, max_retries, created_at, updated_at
                    FROM sync_task_status
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """
                
                result = conn.execute(text(history_query), params)
                
                records = []
                for row in result:
                    records.append({
                        'task_id': row[0],
                        'data_source': row[1],
                        'data_type': row[2],
                        'target_date': row[3].strftime('%Y-%m-%d') if row[3] else None,
                        'status': row[4],
                        'priority': row[5],
                        'start_time': row[6].isoformat() if row[6] else None,
                        'end_time': row[7].isoformat() if row[7] else None,
                        'duration': float(row[8]) if row[8] else None,
                        'progress': float(row[9]) if row[9] else 0.0,
                        'records_processed': row[10] or 0,
                        'records_failed': row[11] or 0,
                        'error_message': row[12],
                        'retry_count': row[13] or 0,
                        'max_retries': row[14] or 3,
                        'created_at': row[15].isoformat() if row[15] else None,
                        'updated_at': row[16].isoformat() if row[16] else None
                    })
                
                return {
                    'records': records,
                    'pagination': {
                        'total': total_count,
                        'limit': limit,
                        'offset': offset,
                        'has_more': offset + limit < total_count
                    },
                    'filters': {
                        'start_date': start_date,
                        'end_date': end_date,
                        'data_source': data_source,
                        'data_type': data_type,
                        'status': status
                    }
                }
                
        except Exception as e:
            logger.error(f"获取同步历史记录失败: {e}")
            return {
                'records': [],
                'pagination': {'total': 0, 'limit': limit, 'offset': offset, 'has_more': False},
                'filters': {},
                'error': str(e)
            }
    
    def get_error_log_analysis(self, start_date: str = None, end_date: str = None,
                              limit: int = 50) -> Dict[str, Any]:
        """
        获取错误日志分析
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            limit: 限制数量
            
        Returns:
            错误分析结果
        """
        try:
            # 构建查询条件
            where_conditions = ["status = 'failed'", "error_message IS NOT NULL"]
            params = {}
            
            if start_date:
                where_conditions.append("target_date >= :start_date")
                params['start_date'] = start_date
            
            if end_date:
                where_conditions.append("target_date <= :end_date")
                params['end_date'] = end_date
            
            where_clause = "WHERE " + " AND ".join(where_conditions)
            
            with self.engine.connect() as conn:
                # 错误统计
                error_stats_query = f"""
                    SELECT 
                        data_source,
                        data_type,
                        COUNT(*) as error_count,
                        COUNT(DISTINCT error_message) as unique_errors,
                        AVG(retry_count) as avg_retries
                    FROM sync_task_status
                    {where_clause}
                    GROUP BY data_source, data_type
                    ORDER BY error_count DESC
                """
                
                error_stats = []
                result = conn.execute(text(error_stats_query), params)
                for row in result:
                    error_stats.append({
                        'data_source': row[0],
                        'data_type': row[1],
                        'error_count': row[2],
                        'unique_errors': row[3],
                        'avg_retries': float(row[4]) if row[4] else 0.0
                    })
                
                # 错误分类
                error_classification_query = f"""
                    SELECT 
                        error_message,
                        COUNT(*) as occurrence_count,
                        data_source,
                        data_type,
                        MAX(updated_at) as last_occurrence
                    FROM sync_task_status
                    {where_clause}
                    GROUP BY error_message, data_source, data_type
                    ORDER BY occurrence_count DESC
                    LIMIT :limit
                """
                
                params['limit'] = limit
                error_classification = []
                result = conn.execute(text(error_classification_query), params)
                for row in result:
                    error_classification.append({
                        'error_message': row[0],
                        'occurrence_count': row[1],
                        'data_source': row[2],
                        'data_type': row[3],
                        'last_occurrence': row[4].isoformat() if row[4] else None,
                        'error_category': self._categorize_error(row[0])
                    })
                
                # 错误趋势（按日期）
                error_trend_query = f"""
                    SELECT 
                        target_date,
                        COUNT(*) as daily_errors,
                        COUNT(DISTINCT data_source) as affected_sources
                    FROM sync_task_status
                    {where_clause}
                    GROUP BY target_date
                    ORDER BY target_date DESC
                    LIMIT 30
                """
                
                error_trend = []
                result = conn.execute(text(error_trend_query), params)
                for row in result:
                    error_trend.append({
                        'date': row[0].strftime('%Y-%m-%d') if row[0] else None,
                        'error_count': row[1],
                        'affected_sources': row[2]
                    })
                
                return {
                    'error_statistics': error_stats,
                    'error_classification': error_classification,
                    'error_trend': error_trend,
                    'analysis_period': {
                        'start_date': start_date,
                        'end_date': end_date
                    }
                }
                
        except Exception as e:
            logger.error(f"获取错误日志分析失败: {e}")
            return {
                'error_statistics': [],
                'error_classification': [],
                'error_trend': [],
                'error': str(e)
            }
    
    def _categorize_error(self, error_message: str) -> str:
        """
        错误分类
        
        Args:
            error_message: 错误信息
            
        Returns:
            错误类别
        """
        if not error_message:
            return "unknown"
        
        error_msg_lower = error_message.lower()
        
        # 网络相关错误
        if any(keyword in error_msg_lower for keyword in 
               ['connection', 'timeout', 'network', 'socket', 'dns']):
            return "network"
        
        # API相关错误
        if any(keyword in error_msg_lower for keyword in 
               ['api', 'rate limit', 'quota', 'permission', 'unauthorized']):
            return "api"
        
        # 数据库相关错误
        if any(keyword in error_msg_lower for keyword in 
               ['database', 'sql', 'constraint', 'duplicate', 'deadlock']):
            return "database"
        
        # 数据格式错误
        if any(keyword in error_msg_lower for keyword in 
               ['format', 'parse', 'json', 'xml', 'invalid data']):
            return "data_format"
        
        # 系统资源错误
        if any(keyword in error_msg_lower for keyword in 
               ['memory', 'disk', 'space', 'resource']):
            return "system_resource"
        
        return "other"
    
    def get_sync_event_history(self, task_id: str = None, event_type: str = None,
                              start_time: str = None, end_time: str = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取同步事件历史
        
        Args:
            task_id: 任务ID过滤
            event_type: 事件类型过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            
        Returns:
            事件历史列表
        """
        try:
            # 构建查询条件
            where_conditions = []
            params = {}
            
            if task_id:
                where_conditions.append("task_id = :task_id")
                params['task_id'] = task_id
            
            if event_type:
                where_conditions.append("event_type = :event_type")
                params['event_type'] = event_type
            
            if start_time:
                where_conditions.append("timestamp >= :start_time")
                params['start_time'] = start_time
            
            if end_time:
                where_conditions.append("timestamp <= :end_time")
                params['end_time'] = end_time
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            with self.engine.connect() as conn:
                params['limit'] = limit
                
                query = f"""
                    SELECT event_id, event_type, task_id, timestamp, data
                    FROM sync_event_log
                    {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """
                
                result = conn.execute(text(query), params)
                
                events = []
                for row in result:
                    events.append({
                        'event_id': row[0],
                        'event_type': row[1],
                        'task_id': row[2],
                        'timestamp': row[3].isoformat() if row[3] else None,
                        'data': json.loads(row[4]) if row[4] else {}
                    })
                
                return events
                
        except Exception as e:
            logger.error(f"获取同步事件历史失败: {e}")
            return []
    
    def cleanup_old_records(self, retention_days: int = None):
        """
        清理旧的历史记录
        
        Args:
            retention_days: 保留天数，默认从配置获取
        """
        if retention_days is None:
            retention_days = self.config.get('monitoring.sync_monitor.retention_days', 90)
        
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            with self.engine.connect() as conn:
                # 清理旧的任务状态记录（保留失败的记录更长时间）
                task_cleanup_result = conn.execute(text("""
                    DELETE FROM sync_task_status 
                    WHERE updated_at < :cutoff_date 
                    AND status IN ('completed', 'cancelled', 'skipped')
                """), {'cutoff_date': cutoff_date})
                
                # 清理旧的事件日志
                event_cleanup_result = conn.execute(text("""
                    DELETE FROM sync_event_log 
                    WHERE timestamp < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                # 清理旧的性能统计（保留更长时间）
                perf_cutoff_date = datetime.now() - timedelta(days=retention_days * 2)
                perf_cleanup_result = conn.execute(text("""
                    DELETE FROM sync_performance_stats 
                    WHERE created_at < :cutoff_date
                """), {'cutoff_date': perf_cutoff_date})
                
                conn.commit()
                
                logger.info(f"清理历史记录完成: "
                          f"任务记录 {task_cleanup_result.rowcount} 条, "
                          f"事件记录 {event_cleanup_result.rowcount} 条, "
                          f"性能记录 {perf_cleanup_result.rowcount} 条")
                
                return {
                    'success': True,
                    'cleaned_tasks': task_cleanup_result.rowcount,
                    'cleaned_events': event_cleanup_result.rowcount,
                    'cleaned_performance': perf_cleanup_result.rowcount,
                    'retention_days': retention_days
                }
                
        except Exception as e:
            logger.error(f"清理历史记录失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_sync_summary_report(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        获取同步摘要报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            摘要报告
        """
        try:
            # 默认查询最近7天
            if not start_date:
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            with self.engine.connect() as conn:
                # 总体统计
                overall_stats_query = text("""
                    SELECT 
                        COUNT(*) as total_tasks,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_tasks,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                        COUNT(CASE WHEN status = 'running' THEN 1 END) as running_tasks,
                        AVG(CASE WHEN duration IS NOT NULL THEN duration END) as avg_duration,
                        SUM(records_processed) as total_records_processed,
                        SUM(records_failed) as total_records_failed
                    FROM sync_task_status
                    WHERE target_date >= :start_date AND target_date <= :end_date
                """)
                
                result = conn.execute(overall_stats_query, {
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                row = result.fetchone()
                overall_stats = {
                    'total_tasks': row[0] or 0,
                    'successful_tasks': row[1] or 0,
                    'failed_tasks': row[2] or 0,
                    'running_tasks': row[3] or 0,
                    'avg_duration': float(row[4]) if row[4] else 0.0,
                    'total_records_processed': row[5] or 0,
                    'total_records_failed': row[6] or 0,
                    'success_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0.0
                }
                
                # 按数据源统计
                source_stats_query = text("""
                    SELECT 
                        data_source,
                        COUNT(*) as total_tasks,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_tasks,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                        AVG(CASE WHEN duration IS NOT NULL THEN duration END) as avg_duration
                    FROM sync_task_status
                    WHERE target_date >= :start_date AND target_date <= :end_date
                    GROUP BY data_source
                    ORDER BY total_tasks DESC
                """)
                
                result = conn.execute(source_stats_query, {
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                source_stats = []
                for row in result:
                    source_stats.append({
                        'data_source': row[0],
                        'total_tasks': row[1],
                        'successful_tasks': row[2],
                        'failed_tasks': row[3],
                        'avg_duration': float(row[4]) if row[4] else 0.0,
                        'success_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0.0
                    })
                
                # 按数据类型统计
                type_stats_query = text("""
                    SELECT 
                        data_type,
                        COUNT(*) as total_tasks,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_tasks,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                        AVG(CASE WHEN duration IS NOT NULL THEN duration END) as avg_duration
                    FROM sync_task_status
                    WHERE target_date >= :start_date AND target_date <= :end_date
                    GROUP BY data_type
                    ORDER BY total_tasks DESC
                """)
                
                result = conn.execute(type_stats_query, {
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                type_stats = []
                for row in result:
                    type_stats.append({
                        'data_type': row[0],
                        'total_tasks': row[1],
                        'successful_tasks': row[2],
                        'failed_tasks': row[3],
                        'avg_duration': float(row[4]) if row[4] else 0.0,
                        'success_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0.0
                    })
                
                # 每日趋势
                daily_trend_query = text("""
                    SELECT 
                        target_date,
                        COUNT(*) as total_tasks,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_tasks,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                        AVG(CASE WHEN duration IS NOT NULL THEN duration END) as avg_duration
                    FROM sync_task_status
                    WHERE target_date >= :start_date AND target_date <= :end_date
                    GROUP BY target_date
                    ORDER BY target_date
                """)
                
                result = conn.execute(daily_trend_query, {
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                daily_trend = []
                for row in result:
                    daily_trend.append({
                        'date': row[0].strftime('%Y-%m-%d') if row[0] else None,
                        'total_tasks': row[1],
                        'successful_tasks': row[2],
                        'failed_tasks': row[3],
                        'avg_duration': float(row[4]) if row[4] else 0.0,
                        'success_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0.0
                    })
                
                return {
                    'report_period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'overall_statistics': overall_stats,
                    'source_statistics': source_stats,
                    'type_statistics': type_stats,
                    'daily_trend': daily_trend,
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"获取同步摘要报告失败: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }

    # ================================
    # 性能分析功能 (Task 7.3)
    # ================================
    
    def calculate_performance_metrics(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        计算同步性能指标
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            性能指标
        """
        try:
            # 默认分析最近24小时
            if not start_date:
                start_date = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with self.engine.connect() as conn:
                # 基础性能指标
                basic_metrics_query = text("""
                    SELECT 
                        COUNT(*) as total_tasks,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                        COUNT(CASE WHEN status = 'running' THEN 1 END) as running_tasks,
                        AVG(CASE WHEN duration IS NOT NULL AND duration > 0 THEN duration END) as avg_duration,
                        MIN(CASE WHEN duration IS NOT NULL AND duration > 0 THEN duration END) as min_duration,
                        MAX(CASE WHEN duration IS NOT NULL AND duration > 0 THEN duration END) as max_duration,
                        STDDEV(CASE WHEN duration IS NOT NULL AND duration > 0 THEN duration END) as duration_stddev,
                        SUM(records_processed) as total_records_processed,
                        SUM(records_failed) as total_records_failed,
                        AVG(records_processed) as avg_records_per_task,
                        COUNT(CASE WHEN retry_count > 0 THEN 1 END) as tasks_with_retries,
                        AVG(retry_count) as avg_retry_count
                    FROM sync_task_status
                    WHERE updated_at >= :start_date AND updated_at <= :end_date
                """)
                
                result = conn.execute(basic_metrics_query, {
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                row = result.fetchone()
                basic_metrics = {
                    'total_tasks': row[0] or 0,
                    'completed_tasks': row[1] or 0,
                    'failed_tasks': row[2] or 0,
                    'running_tasks': row[3] or 0,
                    'success_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0.0,
                    'failure_rate': (row[2] / row[0] * 100) if row[0] > 0 else 0.0,
                    'avg_duration_seconds': float(row[4]) if row[4] else 0.0,
                    'min_duration_seconds': float(row[5]) if row[5] else 0.0,
                    'max_duration_seconds': float(row[6]) if row[6] else 0.0,
                    'duration_stddev': float(row[7]) if row[7] else 0.0,
                    'total_records_processed': row[8] or 0,
                    'total_records_failed': row[9] or 0,
                    'avg_records_per_task': float(row[10]) if row[10] else 0.0,
                    'tasks_with_retries': row[11] or 0,
                    'avg_retry_count': float(row[12]) if row[12] else 0.0,
                    'retry_rate': (row[11] / row[0] * 100) if row[0] > 0 else 0.0
                }
                
                # 计算吞吐量（任务/小时）
                time_diff = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
                hours = time_diff.total_seconds() / 3600
                basic_metrics['throughput_tasks_per_hour'] = basic_metrics['completed_tasks'] / hours if hours > 0 else 0.0
                basic_metrics['throughput_records_per_hour'] = basic_metrics['total_records_processed'] / hours if hours > 0 else 0.0
                
                # 数据质量指标
                if basic_metrics['total_records_processed'] > 0:
                    basic_metrics['data_quality_rate'] = (
                        (basic_metrics['total_records_processed'] - basic_metrics['total_records_failed']) /
                        basic_metrics['total_records_processed'] * 100
                    )
                else:
                    basic_metrics['data_quality_rate'] = 100.0
                
                return basic_metrics
                
        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            return {'error': str(e)}
    
    def analyze_performance_trends(self, days: int = 7, granularity: str = 'hour') -> Dict[str, Any]:
        """
        分析性能趋势
        
        Args:
            days: 分析天数
            granularity: 粒度 ('hour', 'day')
            
        Returns:
            趋势分析结果
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # 根据粒度选择时间分组
            if granularity == 'hour':
                time_format = "DATE_TRUNC('hour', updated_at)"
                time_label = 'hour'
            else:
                time_format = "DATE_TRUNC('day', updated_at)"
                time_label = 'day'
            
            with self.engine.connect() as conn:
                trend_query = text(f"""
                    SELECT 
                        {time_format} as time_period,
                        COUNT(*) as total_tasks,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                        AVG(CASE WHEN duration IS NOT NULL AND duration > 0 THEN duration END) as avg_duration,
                        SUM(records_processed) as total_records,
                        COUNT(CASE WHEN retry_count > 0 THEN 1 END) as tasks_with_retries
                    FROM sync_task_status
                    WHERE updated_at >= :start_time AND updated_at <= :end_time
                    GROUP BY {time_format}
                    ORDER BY time_period
                """)
                
                result = conn.execute(trend_query, {
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                trend_data = []
                for row in result:
                    period_data = {
                        'time_period': row[0].isoformat() if row[0] else None,
                        'total_tasks': row[1] or 0,
                        'completed_tasks': row[2] or 0,
                        'failed_tasks': row[3] or 0,
                        'success_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0.0,
                        'avg_duration': float(row[4]) if row[4] else 0.0,
                        'total_records': row[5] or 0,
                        'tasks_with_retries': row[6] or 0,
                        'retry_rate': (row[6] / row[1] * 100) if row[1] > 0 else 0.0
                    }
                    trend_data.append(period_data)
                
                # 计算趋势指标
                trend_analysis = self._calculate_trend_indicators(trend_data)
                
                return {
                    'analysis_period': {
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'days': days,
                        'granularity': granularity
                    },
                    'trend_data': trend_data,
                    'trend_analysis': trend_analysis
                }
                
        except Exception as e:
            logger.error(f"分析性能趋势失败: {e}")
            return {'error': str(e)}
    
    def _calculate_trend_indicators(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算趋势指标
        
        Args:
            trend_data: 趋势数据
            
        Returns:
            趋势指标
        """
        if len(trend_data) < 2:
            return {'insufficient_data': True}
        
        try:
            # 提取关键指标序列
            success_rates = [d['success_rate'] for d in trend_data if d['total_tasks'] > 0]
            avg_durations = [d['avg_duration'] for d in trend_data if d['avg_duration'] > 0]
            total_tasks = [d['total_tasks'] for d in trend_data]
            
            # 计算趋势方向
            def calculate_trend(values):
                if len(values) < 2:
                    return 'stable'
                
                # 简单线性趋势计算
                x = list(range(len(values)))
                n = len(values)
                sum_x = sum(x)
                sum_y = sum(values)
                sum_xy = sum(x[i] * values[i] for i in range(n))
                sum_x2 = sum(x[i] ** 2 for i in range(n))
                
                # 计算斜率
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                
                if slope > 0.1:
                    return 'improving'
                elif slope < -0.1:
                    return 'declining'
                else:
                    return 'stable'
            
            trend_analysis = {
                'success_rate_trend': calculate_trend(success_rates) if success_rates else 'no_data',
                'duration_trend': calculate_trend(avg_durations) if avg_durations else 'no_data',
                'volume_trend': calculate_trend(total_tasks) if total_tasks else 'no_data',
                'current_success_rate': success_rates[-1] if success_rates else 0.0,
                'avg_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0.0,
                'current_avg_duration': avg_durations[-1] if avg_durations else 0.0,
                'avg_duration_overall': sum(avg_durations) / len(avg_durations) if avg_durations else 0.0,
                'total_periods': len(trend_data),
                'active_periods': len([d for d in trend_data if d['total_tasks'] > 0])
            }
            
            # 计算变异系数（稳定性指标）
            if success_rates and len(success_rates) > 1:
                mean_success = trend_analysis['avg_success_rate']
                variance = sum((x - mean_success) ** 2 for x in success_rates) / len(success_rates)
                std_dev = variance ** 0.5
                trend_analysis['success_rate_stability'] = 1 - (std_dev / mean_success) if mean_success > 0 else 0
            else:
                trend_analysis['success_rate_stability'] = 1.0
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"计算趋势指标失败: {e}")
            return {'error': str(e)}
    
    def generate_performance_forecast(self, forecast_hours: int = 24) -> Dict[str, Any]:
        """
        生成性能预测
        
        Args:
            forecast_hours: 预测小时数
            
        Returns:
            性能预测结果
        """
        try:
            # 获取历史数据用于预测
            historical_data = self.analyze_performance_trends(days=7, granularity='hour')
            
            if 'error' in historical_data or not historical_data.get('trend_data'):
                return {'error': '缺少历史数据进行预测'}
            
            trend_data = historical_data['trend_data']
            
            # 简单的线性预测模型
            def predict_metric(values, periods_ahead):
                if len(values) < 3:
                    return values[-1] if values else 0
                
                # 使用最近的趋势进行预测
                recent_values = values[-min(24, len(values)):]  # 最近24个数据点
                
                # 计算平均变化率
                changes = [recent_values[i] - recent_values[i-1] for i in range(1, len(recent_values))]
                avg_change = sum(changes) / len(changes) if changes else 0
                
                # 预测值
                last_value = recent_values[-1]
                predicted_value = last_value + (avg_change * periods_ahead)
                
                # 确保预测值在合理范围内
                if 'rate' in str(values):  # 对于百分比指标
                    predicted_value = max(0, min(100, predicted_value))
                elif predicted_value < 0:  # 对于其他指标，不能为负
                    predicted_value = 0
                
                return predicted_value
            
            # 提取指标序列
            success_rates = [d['success_rate'] for d in trend_data]
            avg_durations = [d['avg_duration'] for d in trend_data]
            task_volumes = [d['total_tasks'] for d in trend_data]
            
            # 生成预测
            forecast_periods = list(range(1, forecast_hours + 1))
            forecast_data = []
            
            for period in forecast_periods:
                forecast_time = datetime.now() + timedelta(hours=period)
                
                predicted_success_rate = predict_metric(success_rates, period)
                predicted_duration = predict_metric(avg_durations, period)
                predicted_volume = predict_metric(task_volumes, period)
                
                forecast_data.append({
                    'forecast_time': forecast_time.isoformat(),
                    'predicted_success_rate': round(predicted_success_rate, 2),
                    'predicted_avg_duration': round(predicted_duration, 2),
                    'predicted_task_volume': round(predicted_volume, 0),
                    'confidence_level': self._calculate_confidence_level(period, len(trend_data))
                })
            
            # 生成预测摘要
            forecast_summary = {
                'forecast_period_hours': forecast_hours,
                'avg_predicted_success_rate': round(sum(f['predicted_success_rate'] for f in forecast_data) / len(forecast_data), 2),
                'avg_predicted_duration': round(sum(f['predicted_avg_duration'] for f in forecast_data) / len(forecast_data), 2),
                'total_predicted_tasks': sum(f['predicted_task_volume'] for f in forecast_data),
                'trend_direction': historical_data.get('trend_analysis', {}).get('success_rate_trend', 'unknown'),
                'forecast_reliability': 'high' if len(trend_data) >= 24 else 'medium' if len(trend_data) >= 12 else 'low'
            }
            
            return {
                'forecast_generated_at': datetime.now().isoformat(),
                'forecast_summary': forecast_summary,
                'forecast_data': forecast_data,
                'historical_basis': {
                    'data_points_used': len(trend_data),
                    'historical_period_days': 7
                }
            }
            
        except Exception as e:
            logger.error(f"生成性能预测失败: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_level(self, forecast_period: int, historical_data_points: int) -> str:
        """
        计算预测置信度
        
        Args:
            forecast_period: 预测周期
            historical_data_points: 历史数据点数量
            
        Returns:
            置信度等级
        """
        # 基于历史数据量和预测距离计算置信度
        data_score = min(1.0, historical_data_points / 48)  # 48小时数据为满分
        distance_score = max(0.1, 1.0 - (forecast_period / 24))  # 24小时后置信度降低
        
        confidence_score = (data_score + distance_score) / 2
        
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def export_performance_report(self, start_date: str = None, end_date: str = None,
                                 format: str = 'json') -> Dict[str, Any]:
        """
        导出性能报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            format: 导出格式 ('json', 'csv')
            
        Returns:
            导出结果
        """
        try:
            # 收集所有性能数据
            performance_metrics = self.calculate_performance_metrics(start_date, end_date)
            trend_analysis = self.analyze_performance_trends(days=7)
            forecast_data = self.generate_performance_forecast(forecast_hours=24)
            summary_report = self.get_sync_summary_report(start_date, end_date)
            error_analysis = self.get_error_log_analysis(start_date, end_date)
            
            report_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'format': format
                },
                'performance_metrics': performance_metrics,
                'trend_analysis': trend_analysis,
                'forecast': forecast_data,
                'summary_report': summary_report,
                'error_analysis': error_analysis
            }
            
            if format == 'json':
                return {
                    'success': True,
                    'data': report_data,
                    'export_format': 'json'
                }
            elif format == 'csv':
                # 转换为CSV格式的数据结构
                csv_data = self._convert_to_csv_format(report_data)
                return {
                    'success': True,
                    'data': csv_data,
                    'export_format': 'csv'
                }
            else:
                return {
                    'success': False,
                    'error': f'不支持的导出格式: {format}'
                }
                
        except Exception as e:
            logger.error(f"导出性能报告失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _convert_to_csv_format(self, report_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        转换报告数据为CSV格式
        
        Args:
            report_data: 报告数据
            
        Returns:
            CSV格式数据
        """
        csv_data = {}
        
        try:
            # 性能指标表
            if 'performance_metrics' in report_data:
                metrics = report_data['performance_metrics']
                csv_data['performance_metrics'] = [{
                    'metric': k,
                    'value': v
                } for k, v in metrics.items() if not isinstance(v, dict)]
            
            # 趋势数据表
            if 'trend_analysis' in report_data and 'trend_data' in report_data['trend_analysis']:
                csv_data['trend_data'] = report_data['trend_analysis']['trend_data']
            
            # 预测数据表
            if 'forecast' in report_data and 'forecast_data' in report_data['forecast']:
                csv_data['forecast_data'] = report_data['forecast']['forecast_data']
            
            # 错误统计表
            if 'error_analysis' in report_data and 'error_statistics' in report_data['error_analysis']:
                csv_data['error_statistics'] = report_data['error_analysis']['error_statistics']
            
            return csv_data
            
        except Exception as e:
            logger.error(f"转换CSV格式失败: {e}")
            return {}
    
    def get_performance_alerts(self, thresholds: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        获取性能告警
        
        Args:
            thresholds: 告警阈值配置
            
        Returns:
            告警列表
        """
        if thresholds is None:
            thresholds = {
                'min_success_rate': 90.0,  # 最低成功率
                'max_avg_duration': 300.0,  # 最大平均耗时（秒）
                'max_failure_rate': 10.0,  # 最大失败率
                'max_retry_rate': 20.0,  # 最大重试率
                'min_throughput': 10.0  # 最小吞吐量（任务/小时）
            }
        
        try:
            alerts = []
            
            # 获取最近1小时的性能指标
            current_metrics = self.calculate_performance_metrics(
                start_date=(datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                end_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            if 'error' in current_metrics:
                return []
            
            # 检查各项指标
            if current_metrics['success_rate'] < thresholds['min_success_rate']:
                alerts.append({
                    'alert_type': 'low_success_rate',
                    'severity': 'high',
                    'message': f"成功率过低: {current_metrics['success_rate']:.1f}% < {thresholds['min_success_rate']}%",
                    'current_value': current_metrics['success_rate'],
                    'threshold': thresholds['min_success_rate'],
                    'timestamp': datetime.now().isoformat()
                })
            
            if current_metrics['avg_duration_seconds'] > thresholds['max_avg_duration']:
                alerts.append({
                    'alert_type': 'high_duration',
                    'severity': 'medium',
                    'message': f"平均耗时过长: {current_metrics['avg_duration_seconds']:.1f}s > {thresholds['max_avg_duration']}s",
                    'current_value': current_metrics['avg_duration_seconds'],
                    'threshold': thresholds['max_avg_duration'],
                    'timestamp': datetime.now().isoformat()
                })
            
            if current_metrics['failure_rate'] > thresholds['max_failure_rate']:
                alerts.append({
                    'alert_type': 'high_failure_rate',
                    'severity': 'high',
                    'message': f"失败率过高: {current_metrics['failure_rate']:.1f}% > {thresholds['max_failure_rate']}%",
                    'current_value': current_metrics['failure_rate'],
                    'threshold': thresholds['max_failure_rate'],
                    'timestamp': datetime.now().isoformat()
                })
            
            if current_metrics['retry_rate'] > thresholds['max_retry_rate']:
                alerts.append({
                    'alert_type': 'high_retry_rate',
                    'severity': 'medium',
                    'message': f"重试率过高: {current_metrics['retry_rate']:.1f}% > {thresholds['max_retry_rate']}%",
                    'current_value': current_metrics['retry_rate'],
                    'threshold': thresholds['max_retry_rate'],
                    'timestamp': datetime.now().isoformat()
                })
            
            if current_metrics['throughput_tasks_per_hour'] < thresholds['min_throughput']:
                alerts.append({
                    'alert_type': 'low_throughput',
                    'severity': 'medium',
                    'message': f"吞吐量过低: {current_metrics['throughput_tasks_per_hour']:.1f} < {thresholds['min_throughput']} 任务/小时",
                    'current_value': current_metrics['throughput_tasks_per_hour'],
                    'threshold': thresholds['min_throughput'],
                    'timestamp': datetime.now().isoformat()
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"获取性能告警失败: {e}")
            return []

    # ================================
    # 数据库操作
    # ================================
    
    def _save_task_status(self, task_info: SyncTaskInfo):
        """保存任务状态到数据库"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO sync_task_status (
                        task_id, data_source, data_type, target_date, status,
                        priority, start_time, end_time, duration, progress,
                        records_processed, records_failed, error_message,
                        retry_count, max_retries, created_at, updated_at
                    ) VALUES (
                        :task_id, :data_source, :data_type, :target_date, :status,
                        :priority, :start_time, :end_time, :duration, :progress,
                        :records_processed, :records_failed, :error_message,
                        :retry_count, :max_retries, :created_at, :updated_at
                    )
                    ON CONFLICT (task_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        start_time = EXCLUDED.start_time,
                        end_time = EXCLUDED.end_time,
                        duration = EXCLUDED.duration,
                        progress = EXCLUDED.progress,
                        records_processed = EXCLUDED.records_processed,
                        records_failed = EXCLUDED.records_failed,
                        error_message = EXCLUDED.error_message,
                        retry_count = EXCLUDED.retry_count,
                        updated_at = EXCLUDED.updated_at
                """), {
                    'task_id': task_info.task_id,
                    'data_source': task_info.data_source,
                    'data_type': task_info.data_type,
                    'target_date': task_info.target_date,
                    'status': task_info.status.value,
                    'priority': task_info.priority,
                    'start_time': task_info.start_time,
                    'end_time': task_info.end_time,
                    'duration': task_info.duration,
                    'progress': task_info.progress,
                    'records_processed': task_info.records_processed,
                    'records_failed': task_info.records_failed,
                    'error_message': task_info.error_message,
                    'retry_count': task_info.retry_count,
                    'max_retries': task_info.max_retries,
                    'created_at': task_info.created_at,
                    'updated_at': task_info.updated_at
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存任务状态失败 {task_info.task_id}: {e}")
    
    def _update_task_progress_in_db(self, task_id: str, progress: float, 
                                  records_processed: int, records_failed: int):
        """更新数据库中的任务进度"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE sync_task_status 
                    SET progress = :progress,
                        records_processed = :records_processed,
                        records_failed = :records_failed,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = :task_id
                """), {
                    'task_id': task_id,
                    'progress': progress,
                    'records_processed': records_processed,
                    'records_failed': records_failed
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"更新任务进度失败 {task_id}: {e}")
    
    def _save_event_to_db(self, event: SyncEvent):
        """保存事件到数据库"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO sync_event_log (
                        event_id, event_type, task_id, timestamp, data
                    ) VALUES (
                        :event_id, :event_type, :task_id, :timestamp, :data
                    )
                """), {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'task_id': event.task_id,
                    'timestamp': event.timestamp,
                    'data': json.dumps(event.data)
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"保存事件失败 {event.event_id}: {e}")


# 全局同步监控器实例（延迟初始化）
sync_monitor = None

def get_sync_monitor():
    """获取同步监控器实例（单例模式）"""
    global sync_monitor
    if sync_monitor is None:
        sync_monitor = SyncMonitor()
    return sync_monitor


if __name__ == '__main__':
    # 测试代码
    print("测试同步监控器...")
    
    # 创建测试任务
    task_info = SyncTaskInfo(
        task_id="test_task_001",
        data_source="tushare",
        data_type="daily",
        target_date="2024-01-01",
        status=SyncTaskStatus.PENDING,
        priority=1
    )
    
    # 开始监控
    sync_monitor.start_task_monitoring(task_info)
    
    # 模拟进度更新
    import time
    for i in range(0, 101, 20):
        sync_monitor.update_task_progress(
            task_id="test_task_001",
            progress=i,
            records_processed=i * 10,
            records_failed=i // 10
        )
        time.sleep(1)
    
    # 完成任务
    sync_monitor.complete_task_monitoring(
        task_id="test_task_001",
        success=True,
        final_stats={
            'records_processed': 1000,
            'records_failed': 10
        }
    )
    
    # 获取仪表板数据
    dashboard = sync_monitor.get_real_time_dashboard()
    print("仪表板数据:", json.dumps(dashboard, indent=2, ensure_ascii=False))