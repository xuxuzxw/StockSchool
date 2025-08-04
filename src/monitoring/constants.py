#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步监控常量配置
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class MonitoringConstants:
    """监控系统常量"""
    
    # 事件队列配置
    MAX_EVENT_QUEUE_SIZE: int = 1000
    DEFAULT_EVENT_RETENTION_HOURS: int = 24
    DEFAULT_PERFORMANCE_WINDOW_HOURS: int = 1
    
    # 任务配置
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_PRIORITY: int = 1
    
    # 数据库配置
    BATCH_SIZE: int = 1000
    CONNECTION_TIMEOUT: int = 30
    
    # 性能配置
    DASHBOARD_CACHE_TTL: int = 60  # 秒
    RECENT_EVENTS_LIMIT: int = 50
    
    # 监控阈值
    HIGH_ERROR_RATE_THRESHOLD: float = 0.1  # 10%
    SLOW_TASK_THRESHOLD: float = 300.0  # 5分钟
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MonitoringConstants':
        """从配置创建常量实例"""
        return cls(
            MAX_EVENT_QUEUE_SIZE=config.get('monitoring.max_event_queue_size', cls.MAX_EVENT_QUEUE_SIZE),
            DEFAULT_EVENT_RETENTION_HOURS=config.get('monitoring.event_retention_hours', cls.DEFAULT_EVENT_RETENTION_HOURS),
            DEFAULT_PERFORMANCE_WINDOW_HOURS=config.get('monitoring.performance_window_hours', cls.DEFAULT_PERFORMANCE_WINDOW_HOURS),
            DEFAULT_MAX_RETRIES=config.get('monitoring.default_max_retries', cls.DEFAULT_MAX_RETRIES),
            DEFAULT_PRIORITY=config.get('monitoring.default_priority', cls.DEFAULT_PRIORITY),
            BATCH_SIZE=config.get('monitoring.batch_size', cls.BATCH_SIZE),
            CONNECTION_TIMEOUT=config.get('monitoring.connection_timeout', cls.CONNECTION_TIMEOUT),
            DASHBOARD_CACHE_TTL=config.get('monitoring.dashboard_cache_ttl', cls.DASHBOARD_CACHE_TTL),
            RECENT_EVENTS_LIMIT=config.get('monitoring.recent_events_limit', cls.RECENT_EVENTS_LIMIT),
            HIGH_ERROR_RATE_THRESHOLD=config.get('monitoring.high_error_rate_threshold', cls.HIGH_ERROR_RATE_THRESHOLD),
            SLOW_TASK_THRESHOLD=config.get('monitoring.slow_task_threshold', cls.SLOW_TASK_THRESHOLD)
        )


# SQL查询常量
class SQLQueries:
    """SQL查询常量"""
    
    CREATE_TASK_STATUS_TABLE = """
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
    """
    
    CREATE_EVENT_LOG_TABLE = """
        CREATE TABLE IF NOT EXISTS sync_event_log (
            id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
            event_id VARCHAR(100) UNIQUE NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            task_id VARCHAR(100) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    
    CREATE_PERFORMANCE_STATS_TABLE = """
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
    """
    
    # 索引创建语句
    INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_sync_task_status_source_type ON sync_task_status(data_source, data_type)",
        "CREATE INDEX IF NOT EXISTS idx_sync_task_status_date ON sync_task_status(target_date)",
        "CREATE INDEX IF NOT EXISTS idx_sync_event_log_timestamp ON sync_event_log(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_sync_event_log_task_id ON sync_event_log(task_id)",
        "CREATE INDEX IF NOT EXISTS idx_sync_performance_stats_date_hour ON sync_performance_stats(stat_date, stat_hour)"
    ]