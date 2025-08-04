#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统数据模型

定义同步任务、事件等数据结构
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


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
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        
        # 转换datetime为ISO格式字符串
        datetime_fields = ['start_time', 'end_time', 'created_at', 'updated_at']
        for field in datetime_fields:
            if data[field]:
                data[field] = data[field].isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncTaskInfo':
        """从字典创建实例"""
        # 转换状态枚举
        if isinstance(data['status'], str):
            data['status'] = SyncTaskStatus(data['status'])
        
        # 转换datetime字段
        datetime_fields = ['start_time', 'end_time', 'created_at', 'updated_at']
        for field in datetime_fields:
            if data.get(field) and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    def is_active(self) -> bool:
        """判断任务是否活跃"""
        return self.status in [SyncTaskStatus.PENDING, SyncTaskStatus.RUNNING]
    
    def is_completed(self) -> bool:
        """判断任务是否已完成"""
        return self.status in [SyncTaskStatus.COMPLETED, SyncTaskStatus.FAILED, SyncTaskStatus.CANCELLED]
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        total_records = self.records_processed + self.records_failed
        if total_records == 0:
            return 0.0
        return (self.records_processed / total_records) * 100
    
    def get_estimated_remaining_time(self) -> Optional[float]:
        """估算剩余时间（秒）"""
        if not self.start_time or self.progress <= 0:
            return None
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if self.progress >= 100:
            return 0.0
        
        estimated_total_time = elapsed_time / (self.progress / 100)
        return estimated_total_time - elapsed_time


@dataclass
class SyncEvent:
    """同步事件"""
    event_id: str
    event_type: SyncEventType
    task_id: str
    timestamp: datetime
    data: Dict[str, Any]
    
    def __post_init__(self):
        """初始化后处理"""
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncEvent':
        """从字典创建实例"""
        # 转换事件类型枚举
        if isinstance(data['event_type'], str):
            data['event_type'] = SyncEventType(data['event_type'])
        
        # 转换时间戳
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    throughput: float = 0.0  # 任务/小时
    error_rate: float = 0.0
    total_records: int = 0
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    def calculate_success_rate(self) -> float:
        """计算成功率"""
        if self.total_tasks == 0:
            return 0.0
        return (self.successful_tasks / self.total_tasks) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['success_rate'] = self.calculate_success_rate()
        return data


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    description: str
    condition: str  # 条件表达式
    severity: str  # low, medium, high, critical
    enabled: bool = True
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    rule_name: str
    message: str
    severity: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())
        if not self.triggered_at:
            self.triggered_at = datetime.now()
    
    def is_resolved(self) -> bool:
        """判断告警是否已解决"""
        return self.resolved_at is not None
    
    def resolve(self) -> None:
        """解决告警"""
        self.resolved_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['triggered_at'] = self.triggered_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data