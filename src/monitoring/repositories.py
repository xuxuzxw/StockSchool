import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Engine, text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步监控数据访问层

实现Repository模式，分离数据访问逻辑
"""


from .models import SyncEvent, SyncTaskInfo


class SyncTaskRepository(ABC):
    """同步任务数据访问接口"""

    @abstractmethod
    def save_task(self, task_info: SyncTaskInfo) -> bool:
        """方法描述"""

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[SyncTaskInfo]:
        """方法描述"""

    @abstractmethod
    def update_progress(self, task_id: str, progress: float, records_processed: int, records_failed: int) -> bool:
        pass

    @abstractmethod
    def get_tasks_by_criteria(self, **criteria) -> List[SyncTaskInfo]:
        """方法描述"""


class PostgresSyncTaskRepository(SyncTaskRepository):
    """PostgreSQL同步任务数据访问实现"""

    def __init__(self, engine: Engine):
        """方法描述"""

    def save_task(self, task_info: SyncTaskInfo) -> bool:
        """保存任务信息"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(
                        """
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
                """
                    ),
                    self._task_to_dict(task_info),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"保存任务失败 {task_info.task_id}: {e}")
            return False

    def get_task(self, task_id: str) -> Optional[SyncTaskInfo]:
        """获取任务信息"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT task_id, data_source, data_type, target_date, status,
                           priority, start_time, end_time, duration, progress,
                           records_processed, records_failed, error_message,
                           retry_count, max_retries, created_at, updated_at
                    FROM sync_task_status
                    WHERE task_id = :task_id
                """
                    ),
                    {"task_id": task_id},
                )

                row = result.fetchone()
                return self._row_to_task(row) if row else None
        except Exception as e:
            logger.error(f"获取任务失败 {task_id}: {e}")
            return None

    def update_progress(self, task_id: str, progress: float, records_processed: int, records_failed: int) -> bool:
        """更新任务进度"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    UPDATE sync_task_status
                    SET progress = :progress,
                        records_processed = :records_processed,
                        records_failed = :records_failed,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = :task_id
                """
                    ),
                    {
                        "task_id": task_id,
                        "progress": progress,
                        "records_processed": records_processed,
                        "records_failed": records_failed,
                    },
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"更新任务进度失败 {task_id}: {e}")
            return False

    def _task_to_dict(self, task_info: SyncTaskInfo) -> Dict[str, Any]:
        """转换任务信息为字典"""
        return {
            "task_id": task_info.task_id,
            "data_source": task_info.data_source,
            "data_type": task_info.data_type,
            "target_date": task_info.target_date,
            "status": task_info.status.value,
            "priority": task_info.priority,
            "start_time": task_info.start_time,
            "end_time": task_info.end_time,
            "duration": task_info.duration,
            "progress": task_info.progress,
            "records_processed": task_info.records_processed,
            "records_failed": task_info.records_failed,
            "error_message": task_info.error_message,
            "retry_count": task_info.retry_count,
            "max_retries": task_info.max_retries,
            "created_at": task_info.created_at,
            "updated_at": task_info.updated_at,
        }

    def _row_to_task(self, row) -> SyncTaskInfo:
        """转换数据库行为任务信息"""
        from .models import SyncTaskStatus

        return SyncTaskInfo(
            task_id=row[0],
            data_source=row[1],
            data_type=row[2],
            target_date=row[3].strftime("%Y-%m-%d") if row[3] else None,
            status=SyncTaskStatus(row[4]),
            priority=row[5],
            start_time=row[6],
            end_time=row[7],
            duration=float(row[8]) if row[8] else None,
            progress=float(row[9]) if row[9] else 0.0,
            records_processed=row[10] or 0,
            records_failed=row[11] or 0,
            error_message=row[12],
            retry_count=row[13] or 0,
            max_retries=row[14] or 3,
            created_at=row[15],
            updated_at=row[16],
        )


class SyncEventRepository:
    """同步事件数据访问"""

    def __init__(self, engine: Engine):
        """方法描述"""

    def save_event(self, event: SyncEvent) -> bool:
        """保存事件"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO sync_event_log (
                        event_id, event_type, task_id, timestamp, data
                    ) VALUES (
                        :event_id, :event_type, :task_id, :timestamp, :data
                    )
                """
                    ),
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "task_id": event.task_id,
                        "timestamp": event.timestamp,
                        "data": json.dumps(event.data),
                    },
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"保存事件失败 {event.event_id}: {e}")
            return False
