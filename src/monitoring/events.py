import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List

from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件处理系统

实现Observer模式，提供可扩展的事件通知机制
"""


from .models import SyncEvent, SyncEventType


class EventObserver(ABC):
    """事件观察者接口"""

    @abstractmethod
    def handle_event(self, event: SyncEvent) -> None:
        """处理事件"""
        pass


class EventPublisher:
    """事件发布者"""

    def __init__(self):
        """方法描述"""
        self._global_observers: List[EventObserver] = []
        self._lock = threading.RLock()

    def subscribe(self, event_type: SyncEventType, observer: EventObserver) -> None:
        """订阅特定类型事件"""
        with self._lock:
            self._observers[event_type].append(observer)
            logger.debug(f"观察者 {observer.__class__.__name__} 订阅事件类型 {event_type.value}")

    def subscribe_all(self, observer: EventObserver) -> None:
        """订阅所有事件"""
        with self._lock:
            self._global_observers.append(observer)
            logger.debug(f"观察者 {observer.__class__.__name__} 订阅所有事件")

    def unsubscribe(self, event_type: SyncEventType, observer: EventObserver) -> None:
        """取消订阅特定类型事件"""
        with self._lock:
            if observer in self._observers[event_type]:
                self._observers[event_type].remove(observer)
                logger.debug(f"观察者 {observer.__class__.__name__} 取消订阅事件类型 {event_type.value}")

    def unsubscribe_all(self, observer: EventObserver) -> None:
        """取消订阅所有事件"""
        with self._lock:
            if observer in self._global_observers:
                self._global_observers.remove(observer)

            # 从所有特定事件类型中移除
            for observers in self._observers.values():
                if observer in observers:
                    observers.remove(observer)

            logger.debug(f"观察者 {observer.__class__.__name__} 取消订阅所有事件")

    def publish(self, event: SyncEvent) -> None:
        """发布事件"""
        with self._lock:
            # 通知全局观察者
            for observer in self._global_observers:
                self._notify_observer(observer, event)

            # 通知特定类型观察者
            for observer in self._observers[event.event_type]:
                self._notify_observer(observer, event)

    def _notify_observer(self, observer: EventObserver, event: SyncEvent) -> None:
        """通知单个观察者"""
        try:
            observer.handle_event(event)
        except Exception as e:
            logger.error(f"观察者 {observer.__class__.__name__} 处理事件失败: {e}")


class DatabaseEventObserver(EventObserver):
    """数据库事件观察者"""

    def __init__(self, event_repository):
        """方法描述"""

    def handle_event(self, event: SyncEvent) -> None:
        """保存事件到数据库"""
        self.event_repository.save_event(event)


class LoggingEventObserver(EventObserver):
    """日志事件观察者"""

    def handle_event(self, event: SyncEvent) -> None:
        """记录事件到日志"""
        logger.info(f"事件: {event.event_type.value} | 任务: {event.task_id} | 数据: {event.data}")


class MetricsEventObserver(EventObserver):
    """指标统计事件观察者"""

    def __init__(self):
        """方法描述"""
        self._lock = threading.Lock()

    def handle_event(self, event: SyncEvent) -> None:
        """更新指标统计"""
        with self._lock:
            self.metrics[f"event_{event.event_type.value}"] += 1

            if event.event_type == SyncEventType.TASK_COMPLETED:
                self.metrics["tasks_completed"] += 1
            elif event.event_type == SyncEventType.TASK_FAILED:
                self.metrics["tasks_failed"] += 1

    def get_metrics(self) -> Dict[str, int]:
        """获取指标统计"""
        with self._lock:
            return dict(self.metrics)


class WebSocketEventObserver(EventObserver):
    """WebSocket事件观察者"""

    def __init__(self):
        """方法描述"""
        self._lock = threading.Lock()

    def add_connection(self, websocket) -> None:
        """添加WebSocket连接"""
        with self._lock:
            self.connections.add(websocket)

    def remove_connection(self, websocket) -> None:
        """移除WebSocket连接"""
        with self._lock:
            self.connections.discard(websocket)

    def handle_event(self, event: SyncEvent) -> None:
        """向所有WebSocket连接发送事件"""
        with self._lock:
            disconnected = set()
            for websocket in self.connections:
                try:
                    # 这里需要异步发送，实际实现时需要使用asyncio
                    # websocket.send(json.dumps(event.to_dict()))
                    pass
                except Exception as e:
                    logger.warning(f"WebSocket发送失败: {e}")
                    disconnected.add(websocket)

            # 清理断开的连接
            self.connections -= disconnected


class CallbackEventObserver(EventObserver):
    """回调函数事件观察者"""

    def __init__(self, callback: Callable[[SyncEvent], None]):
        """方法描述"""

    def handle_event(self, event: SyncEvent) -> None:
        """执行回调函数"""
        self.callback(event)
