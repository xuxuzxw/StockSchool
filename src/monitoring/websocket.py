#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控WebSocket模块

提供实时监控数据推送功能
包括系统健康状态、告警通知、性能指标等实时数据

作者: StockSchool Team
创建时间: 2025-01-02
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.monitoring.collectors import collect_system_health, CollectorConfig
from src.monitoring.alerts import AlertEngine, Alert
from src.monitoring.performance import PerformanceMonitor

# 配置日志
logger = logging.getLogger(__name__)

class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""
    type: str
    data: Any
    timestamp: datetime = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        super().__init__(**data)

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 活跃连接
        self.active_connections: Set[WebSocket] = set()
        # 订阅信息 {websocket: {subscriptions}}
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
        # 客户端信息 {websocket: client_info}
        self.client_info: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = set()
        self.client_info[websocket] = {
            'client_id': client_id or f"client_{len(self.active_connections)}",
            'connected_at': datetime.now(),
            'last_ping': datetime.now()
        }
        
        logger.info(f"WebSocket客户端已连接: {self.client_info[websocket]['client_id']}")
        
        # 发送连接确认消息
        await self.send_personal_message(
            websocket,
            WebSocketMessage(
                type="connection_established",
                data={
                    "client_id": self.client_info[websocket]['client_id'],
                    "server_time": datetime.now().isoformat(),
                    "available_subscriptions": [
                        "system_health",
                        "alerts",
                        "performance",
                        "dashboard"
                    ]
                }
            )
        )
    
    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.active_connections:
            client_id = self.client_info.get(websocket, {}).get('client_id', 'unknown')
            self.active_connections.remove(websocket)
            self.subscriptions.pop(websocket, None)
            self.client_info.pop(websocket, None)
            logger.info(f"WebSocket客户端已断开: {client_id}")
    
    async def send_personal_message(self, websocket: WebSocket, message: WebSocketMessage):
        """发送个人消息"""
        try:
            await websocket.send_text(message.json())
        except Exception as e:
            logger.error(f"发送个人消息失败: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: WebSocketMessage, subscription_type: str = None):
        """广播消息"""
        if not self.active_connections:
            return
        
        # 确定目标连接
        target_connections = set()
        if subscription_type:
            # 只发送给订阅了特定类型的客户端
            for websocket, subscriptions in self.subscriptions.items():
                if subscription_type in subscriptions:
                    target_connections.add(websocket)
        else:
            # 发送给所有连接
            target_connections = self.active_connections.copy()
        
        # 并发发送消息
        if target_connections:
            tasks = []
            for websocket in target_connections:
                tasks.append(self.send_personal_message(websocket, message))
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def subscribe(self, websocket: WebSocket, subscription_type: str):
        """添加订阅"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(subscription_type)
            logger.info(f"客户端 {self.client_info[websocket]['client_id']} 订阅了 {subscription_type}")
    
    def unsubscribe(self, websocket: WebSocket, subscription_type: str):
        """取消订阅"""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(subscription_type)
            logger.info(f"客户端 {self.client_info[websocket]['client_id']} 取消订阅 {subscription_type}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            "total_connections": len(self.active_connections),
            "subscription_stats": {
                subscription_type: sum(
                    1 for subs in self.subscriptions.values() 
                    if subscription_type in subs
                )
                for subscription_type in ["system_health", "alerts", "performance", "dashboard"]
            },
            "clients": [
                {
                    "client_id": info['client_id'],
                    "connected_at": info['connected_at'].isoformat(),
                    "subscriptions": list(self.subscriptions.get(ws, set()))
                }
                for ws, info in self.client_info.items()
            ]
        }

class MonitoringWebSocketHandler:
    """监控WebSocket处理器"""
    
    def __init__(self):
        self.manager = ConnectionManager()
        self.alert_engine: Optional[AlertEngine] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.collector_config: Optional[CollectorConfig] = None
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
    
    def setup(self, alert_engine: AlertEngine, performance_monitor: PerformanceMonitor, collector_config: CollectorConfig):
        """设置依赖组件"""
        self.alert_engine = alert_engine
        self.performance_monitor = performance_monitor
        self.collector_config = collector_config
    
    async def handle_websocket(self, websocket: WebSocket, client_id: str = None):
        """处理WebSocket连接"""
        await self.manager.connect(websocket, client_id)
        
        try:
            while True:
                # 接收客户端消息
                data = await websocket.receive_text()
                await self.handle_client_message(websocket, data)
                
        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket处理错误: {e}")
            self.manager.disconnect(websocket)
    
    async def handle_client_message(self, websocket: WebSocket, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            payload = data.get('data', {})
            
            if message_type == 'subscribe':
                subscription_type = payload.get('subscription_type')
                if subscription_type:
                    self.manager.subscribe(websocket, subscription_type)
                    await self.manager.send_personal_message(
                        websocket,
                        WebSocketMessage(
                            type="subscription_confirmed",
                            data={"subscription_type": subscription_type}
                        )
                    )
            
            elif message_type == 'unsubscribe':
                subscription_type = payload.get('subscription_type')
                if subscription_type:
                    self.manager.unsubscribe(websocket, subscription_type)
                    await self.manager.send_personal_message(
                        websocket,
                        WebSocketMessage(
                            type="unsubscription_confirmed",
                            data={"subscription_type": subscription_type}
                        )
                    )
            
            elif message_type == 'ping':
                # 更新最后ping时间
                if websocket in self.manager.client_info:
                    self.manager.client_info[websocket]['last_ping'] = datetime.now()
                
                await self.manager.send_personal_message(
                    websocket,
                    WebSocketMessage(
                        type="pong",
                        data={"server_time": datetime.now().isoformat()}
                    )
                )
            
            elif message_type == 'get_current_data':
                # 发送当前数据
                data_type = payload.get('data_type')
                await self.send_current_data(websocket, data_type)
            
            else:
                await self.manager.send_personal_message(
                    websocket,
                    WebSocketMessage(
                        type="error",
                        data={"message": f"未知消息类型: {message_type}"}
                    )
                )
        
        except json.JSONDecodeError:
            await self.manager.send_personal_message(
                websocket,
                WebSocketMessage(
                    type="error",
                    data={"message": "无效的JSON格式"}
                )
            )
        except Exception as e:
            logger.error(f"处理客户端消息失败: {e}")
            await self.manager.send_personal_message(
                websocket,
                WebSocketMessage(
                    type="error",
                    data={"message": f"处理消息失败: {str(e)}"}
                )
            )
    
    async def send_current_data(self, websocket: WebSocket, data_type: str):
        """发送当前数据"""
        try:
            if data_type == 'system_health':
                health_data = await collect_system_health(self.collector_config)
                await self.manager.send_personal_message(
                    websocket,
                    WebSocketMessage(
                        type="system_health_update",
                        data=self.format_health_data(health_data)
                    )
                )
            
            elif data_type == 'alerts':
                if self.alert_engine:
                    active_alerts = self.alert_engine.get_active_alerts()
                    alert_stats = self.alert_engine.get_alert_statistics()
                    await self.manager.send_personal_message(
                        websocket,
                        WebSocketMessage(
                            type="alerts_update",
                            data={
                                "active_alerts": [self.format_alert(alert) for alert in active_alerts],
                                "statistics": alert_stats
                            }
                        )
                    )
            
            elif data_type == 'performance':
                if self.performance_monitor:
                    # 获取关键性能指标
                    key_metrics = ['cpu_usage', 'memory_usage', 'disk_usage', 'response_time']
                    performance_data = {}
                    
                    for metric in key_metrics:
                        stats = self.performance_monitor.metric_collector.get_metric_stats(metric, 300)  # 最近5分钟
                        if stats:
                            performance_data[metric] = stats
                    
                    await self.manager.send_personal_message(
                        websocket,
                        WebSocketMessage(
                            type="performance_update",
                            data=performance_data
                        )
                    )
            
            elif data_type == 'dashboard':
                # 发送仪表板数据
                dashboard_data = await self.get_dashboard_data()
                await self.manager.send_personal_message(
                    websocket,
                    WebSocketMessage(
                        type="dashboard_update",
                        data=dashboard_data
                    )
                )
        
        except Exception as e:
            logger.error(f"发送当前数据失败: {e}")
            await self.manager.send_personal_message(
                websocket,
                WebSocketMessage(
                    type="error",
                    data={"message": f"获取数据失败: {str(e)}"}
                )
            )
    
    def format_health_data(self, health_data: Dict) -> Dict:
        """格式化健康数据"""
        formatted_data = {}
        for category, metrics in health_data.items():
            formatted_metrics = []
            for metric in metrics:
                formatted_metrics.append({
                    "component": metric.component,
                    "status": metric.status,
                    "value": metric.value,
                    "unit": metric.unit,
                    "message": metric.message,
                    "timestamp": metric.timestamp.isoformat(),
                    "metadata": metric.metadata
                })
            formatted_data[category] = formatted_metrics
        return formatted_data
    
    def format_alert(self, alert: Alert) -> Dict:
        """格式化告警数据"""
        return {
            "id": alert.id,
            "rule_id": alert.rule_id,
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.value,
            "alert_type": alert.alert_type.value,
            "status": alert.status.value,
            "created_at": alert.created_at.isoformat(),
            "updated_at": alert.updated_at.isoformat(),
            "source": alert.source,
            "tags": alert.tags,
            "metadata": alert.metadata
        }
    
    async def get_dashboard_data(self) -> Dict:
        """获取仪表板数据"""
        try:
            # 获取系统健康状态
            health_data = await collect_system_health(self.collector_config)
            
            # 获取告警统计
            alert_stats = {}
            if self.alert_engine:
                alert_stats = self.alert_engine.get_alert_statistics()
            
            # 获取性能指标摘要
            performance_summary = {}
            if self.performance_monitor:
                key_metrics = ['cpu_usage', 'memory_usage', 'disk_usage', 'response_time']
                for metric in key_metrics:
                    stats = self.performance_monitor.metric_collector.get_metric_stats(metric, 3600)  # 最近1小时
                    if stats:
                        performance_summary[metric] = stats
            
            # 计算整体健康分数
            all_metrics = []
            for metrics in health_data.values():
                all_metrics.extend(metrics)
            
            total_components = len(all_metrics)
            healthy_components = sum(1 for m in all_metrics if m.status == "healthy")
            health_score = (healthy_components / total_components * 100) if total_components > 0 else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "health_score": round(health_score, 1),
                "system_health": self.format_health_data(health_data),
                "alert_statistics": alert_stats,
                "performance_summary": performance_summary,
                "summary": {
                    "total_components": total_components,
                    "healthy_components": healthy_components,
                    "active_alerts": alert_stats.get('active_alerts', 0),
                    "critical_alerts": alert_stats.get('severity_distribution', {}).get('critical', 0)
                }
            }
        
        except Exception as e:
            logger.error(f"获取仪表板数据失败: {e}")
            return {"error": f"获取数据失败: {str(e)}"}
    
    async def start_background_tasks(self):
        """启动后台任务"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动定期数据推送任务
        tasks = [
            asyncio.create_task(self.health_monitor_task()),
            asyncio.create_task(self.alert_monitor_task()),
            asyncio.create_task(self.performance_monitor_task()),
            asyncio.create_task(self.connection_cleanup_task())
        ]
        
        self.background_tasks.update(tasks)
        logger.info("WebSocket后台任务已启动")
    
    async def stop_background_tasks(self):
        """停止后台任务"""
        self.is_running = False
        
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        logger.info("WebSocket后台任务已停止")
    
    async def health_monitor_task(self):
        """系统健康监控任务"""
        while self.is_running:
            try:
                if self.collector_config:
                    health_data = await collect_system_health(self.collector_config)
                    await self.manager.broadcast(
                        WebSocketMessage(
                            type="system_health_update",
                            data=self.format_health_data(health_data)
                        ),
                        subscription_type="system_health"
                    )
                
                await asyncio.sleep(30)  # 每30秒推送一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"系统健康监控任务错误: {e}")
                await asyncio.sleep(30)
    
    async def alert_monitor_task(self):
        """告警监控任务"""
        last_alert_count = 0
        
        while self.is_running:
            try:
                if self.alert_engine:
                    active_alerts = self.alert_engine.get_active_alerts()
                    current_alert_count = len(active_alerts)
                    
                    # 如果告警数量发生变化，推送更新
                    if current_alert_count != last_alert_count:
                        alert_stats = self.alert_engine.get_alert_statistics()
                        await self.manager.broadcast(
                            WebSocketMessage(
                                type="alerts_update",
                                data={
                                    "active_alerts": [self.format_alert(alert) for alert in active_alerts],
                                    "statistics": alert_stats
                                }
                            ),
                            subscription_type="alerts"
                        )
                        last_alert_count = current_alert_count
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"告警监控任务错误: {e}")
                await asyncio.sleep(10)
    
    async def performance_monitor_task(self):
        """性能监控任务"""
        while self.is_running:
            try:
                if self.performance_monitor:
                    # 获取关键性能指标
                    key_metrics = ['cpu_usage', 'memory_usage', 'disk_usage', 'response_time']
                    performance_data = {}
                    
                    for metric in key_metrics:
                        stats = self.performance_monitor.metric_collector.get_metric_stats(metric, 300)  # 最近5分钟
                        if stats:
                            performance_data[metric] = stats
                    
                    if performance_data:
                        await self.manager.broadcast(
                            WebSocketMessage(
                                type="performance_update",
                                data=performance_data
                            ),
                            subscription_type="performance"
                        )
                
                await asyncio.sleep(60)  # 每分钟推送一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"性能监控任务错误: {e}")
                await asyncio.sleep(60)
    
    async def connection_cleanup_task(self):
        """连接清理任务"""
        while self.is_running:
            try:
                # 清理超时连接
                current_time = datetime.now()
                timeout_connections = []
                
                for websocket, client_info in self.manager.client_info.items():
                    last_ping = client_info.get('last_ping', client_info['connected_at'])
                    if (current_time - last_ping).total_seconds() > 300:  # 5分钟超时
                        timeout_connections.append(websocket)
                
                for websocket in timeout_connections:
                    logger.info(f"清理超时连接: {self.manager.client_info[websocket]['client_id']}")
                    self.manager.disconnect(websocket)
                
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"连接清理任务错误: {e}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取WebSocket统计信息"""
        return {
            "connection_stats": self.manager.get_connection_stats(),
            "background_tasks": {
                "total_tasks": len(self.background_tasks),
                "running_tasks": sum(1 for task in self.background_tasks if not task.done()),
                "is_running": self.is_running
            }
        }

# 全局WebSocket处理器实例
websocket_handler = MonitoringWebSocketHandler()

# WebSocket端点函数
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """WebSocket端点"""
    await websocket_handler.handle_websocket(websocket, client_id)

# 设置函数
def setup_websocket_handler(alert_engine, performance_monitor, collector_config):
    """设置WebSocket处理器"""
    websocket_handler.setup(alert_engine, performance_monitor, collector_config)
    return websocket_handler