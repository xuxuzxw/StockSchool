#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统模块

提供完整的系统监控功能，包括：
- 系统健康状态监控
- 性能指标收集
- 告警管理
- 实时数据推送
- 通知管理
"""

import asyncio
from typing import Optional
from loguru import logger

from .config import get_monitoring_config
from .collectors import collect_system_health, CollectorConfig
from .performance import PerformanceMonitor
from .alerts import AlertEngine
from .notifications import NotificationManager
from .websocket import MonitoringWebSocketHandler


class MonitoringSystem:
    """监控系统主类"""
    
    def __init__(self):
        self.config = get_monitoring_config()
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.alert_engine: Optional[AlertEngine] = None
        self.notification_manager: Optional[NotificationManager] = None
        self.websocket_handler: Optional[MonitoringWebSocketHandler] = None
        self.collector_config: Optional[CollectorConfig] = None
        self._running = False
        self._tasks = []
    
    async def initialize(self):
        """初始化监控系统"""
        try:
            logger.info("正在初始化监控系统...")
            
            # 初始化收集器配置
            self.collector_config = CollectorConfig()
            
            # 初始化性能监控器
            self.performance_monitor = PerformanceMonitor()
            
            # 初始化通知管理器
            self.notification_manager = NotificationManager()
            
            # 初始化告警引擎
            self.alert_engine = AlertEngine(
                notification_manager=self.notification_manager
            )
            
            # 初始化WebSocket处理器
            self.websocket_handler = MonitoringWebSocketHandler()
            
            # 添加默认告警规则
            await self._add_default_alert_rules()
            
            logger.info("监控系统初始化完成")
            
        except Exception as e:
            logger.error(f"监控系统初始化失败: {e}")
            raise
    
    async def start(self):
        """启动监控系统"""
        if self._running:
            logger.warning("监控系统已在运行中")
            return
        
        try:
            logger.info("正在启动监控系统...")
            
            # 启动性能监控器
            if self.performance_monitor:
                self.performance_monitor.start()
            
            # 启动告警引擎
            if self.alert_engine:
                self.alert_engine.start()
            
            # 启动后台任务
            self._running = True
            self._tasks = [
                asyncio.create_task(self._health_monitoring_task()),
                asyncio.create_task(self._cleanup_task())
            ]
            
            logger.info("监控系统启动完成")
            
        except Exception as e:
            logger.error(f"监控系统启动失败: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止监控系统"""
        if not self._running:
            return
        
        try:
            logger.info("正在停止监控系统...")
            
            self._running = False
            
            # 取消后台任务
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            # 等待任务完成
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # 停止告警引擎
            if self.alert_engine:
                self.alert_engine.stop()
            
            # 停止性能监控器
            if self.performance_monitor:
                self.performance_monitor.stop()
            
            # 关闭WebSocket连接
            if self.websocket_handler:
                await self.websocket_handler.disconnect_all()
            
            logger.info("监控系统已停止")
            
        except Exception as e:
            logger.error(f"停止监控系统时出错: {e}")
    
    async def get_system_health(self):
        """获取系统健康状态"""
        if not self.collector_config:
            raise RuntimeError("监控系统未初始化")
        
        return await collect_system_health(self.collector_config)
    
    async def get_performance_metrics(self, hours: int = 1):
        """获取性能指标"""
        if not self.performance_monitor:
            raise RuntimeError("性能监控器未初始化")
        
        return self.performance_monitor.get_metrics_summary(hours=hours)
    
    async def get_alerts(self, status: Optional[str] = None, limit: int = 100):
        """获取告警列表"""
        if not self.alert_engine:
            raise RuntimeError("告警引擎未初始化")
        
        return self.alert_engine.get_alerts(status=status, limit=limit)
    
    async def get_alert_stats(self):
        """获取告警统计"""
        if not self.alert_engine:
            raise RuntimeError("告警引擎未初始化")
        
        return self.alert_engine.get_alert_stats()
    
    async def acknowledge_alert(self, alert_id: str, user: str, comment: str = ""):
        """确认告警"""
        if not self.alert_engine:
            raise RuntimeError("告警引擎未初始化")
        
        return self.alert_engine.acknowledge_alert(alert_id, user, comment)
    
    async def resolve_alert(self, alert_id: str, user: str, comment: str = ""):
        """解决告警"""
        if not self.alert_engine:
            raise RuntimeError("告警引擎未初始化")
        
        return self.alert_engine.resolve_alert(alert_id, user, comment)
    
    async def _health_monitoring_task(self):
        """健康监控后台任务"""
        while self._running:
            try:
                # 收集系统健康状态
                health_data = await self.get_system_health()
                
                # 检查健康状态并触发告警
                await self._check_health_alerts(health_data)
                
                # 通过WebSocket推送数据
                if self.websocket_handler:
                    await self.websocket_handler.broadcast_health_update(health_data)
                
                # 等待下次检查
                await asyncio.sleep(self.config.system_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康监控任务出错: {e}")
                await asyncio.sleep(10)  # 出错时等待10秒再重试
    
    async def _cleanup_task(self):
        """清理任务"""
        while self._running:
            try:
                # 清理过期的性能数据
                if self.performance_monitor:
                    self.performance_monitor.cleanup_old_data(
                        days=self.config.performance_retention_days
                    )
                
                # 清理过期的告警数据
                if self.alert_engine:
                    # 这里可以添加告警数据清理逻辑
                    pass
                
                # 每小时执行一次清理
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理任务出错: {e}")
                await asyncio.sleep(600)  # 出错时等待10分钟再重试
    
    async def _check_health_alerts(self, health_data):
        """检查健康状态并触发告警"""
        if not self.alert_engine:
            return
        
        try:
            for component, metrics in health_data.items():
                if isinstance(metrics, dict) and 'status' in metrics:
                    if metrics['status'] != 'healthy':
                        # 触发告警
                        await self.alert_engine.trigger_alert(
                            rule_name=f"{component}_unhealthy",
                            metric_name=f"{component}.status",
                            current_value=metrics['status'],
                            threshold="healthy",
                            context={
                                'component': component,
                                'metrics': metrics,
                                'timestamp': metrics.get('timestamp')
                            }
                        )
        except Exception as e:
            logger.error(f"检查健康状态告警时出错: {e}")
    
    async def _add_default_alert_rules(self):
        """添加默认告警规则"""
        if not self.alert_engine:
            return
        
        try:
            # 系统资源告警规则
            default_rules = [
                {
                    'name': 'high_cpu_usage',
                    'metric': 'system.cpu_usage',
                    'condition': 'greater_than',
                    'threshold': self.config.thresholds['cpu_usage'],
                    'severity': 'warning',
                    'description': 'CPU使用率过高'
                },
                {
                    'name': 'high_memory_usage',
                    'metric': 'system.memory_usage',
                    'condition': 'greater_than',
                    'threshold': self.config.thresholds['memory_usage'],
                    'severity': 'warning',
                    'description': '内存使用率过高'
                },
                {
                    'name': 'high_disk_usage',
                    'metric': 'system.disk_usage',
                    'condition': 'greater_than',
                    'threshold': self.config.thresholds['disk_usage'],
                    'severity': 'critical',
                    'description': '磁盘使用率过高'
                },
                {
                    'name': 'database_unhealthy',
                    'metric': 'database.status',
                    'condition': 'not_equals',
                    'threshold': 'healthy',
                    'severity': 'critical',
                    'description': '数据库连接异常'
                },
                {
                    'name': 'redis_unhealthy',
                    'metric': 'redis.status',
                    'condition': 'not_equals',
                    'threshold': 'healthy',
                    'severity': 'warning',
                    'description': 'Redis连接异常'
                }
            ]
            
            for rule_config in default_rules:
                await self.alert_engine.add_rule(**rule_config)
            
            logger.info(f"已添加 {len(default_rules)} 个默认告警规则")
            
        except Exception as e:
            logger.error(f"添加默认告警规则失败: {e}")


# 全局监控系统实例
_monitoring_system: Optional[MonitoringSystem] = None


async def get_monitoring_system() -> MonitoringSystem:
    """获取监控系统实例"""
    global _monitoring_system
    
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
        await _monitoring_system.initialize()
    
    return _monitoring_system


async def start_monitoring_system():
    """启动监控系统"""
    system = await get_monitoring_system()
    await system.start()
    return system


async def stop_monitoring_system():
    """停止监控系统"""
    global _monitoring_system
    
    if _monitoring_system:
        await _monitoring_system.stop()
        _monitoring_system = None


# 导出主要类和函数
__all__ = [
    'MonitoringSystem',
    'get_monitoring_system',
    'start_monitoring_system',
    'stop_monitoring_system',
    'get_monitoring_config'
]