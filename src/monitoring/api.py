#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控API模块

提供监控数据的REST API接口
包括系统健康状态、告警管理、性能指标等

作者: StockSchool Team
创建时间: 2025-01-02
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.monitoring.collectors import collect_system_health, CollectorConfig, HealthMetric
from src.monitoring.alerts import AlertEngine, AlertRule, Alert, AlertSeverity, AlertType, AlertStatus
from src.monitoring.performance import PerformanceMonitor
from src.config.unified_config import config

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# 全局实例
alert_engine = None
performance_monitor = None
collector_config = None

def get_alert_engine() -> AlertEngine:
    """获取告警引擎实例"""
    global alert_engine
    if alert_engine is None:
        alert_engine = AlertEngine()
        alert_engine.start()
    return alert_engine

def get_performance_monitor() -> PerformanceMonitor:
    """获取性能监控实例"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
    return performance_monitor

def get_collector_config() -> CollectorConfig:
    """获取收集器配置"""
    global collector_config
    if collector_config is None:
        collector_config = CollectorConfig()
    return collector_config

# Pydantic模型
class HealthMetricResponse(BaseModel):
    """健康指标响应模型"""
    component: str
    status: str
    value: Optional[float] = None
    unit: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class SystemHealthResponse(BaseModel):
    """系统健康状态响应模型"""
    timestamp: datetime
    overall_status: str
    categories: Dict[str, List[HealthMetricResponse]]
    summary: Dict[str, Any]

class AlertResponse(BaseModel):
    """告警响应模型"""
    id: str
    rule_id: str
    title: str
    message: str
    severity: str
    alert_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    source: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class AlertRuleRequest(BaseModel):
    """告警规则请求模型"""
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    alert_type: str = Field(..., description="告警类型")
    severity: str = Field(..., description="严重程度")
    condition: str = Field(..., description="条件表达式")
    threshold: float = Field(..., description="阈值")
    comparison: str = Field(..., description="比较操作符")
    evaluation_window: int = Field(60, description="评估窗口(秒)")
    trigger_count: int = Field(1, description="触发次数")
    enabled: bool = Field(True, description="是否启用")
    tags: List[str] = Field([], description="标签")
    metadata: Dict[str, Any] = Field({}, description="元数据")

class AlertAcknowledgeRequest(BaseModel):
    """告警确认请求模型"""
    acknowledged_by: str = Field(..., description="确认人")

class PerformanceMetricResponse(BaseModel):
    """性能指标响应模型"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}

# API端点
@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """获取系统健康状态"""
    try:
        config = get_collector_config()
        health_data = await collect_system_health(config)
        
        # 转换为响应格式
        categories = {}
        all_metrics = []
        
        for category, metrics in health_data.items():
            category_metrics = []
            for metric in metrics:
                metric_response = HealthMetricResponse(
                    component=metric.component,
                    status=metric.status,
                    value=metric.value,
                    unit=metric.unit,
                    message=metric.message,
                    timestamp=metric.timestamp,
                    metadata=metric.metadata
                )
                category_metrics.append(metric_response)
                all_metrics.append(metric)
            categories[category] = category_metrics
        
        # 计算整体状态
        overall_status = "healthy"
        critical_count = sum(1 for m in all_metrics if m.status == "critical")
        warning_count = sum(1 for m in all_metrics if m.status == "warning")
        
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        
        # 生成摘要
        summary = {
            "total_components": len(all_metrics),
            "healthy_count": sum(1 for m in all_metrics if m.status == "healthy"),
            "warning_count": warning_count,
            "critical_count": critical_count,
            "unknown_count": sum(1 for m in all_metrics if m.status == "unknown")
        }
        
        return SystemHealthResponse(
            timestamp=datetime.now(),
            overall_status=overall_status,
            categories=categories,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"获取系统健康状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统健康状态失败: {str(e)}")

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """获取告警列表"""
    try:
        engine = get_alert_engine()
        
        # 根据状态过滤
        alert_status = None
        if status:
            try:
                alert_status = AlertStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的告警状态: {status}")
        
        alerts = engine.storage.load_alerts(status=alert_status, limit=limit)
        
        # 根据严重程度过滤
        if severity:
            try:
                severity_enum = AlertSeverity(severity)
                alerts = [a for a in alerts if a.severity == severity_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的严重程度: {severity}")
        
        # 转换为响应格式
        alert_responses = []
        for alert in alerts:
            alert_response = AlertResponse(
                id=alert.id,
                rule_id=alert.rule_id,
                title=alert.title,
                message=alert.message,
                severity=alert.severity.value,
                alert_type=alert.alert_type.value,
                status=alert.status.value,
                created_at=alert.created_at,
                updated_at=alert.updated_at,
                resolved_at=alert.resolved_at,
                acknowledged_at=alert.acknowledged_at,
                acknowledged_by=alert.acknowledged_by,
                source=alert.source,
                tags=alert.tags,
                metadata=alert.metadata
            )
            alert_responses.append(alert_response)
        
        return alert_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取告警列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取告警列表失败: {str(e)}")

@router.get("/alerts/statistics")
async def get_alert_statistics():
    """获取告警统计信息"""
    try:
        engine = get_alert_engine()
        stats = engine.get_alert_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"获取告警统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取告警统计失败: {str(e)}")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    request: AlertAcknowledgeRequest
):
    """确认告警"""
    try:
        engine = get_alert_engine()
        engine.acknowledge_alert(alert_id, request.acknowledged_by)
        return {"message": "告警已确认", "alert_id": alert_id}
        
    except Exception as e:
        logger.error(f"确认告警失败: {e}")
        raise HTTPException(status_code=500, detail=f"确认告警失败: {str(e)}")

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """解决告警"""
    try:
        engine = get_alert_engine()
        engine.resolve_alert(alert_id)
        return {"message": "告警已解决", "alert_id": alert_id}
        
    except Exception as e:
        logger.error(f"解决告警失败: {e}")
        raise HTTPException(status_code=500, detail=f"解决告警失败: {str(e)}")

@router.get("/rules")
async def get_alert_rules():
    """获取告警规则列表"""
    try:
        engine = get_alert_engine()
        rules = engine.storage.load_rules()
        
        rule_responses = []
        for rule in rules:
            rule_data = {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "alert_type": rule.alert_type.value,
                "severity": rule.severity.value,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "comparison": rule.comparison,
                "evaluation_window": rule.evaluation_window,
                "trigger_count": rule.trigger_count,
                "enabled": rule.enabled,
                "tags": rule.tags,
                "metadata": rule.metadata
            }
            rule_responses.append(rule_data)
        
        return rule_responses
        
    except Exception as e:
        logger.error(f"获取告警规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取告警规则失败: {str(e)}")

@router.post("/rules")
async def create_alert_rule(request: AlertRuleRequest):
    """创建告警规则"""
    try:
        # 验证枚举值
        try:
            alert_type = AlertType(request.alert_type)
            severity = AlertSeverity(request.severity)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"无效的枚举值: {str(e)}")
        
        # 创建规则
        rule = AlertRule(
            id=f"rule_{int(datetime.now().timestamp())}",
            name=request.name,
            description=request.description,
            alert_type=alert_type,
            severity=severity,
            condition=request.condition,
            threshold=request.threshold,
            comparison=request.comparison,
            evaluation_window=request.evaluation_window,
            trigger_count=request.trigger_count,
            enabled=request.enabled,
            tags=request.tags,
            metadata=request.metadata
        )
        
        # 保存规则
        engine = get_alert_engine()
        engine.storage.save_rule(rule)
        engine.add_rule(rule)
        
        return {"message": "告警规则已创建", "rule_id": rule.id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建告警规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建告警规则失败: {str(e)}")

@router.get("/performance/metrics")
async def get_performance_metrics(
    metric_name: Optional[str] = None,
    window_seconds: Optional[int] = None,
    limit: int = 100
):
    """获取性能指标"""
    try:
        monitor = get_performance_monitor()
        
        if metric_name:
            # 获取特定指标
            values = monitor.metric_collector.get_metric_values(metric_name, window_seconds)
            stats = monitor.metric_collector.get_metric_stats(metric_name, window_seconds)
            
            return {
                "metric_name": metric_name,
                "values": [(ts.isoformat(), val) for ts, val in values[-limit:]],
                "statistics": stats
            }
        else:
            # 获取所有指标的统计信息
            all_metrics = {}
            for name in monitor.metric_collector.metrics.keys():
                stats = monitor.metric_collector.get_metric_stats(name, window_seconds)
                if stats:
                    all_metrics[name] = stats
            
            return {"metrics": all_metrics}
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")

@router.get("/dashboard")
async def get_monitoring_dashboard():
    """获取监控仪表板数据"""
    try:
        # 获取系统健康状态
        config = get_collector_config()
        health_data = await collect_system_health(config)
        
        # 获取告警统计
        engine = get_alert_engine()
        alert_stats = engine.get_alert_statistics()
        
        # 获取性能指标摘要
        monitor = get_performance_monitor()
        performance_summary = {}
        
        # 获取关键性能指标
        key_metrics = ['cpu_usage', 'memory_usage', 'disk_usage', 'response_time']
        for metric in key_metrics:
            stats = monitor.metric_collector.get_metric_stats(metric, 3600)  # 最近1小时
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
            "system_health": health_data,
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
        logger.error(f"获取监控仪表板失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控仪表板失败: {str(e)}")

# 后台任务
async def start_monitoring_background_tasks():
    """启动监控后台任务"""
    try:
        # 启动告警引擎
        engine = get_alert_engine()
        
        # 启动性能监控
        monitor = get_performance_monitor()
        
        # 添加默认告警规则
        await add_default_alert_rules(engine)
        
        logger.info("监控后台任务已启动")
        
    except Exception as e:
        logger.error(f"启动监控后台任务失败: {e}")

async def add_default_alert_rules(engine: AlertEngine):
    """添加默认告警规则"""
    try:
        # 获取配置中的阈值
        monitoring_config = config.get('monitoring_params', {})
        alert_config = monitoring_config.get('alerts', {})
        
        default_rules = [
            {
                "id": "cpu_high",
                "name": "CPU使用率过高",
                "description": "CPU使用率超过阈值",
                "alert_type": AlertType.SYSTEM,
                "severity": AlertSeverity.WARNING,
                "condition": "system.cpu",
                "threshold": alert_config.get('cpu_threshold', 80.0),
                "comparison": ">",
                "evaluation_window": 60,
                "trigger_count": 3
            },
            {
                "id": "memory_high",
                "name": "内存使用率过高",
                "description": "内存使用率超过阈值",
                "alert_type": AlertType.SYSTEM,
                "severity": AlertSeverity.WARNING,
                "condition": "system.memory",
                "threshold": alert_config.get('memory_threshold', 85.0),
                "comparison": ">",
                "evaluation_window": 60,
                "trigger_count": 3
            },
            {
                "id": "database_down",
                "name": "数据库连接失败",
                "description": "数据库连接不可用",
                "alert_type": AlertType.DATABASE,
                "severity": AlertSeverity.CRITICAL,
                "condition": "database.connection",
                "threshold": 0,
                "comparison": "==",
                "evaluation_window": 30,
                "trigger_count": 1
            }
        ]
        
        for rule_data in default_rules:
            rule = AlertRule(
                id=rule_data["id"],
                name=rule_data["name"],
                description=rule_data["description"],
                alert_type=rule_data["alert_type"],
                severity=rule_data["severity"],
                condition=rule_data["condition"],
                threshold=rule_data["threshold"],
                comparison=rule_data["comparison"],
                evaluation_window=rule_data["evaluation_window"],
                trigger_count=rule_data["trigger_count"],
                enabled=True,
                tags=[],
                metadata={}
            )
            
            # 检查规则是否已存在
            existing_rules = engine.storage.load_rules()
            if not any(r.id == rule.id for r in existing_rules):
                engine.storage.save_rule(rule)
                engine.add_rule(rule)
                logger.info(f"添加默认告警规则: {rule.name}")
        
    except Exception as e:
        logger.error(f"添加默认告警规则失败: {e}")

# 启动时执行
def setup_monitoring():
    """设置监控模块"""
    asyncio.create_task(start_monitoring_background_tasks())