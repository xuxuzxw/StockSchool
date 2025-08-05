import json
import smtplib
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算进度监控器
实现计算进度的实时跟踪、性能统计和异常告警
"""


from .factor_models import CalculationStatus, FactorType
from .task_scheduler import TaskExecution, TaskStatus


class AlertLevel(Enum):
    """告警级别枚举"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型枚举"""

    TASK_FAILED = "task_failed"
    RESOURCE_HIGH = "resource_high"
    PERFORMANCE_DEGRADED = "performance_degraded"
    SYSTEM_ERROR = "system_error"
    DATA_QUALITY = "data_quality"


@dataclass
class Alert:
    """告警信息"""

    alert_id: str
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None
    resolved: bool = False
    resolved_time: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_tasks: int
    completed_tasks_1h: int
    failed_tasks_1h: int
    avg_task_duration: float
    throughput_per_hour: float


class FactorCalculationMonitor:
    """因子计算监控器"""

    def __init__(self, engine):
        """初始化计算监控器"""
        self.engine = engine

        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None

        # 性能指标历史
        self.performance_history = deque(maxlen=1440)  # 保留24小时数据（每分钟一个点）

        # 告警管理
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_handlers = []  # 告警处理器列表
        self.alert_rules = {}  # 告警规则

        # 计算统计
        self.calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "total_execution_time": timedelta(0),
            "last_update_time": datetime.now(),
        }

        # 线程锁
        self.stats_lock = threading.Lock()
        self.alerts_lock = threading.Lock()

        # 初始化告警规则
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        # CPU使用率过高
        self.alert_rules["high_cpu"] = {
            "condition": lambda metrics: metrics.cpu_percent > 90,
            "alert_type": AlertType.RESOURCE_HIGH,
            "level": AlertLevel.WARNING,
            "title": "CPU使用率过高",
            "cooldown": 300,  # 5分钟冷却期
        }

        # 内存使用率过高
        self.alert_rules["high_memory"] = {
            "condition": lambda metrics: metrics.memory_percent > 85,
            "alert_type": AlertType.RESOURCE_HIGH,
            "level": AlertLevel.WARNING,
            "title": "内存使用率过高",
            "cooldown": 300,
        }

        # 任务失败率过高
        self.alert_rules["high_failure_rate"] = {
            "condition": lambda metrics: (
                metrics.failed_tasks_1h > 0
                and metrics.failed_tasks_1h / (metrics.completed_tasks_1h + metrics.failed_tasks_1h) > 0.2
            ),
            "alert_type": AlertType.PERFORMANCE_DEGRADED,
            "level": AlertLevel.ERROR,
            "title": "任务失败率过高",
            "cooldown": 600,  # 10分钟冷却期
        }

        # 吞吐量下降
        self.alert_rules["low_throughput"] = {
            "condition": lambda metrics: metrics.throughput_per_hour < 10,
            "alert_type": AlertType.PERFORMANCE_DEGRADED,
            "level": AlertLevel.WARNING,
            "title": "计算吞吐量下降",
            "cooldown": 900,  # 15分钟冷却期
        }

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)

    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_active:
            logger.warning("监控已在运行")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("计算监控器已启动")

    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring_active:
            logger.warning("监控未在运行")
            return

        self.monitoring_active = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("计算监控器已停止")

    def _monitoring_loop(self):
        """监控主循环"""
        logger.info("开始监控循环")

        while self.monitoring_active:
            try:
                # 收集性能指标
                metrics = self._collect_performance_metrics()

                # 保存指标历史
                self.performance_history.append(metrics)

                # 检查告警规则
                self._check_alert_rules(metrics)

                # 更新计算统计
                self._update_calculation_stats()

                # 休眠60秒
                time.sleep(60)

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(30)

        logger.info("监控循环结束")

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        import psutil

        # 系统资源指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage("/").percent

        # 任务执行指标
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(hours=1)

        try:
            # 查询最近1小时的任务执行情况
            query = text(
                """
                SELECT
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as completed_tasks,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_tasks,
                    AVG(duration_seconds) as avg_duration
                FROM task_execution_log
                WHERE start_time >= :one_hour_ago
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(query, {"one_hour_ago": one_hour_ago})
                row = result.fetchone()

                if row:
                    completed_tasks_1h = int(row.completed_tasks or 0)
                    failed_tasks_1h = int(row.failed_tasks or 0)
                    avg_duration = float(row.avg_duration or 0)
                else:
                    completed_tasks_1h = 0
                    failed_tasks_1h = 0
                    avg_duration = 0

            # 查询当前活跃任务数
            active_query = text(
                """
                SELECT COUNT(*) as active_count
                FROM task_execution_log
                WHERE status = 'running'
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(active_query)
                row = result.fetchone()
                active_tasks = int(row.active_count or 0) if row else 0

        except Exception as e:
            logger.error(f"查询任务执行指标失败: {e}")
            completed_tasks_1h = 0
            failed_tasks_1h = 0
            avg_duration = 0
            active_tasks = 0

        # 计算吞吐量
        throughput_per_hour = completed_tasks_1h

        return PerformanceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            active_tasks=active_tasks,
            completed_tasks_1h=completed_tasks_1h,
            failed_tasks_1h=failed_tasks_1h,
            avg_task_duration=avg_duration,
            throughput_per_hour=throughput_per_hour,
        )

    def _check_alert_rules(self, metrics: PerformanceMetrics):
        """检查告警规则"""
        current_time = datetime.now()

        for rule_name, rule_config in self.alert_rules.items():
            try:
                # 检查条件
                if rule_config["condition"](metrics):
                    # 检查冷却期
                    last_alert_time = getattr(self, f"_last_alert_{rule_name}", None)
                    if last_alert_time and (current_time - last_alert_time).total_seconds() < rule_config["cooldown"]:
                        continue

                    # 创建告警
                    alert = Alert(
                        alert_id=f"{rule_name}_{int(current_time.timestamp())}",
                        alert_type=rule_config["alert_type"],
                        level=rule_config["level"],
                        title=rule_config["title"],
                        message=self._generate_alert_message(rule_name, metrics),
                        timestamp=current_time,
                        source="calculation_monitor",
                        metadata={"rule_name": rule_name, "metrics": asdict(metrics)},
                    )

                    # 触发告警
                    self._trigger_alert(alert)

                    # 记录最后告警时间
                    setattr(self, f"_last_alert_{rule_name}", current_time)

            except Exception as e:
                logger.error(f"检查告警规则 {rule_name} 失败: {e}")

    def _generate_alert_message(self, rule_name: str, metrics: PerformanceMetrics) -> str:
        """生成告警消息"""
        if rule_name == "high_cpu":
            return f"CPU使用率达到 {metrics.cpu_percent:.1f}%，超过阈值90%"
        elif rule_name == "high_memory":
            return f"内存使用率达到 {metrics.memory_percent:.1f}%，超过阈值85%"
        elif rule_name == "high_failure_rate":
            total_tasks = metrics.completed_tasks_1h + metrics.failed_tasks_1h
            failure_rate = metrics.failed_tasks_1h / total_tasks if total_tasks > 0 else 0
            return f"最近1小时任务失败率达到 {failure_rate:.1%}，失败任务数: {metrics.failed_tasks_1h}"
        elif rule_name == "low_throughput":
            return f"计算吞吐量降至 {metrics.throughput_per_hour:.1f} 任务/小时，低于预期"
        else:
            return f"触发告警规则: {rule_name}"

    def _trigger_alert(self, alert: Alert):
        """触发告警"""
        with self.alerts_lock:
            # 保存告警
            self.active_alerts[alert.alert_id] = alert

            # 记录到数据库
            self._save_alert_to_db(alert)

            # 调用告警处理器
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"告警处理器执行失败: {e}")

            logger.warning(f"触发告警: {alert.title} - {alert.message}")

    def _save_alert_to_db(self, alert: Alert):
        """保存告警到数据库"""
        try:
            alert_data = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "source": alert.source,
                "metadata": json.dumps(alert.metadata) if alert.metadata else None,
                "resolved": alert.resolved,
                "resolved_time": alert.resolved_time,
            }

            df = pd.DataFrame([alert_data])
            df.to_sql("calculation_alerts", self.engine, if_exists="append", index=False)

        except Exception as e:
            logger.error(f"保存告警到数据库失败: {e}")

    def _update_calculation_stats(self):
        """更新计算统计"""
        try:
            with self.stats_lock:
                # 查询总体统计
                query = text(
                    """
                    SELECT
                        COUNT(*) as total_calculations,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_calculations,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_calculations,
                        SUM(duration_seconds) as total_duration_seconds
                    FROM task_execution_log
                """
                )

                with self.engine.connect() as conn:
                    result = conn.execute(query)
                    row = result.fetchone()

                    if row:
                        self.calculation_stats.update(
                            {
                                "total_calculations": int(row.total_calculations or 0),
                                "successful_calculations": int(row.successful_calculations or 0),
                                "failed_calculations": int(row.failed_calculations or 0),
                                "total_execution_time": timedelta(seconds=float(row.total_duration_seconds or 0)),
                                "last_update_time": datetime.now(),
                            }
                        )

        except Exception as e:
            logger.error(f"更新计算统计失败: {e}")

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        if self.performance_history:
            return self.performance_history[-1]
        return None

    def get_metrics_history(self, hours: int = 24) -> List[PerformanceMetrics]:
        """获取指标历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [metrics for metrics in self.performance_history if metrics.timestamp >= cutoff_time]

    def get_calculation_progress(self, task_type: str = None) -> Dict[str, Any]:
        """获取计算进度"""
        try:
            # 构建查询条件
            where_clause = ""
            params = {}

            if task_type:
                where_clause = "WHERE task_id LIKE :task_type"
                params["task_type"] = f"%{task_type}%"

            # 查询进度信息
            query = text(
                f"""
                SELECT
                    status,
                    COUNT(*) as count,
                    AVG(duration_seconds) as avg_duration
                FROM task_execution_log
                {where_clause}
                GROUP BY status
            """
            )

            with self.engine.connect() as conn:
                result = conn.execute(query, params)

                progress_data = {}
                total_tasks = 0

                for row in result.fetchall():
                    status = row.status
                    count = int(row.count)
                    avg_duration = float(row.avg_duration or 0)

                    progress_data[status] = {"count": count, "avg_duration": avg_duration}
                    total_tasks += count

                # 计算进度百分比
                if total_tasks > 0:
                    for status_data in progress_data.values():
                        status_data["percentage"] = (status_data["count"] / total_tasks) * 100

                return {
                    "total_tasks": total_tasks,
                    "status_breakdown": progress_data,
                    "last_updated": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"获取计算进度失败: {e}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        current_metrics = self.get_current_metrics()
        recent_metrics = self.get_metrics_history(hours=1)

        if not current_metrics:
            return {}

        # 计算平均值
        if recent_metrics:
            avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = np.mean([m.memory_percent for m in recent_metrics])
            avg_throughput = np.mean([m.throughput_per_hour for m in recent_metrics])
        else:
            avg_cpu = current_metrics.cpu_percent
            avg_memory = current_metrics.memory_percent
            avg_throughput = current_metrics.throughput_per_hour

        with self.stats_lock:
            stats = self.calculation_stats.copy()

        return {
            "current_status": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "disk_percent": current_metrics.disk_percent,
                "active_tasks": current_metrics.active_tasks,
            },
            "recent_performance": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "avg_throughput_per_hour": avg_throughput,
                "completed_tasks_1h": current_metrics.completed_tasks_1h,
                "failed_tasks_1h": current_metrics.failed_tasks_1h,
            },
            "overall_statistics": {
                "total_calculations": stats["total_calculations"],
                "successful_calculations": stats["successful_calculations"],
                "failed_calculations": stats["failed_calculations"],
                "success_rate": (stats["successful_calculations"] / max(1, stats["total_calculations"])) * 100,
                "total_execution_time_hours": stats["total_execution_time"].total_seconds() / 3600,
                "avg_execution_time_seconds": (
                    stats["total_execution_time"].total_seconds() / max(1, stats["total_calculations"])
                ),
            },
            "active_alerts": len(self.active_alerts),
            "last_updated": datetime.now().isoformat(),
        }

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self.alerts_lock:
            return [alert for alert in self.active_alerts.values() if not alert.resolved]

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self.alerts_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_time = datetime.now()

                # 更新数据库
                try:
                    query = text(
                        """
                        UPDATE calculation_alerts
                        SET resolved = :resolved, resolved_time = :resolved_time
                        WHERE alert_id = :alert_id
                    """
                    )

                    with self.engine.connect() as conn:
                        conn.execute(
                            query, {"resolved": True, "resolved_time": alert.resolved_time, "alert_id": alert_id}
                        )

                    logger.info(f"告警已解决: {alert.title}")
                    return True

                except Exception as e:
                    logger.error(f"更新告警状态失败: {e}")

            return False


class EmailAlertHandler:
    """邮件告警处理器"""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, recipients: List[str]):
        """初始化邮件告警处理器"""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients

    def __call__(self, alert: Alert):
        """处理告警"""
        try:
            # 创建邮件
            msg = MimeMultipart()
            msg["From"] = self.username
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"

            # 邮件内容
            body = f"""
告警详情:
- 告警ID: {alert.alert_id}
- 告警类型: {alert.alert_type.value}
- 告警级别: {alert.level.value}
- 发生时间: {alert.timestamp}
- 来源: {alert.source}

告警消息:
{alert.message}

请及时处理相关问题。
            """

            msg.attach(MimeText(body, "plain", "utf-8"))

            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"告警邮件已发送: {alert.title}")

        except Exception as e:
            logger.error(f"发送告警邮件失败: {e}")


class WebhookAlertHandler:
    """Webhook告警处理器"""

    def __init__(self, webhook_url: str):
        """初始化Webhook告警处理器"""
        self.webhook_url = webhook_url

    def __call__(self, alert: Alert):
        """处理告警"""
        try:
            import requests

            # 构建Webhook数据
            webhook_data = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metadata": alert.metadata,
            }

            # 发送Webhook
            response = requests.post(self.webhook_url, json=webhook_data, timeout=10)

            if response.status_code == 200:
                logger.info(f"Webhook告警已发送: {alert.title}")
            else:
                logger.error(f"Webhook告警发送失败: {response.status_code}")

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
