import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
import time
from queue import Queue, Empty
import sqlite3
from pathlib import Path
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger
from src.utils.config_loader import config
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"

class AlertType(Enum):
    """告警类型"""
    SYSTEM = "SYSTEM"
    DATA_QUALITY = "DATA_QUALITY"
    PERFORMANCE = "PERFORMANCE"
    BUSINESS = "BUSINESS"
    SECURITY = "SECURITY"

@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # 条件表达式
    threshold: float
    comparison: str  # '>', '<', '>=', '<=', '==', '!='
    evaluation_window: int  # 评估窗口（秒）
    trigger_count: int  # 触发次数阈值
    enabled: bool = True
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Alert:
    """告警实例"""
    id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    alert_type: AlertType
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    source: str = ""
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换datetime对象
        for key in ['created_at', 'updated_at', 'resolved_at', 'acknowledged_at']:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        # 转换枚举
        data['severity'] = self.severity.value
        data['alert_type'] = self.alert_type.value
        data['status'] = self.status.value
        return data

class MetricCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = None):
        if max_history is None:
            max_history = config.get('monitoring.max_metric_history', 1000)
        """初始化指标收集器"""
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, timestamp: datetime = None):
        """记录指标"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            self.metrics[name].append((timestamp, value))
    
    def get_metric_values(self, name: str, window_seconds: int = None) -> List[Tuple[datetime, float]]:
        """获取指标值"""
        with self.lock:
            values = list(self.metrics[name])
        
        if window_seconds is not None:
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            values = [(ts, val) for ts, val in values if ts >= cutoff_time]
        
        return values
    
    def get_metric_stats(self, name: str, window_seconds: int = None) -> Dict[str, float]:
        """获取指标统计信息"""
        values = self.get_metric_values(name, window_seconds)
        
        if not values:
            return {}
        
        numeric_values = [val for _, val in values]
        
        return {
            'count': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': np.mean(numeric_values),
            'median': np.median(numeric_values),
            'std': np.std(numeric_values),
            'latest': numeric_values[-1] if numeric_values else 0
        }

class AlertStorage:
    """告警存储"""
    
    def __init__(self, db_path: str = "alerts.db"):
        """初始化告警存储"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    condition_expr TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    comparison TEXT NOT NULL,
                    evaluation_window INTEGER NOT NULL,
                    trigger_count INTEGER NOT NULL,
                    enabled BOOLEAN NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    resolved_at TIMESTAMP,
                    acknowledged_at TIMESTAMP,
                    acknowledged_by TEXT,
                    source TEXT,
                    tags TEXT,
                    metadata TEXT,
                    FOREIGN KEY (rule_id) REFERENCES alert_rules (id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at)
            """)
    
    def save_rule(self, rule: AlertRule):
        """保存告警规则"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alert_rules 
                (id, name, description, alert_type, severity, condition_expr, 
                 threshold, comparison, evaluation_window, trigger_count, enabled, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.id, rule.name, rule.description, rule.alert_type.value,
                rule.severity.value, rule.condition, rule.threshold, rule.comparison,
                rule.evaluation_window, rule.trigger_count, rule.enabled,
                json.dumps(rule.tags), json.dumps(rule.metadata)
            ))
    
    def load_rules(self) -> List[AlertRule]:
        """加载告警规则"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, name, description, alert_type, severity, condition_expr,
                       threshold, comparison, evaluation_window, trigger_count, enabled, tags, metadata
                FROM alert_rules WHERE enabled = 1
            """)
            
            rules = []
            for row in cursor.fetchall():
                rules.append(AlertRule(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    alert_type=AlertType(row[3]),
                    severity=AlertSeverity(row[4]),
                    condition=row[5],
                    threshold=row[6],
                    comparison=row[7],
                    evaluation_window=row[8],
                    trigger_count=row[9],
                    enabled=bool(row[10]),
                    tags=json.loads(row[11]) if row[11] else [],
                    metadata=json.loads(row[12]) if row[12] else {}
                ))
            
            return rules
    
    def save_alert(self, alert: Alert):
        """保存告警"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts 
                (id, rule_id, title, message, severity, alert_type, status,
                 created_at, updated_at, resolved_at, acknowledged_at, acknowledged_by,
                 source, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id, alert.rule_id, alert.title, alert.message,
                alert.severity.value, alert.alert_type.value, alert.status.value,
                alert.created_at.isoformat(), alert.updated_at.isoformat(),
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.acknowledged_by, alert.source,
                json.dumps(alert.tags), json.dumps(alert.metadata)
            ))
    
    def load_alerts(self, status: AlertStatus = None, limit: int = None) -> List[Alert]:
        if limit is None:
            limit = config.get('database_params.default_limit', 100)
        """加载告警"""
        with sqlite3.connect(self.db_path) as conn:
            if status:
                cursor = conn.execute("""
                    SELECT * FROM alerts WHERE status = ? 
                    ORDER BY created_at DESC LIMIT ?
                """, (status.value, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append(Alert(
                    id=row[0],
                    rule_id=row[1],
                    title=row[2],
                    message=row[3],
                    severity=AlertSeverity(row[4]),
                    alert_type=AlertType(row[5]),
                    status=AlertStatus(row[6]),
                    created_at=datetime.fromisoformat(row[7]),
                    updated_at=datetime.fromisoformat(row[8]),
                    resolved_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    acknowledged_at=datetime.fromisoformat(row[10]) if row[10] else None,
                    acknowledged_by=row[11],
                    source=row[12],
                    tags=json.loads(row[13]) if row[13] else [],
                    metadata=json.loads(row[14]) if row[14] else {}
                ))
            
            return alerts

class NotificationChannel:
    """通知渠道基类"""
    
    def send(self, alert: Alert) -> bool:
        """发送通知"""
        raise NotImplementedError

class EmailChannel(NotificationChannel):
    """邮件通知渠道"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, recipients: List[str], use_tls: bool = True):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        self.use_tls = use_tls
    
    def send(self, alert: Alert) -> bool:
        """发送邮件通知"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[{alert.severity.value}] {alert.title}"
            
            body = f"""
告警详情:

标题: {alert.title}
严重程度: {alert.severity.value}
类型: {alert.alert_type.value}
状态: {alert.status.value}
创建时间: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
来源: {alert.source}

消息:
{alert.message}

元数据:
{json.dumps(alert.metadata, indent=2, ensure_ascii=False)}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
        
        except Exception as e:
            logger.error(f"发送邮件通知失败: {e}")
            return False

class WebhookChannel(NotificationChannel):
    """Webhook通知渠道"""
    
    def __init__(self, url: str, headers: Dict[str, str] = None, timeout: int = None):
        if timeout is None:
            timeout = config.get('monitoring.webhook_timeout', 10)
        self.url = url
        self.headers = headers or {'Content-Type': 'application/json'}
        self.timeout = timeout
    
    def send(self, alert: Alert) -> bool:
        """发送Webhook通知"""
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return True
        
        except Exception as e:
            logger.error(f"发送Webhook通知失败: {e}")
            return False

class AlertEngine:
    """告警引擎"""
    
    def __init__(self, storage: AlertStorage = None):
        """初始化告警引擎"""
        self.storage = storage or AlertStorage()
        self.metric_collector = MetricCollector()
        self.notification_channels = []
        self.rules = {}
        self.rule_states = defaultdict(lambda: {'trigger_count': 0, 'last_triggered': None})
        self.alert_queue = Queue()
        self.running = False
        self.worker_thread = None
        self.suppression_rules = {}  # 抑制规则
        
        # 加载规则
        self._load_rules()
    
    def _load_rules(self):
        """加载告警规则"""
        rules = self.storage.load_rules()
        for rule in rules:
            self.rules[rule.id] = rule
        logger.info(f"加载了 {len(rules)} 个告警规则")
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.id] = rule
        self.storage.save_rule(rule)
        logger.info(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"移除告警规则: {rule_id}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """添加通知渠道"""
        self.notification_channels.append(channel)
    
    def add_suppression_rule(self, alert_title: str, duration_minutes: int):
        """添加抑制规则"""
        self.suppression_rules[alert_title] = duration_minutes
    
    def record_metric(self, name: str, value: float, timestamp: datetime = None):
        """记录指标"""
        self.metric_collector.record_metric(name, value, timestamp)
        
        # 检查告警规则
        self._evaluate_rules(name, value)
    
    def _evaluate_rules(self, metric_name: str, current_value: float):
        """评估告警规则"""
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # 检查条件是否匹配
            if rule.condition != metric_name:
                continue
            
            # 评估阈值条件
            triggered = self._evaluate_condition(current_value, rule.threshold, rule.comparison)
            
            if triggered:
                self._handle_rule_trigger(rule, metric_name, current_value)
            else:
                # 重置触发计数
                self.rule_states[rule.id]['trigger_count'] = 0
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """评估条件"""
        if comparison == '>':
            return value > threshold
        elif comparison == '<':
            return value < threshold
        elif comparison == '>=':
            return value >= threshold
        elif comparison == '<=':
            return value <= threshold
        elif comparison == '==':
            return value == threshold
        elif comparison == '!=':
            return value != threshold
        else:
            return False
    
    def _handle_rule_trigger(self, rule: AlertRule, metric_name: str, current_value: float):
        """处理规则触发"""
        rule_state = self.rule_states[rule.id]
        rule_state['trigger_count'] += 1
        
        # 检查是否达到触发次数阈值
        if rule_state['trigger_count'] >= rule.trigger_count:
            # 检查评估窗口
            now = datetime.now()
            if (rule_state['last_triggered'] is None or 
                (now - rule_state['last_triggered']).total_seconds() >= rule.evaluation_window):
                
                # 创建告警
                alert = self._create_alert(rule, metric_name, current_value)
                
                # 检查抑制规则
                if not self._is_suppressed(alert):
                    self.alert_queue.put(alert)
                    rule_state['last_triggered'] = now
                    rule_state['trigger_count'] = 0
    
    def _create_alert(self, rule: AlertRule, metric_name: str, current_value: float) -> Alert:
        """创建告警"""
        alert_id = f"{rule.id}_{int(datetime.now().timestamp())}"
        
        message = f"指标 '{metric_name}' 当前值 {current_value} {rule.comparison} 阈值 {rule.threshold}"
        
        return Alert(
            id=alert_id,
            rule_id=rule.id,
            title=rule.name,
            message=message,
            severity=rule.severity,
            alert_type=rule.alert_type,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="alert_engine",
            tags=rule.tags.copy(),
            metadata={
                'metric_name': metric_name,
                'current_value': current_value,
                'threshold': rule.threshold,
                'comparison': rule.comparison,
                'rule_metadata': rule.metadata
            }
        )
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """检查告警是否被抑制"""
        suppression_duration = self.suppression_rules.get(alert.title, 0)
        if suppression_duration == 0:
            return False
        
        # 检查最近的告警
        alert_limit = config.get('monitoring.alert_check_limit', 100)
        recent_alerts = self.storage.load_alerts(AlertStatus.ACTIVE, limit=alert_limit)
        cutoff_time = datetime.now() - timedelta(minutes=suppression_duration)
        
        for recent_alert in recent_alerts:
            if (recent_alert.title == alert.title and 
                recent_alert.created_at >= cutoff_time):
                return True
        
        return False
    
    def start(self):
        """启动告警引擎"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("告警引擎已启动")
    
    def stop(self):
        """停止告警引擎"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("告警引擎已停止")
    
    def _worker(self):
        """工作线程"""
        while self.running:
            try:
                # 处理告警队列
                try:
                    alert = self.alert_queue.get(timeout=1)
                    self._process_alert(alert)
                except Empty:
                    continue
            
            except Exception as e:
                logger.error(f"告警处理异常: {e}")
                time.sleep(1)
    
    def _process_alert(self, alert: Alert):
        """处理告警"""
        logger.info(f"处理告警: {alert.title} [{alert.severity.value}]")
        
        # 保存告警
        self.storage.save_alert(alert)
        
        # 发送通知
        for channel in self.notification_channels:
            try:
                success = channel.send(alert)
                if success:
                    logger.debug(f"通知发送成功: {type(channel).__name__}")
                else:
                    logger.warning(f"通知发送失败: {type(channel).__name__}")
            except Exception as e:
                logger.error(f"发送通知异常: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认告警"""
        alerts = self.storage.load_alerts()
        for alert in alerts:
            if alert.id == alert_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                alert.updated_at = datetime.now()
                self.storage.save_alert(alert)
                logger.info(f"告警已确认: {alert_id} by {acknowledged_by}")
                break
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        alerts = self.storage.load_alerts()
        for alert in alerts:
            if alert.id == alert_id and alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.updated_at = datetime.now()
                self.storage.save_alert(alert)
                logger.info(f"告警已解决: {alert_id}")
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return self.storage.load_alerts(AlertStatus.ACTIVE)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        stats_limit = config.get('monitoring.stats_limit', 1000)
        all_alerts = self.storage.load_alerts(limit=stats_limit)
        
        stats = {
            'total_alerts': len(all_alerts),
            'active_alerts': len([a for a in all_alerts if a.status == AlertStatus.ACTIVE]),
            'acknowledged_alerts': len([a for a in all_alerts if a.status == AlertStatus.ACKNOWLEDGED]),
            'resolved_alerts': len([a for a in all_alerts if a.status == AlertStatus.RESOLVED]),
            'severity_distribution': {},
            'type_distribution': {},
            'recent_24h': 0
        }
        
        # 统计严重程度分布
        for severity in AlertSeverity:
            stats['severity_distribution'][severity.value] = len([
                a for a in all_alerts if a.severity == severity
            ])
        
        # 统计类型分布
        for alert_type in AlertType:
            stats['type_distribution'][alert_type.value] = len([
                a for a in all_alerts if a.alert_type == alert_type
            ])
        
        # 统计最近24小时的告警
        cutoff_time = datetime.now() - timedelta(hours=24)
        stats['recent_24h'] = len([
            a for a in all_alerts if a.created_at >= cutoff_time
        ])
        
        return stats

if __name__ == '__main__':
    # 测试代码
    print("测试告警系统...")
    
    # 创建告警引擎
    engine = AlertEngine()
    
    # 添加测试规则
    rule1 = AlertRule(
        id="cpu_high",
        name="CPU使用率过高",
        description="CPU使用率超过80%",
        alert_type=AlertType.SYSTEM,
        severity=AlertSeverity.WARNING,
        condition="cpu_usage",
        threshold=80.0,
        comparison=">",
        evaluation_window=config.get('monitoring_params.evaluation_window', 60),
        trigger_count=3
    )
    
    rule2 = AlertRule(
        id="error_rate_high",
        name="错误率过高",
        description="错误率超过5%",
        alert_type=AlertType.SYSTEM,
        severity=AlertSeverity.ERROR,
        condition="error_rate",
        threshold=5.0,
        comparison=">",
        evaluation_window=300,
        trigger_count=2
    )
    
    engine.add_rule(rule1)
    engine.add_rule(rule2)
    
    # 添加抑制规则
    engine.add_suppression_rule("CPU使用率过高", 30)  # 30分钟内不重复告警
    suppression_duration = config.get('monitoring_params.suppression_duration', 60)
    engine.add_suppression_rule("错误率过高", suppression_duration)   # N分钟内不重复告警
    
    # 启动引擎
    engine.start()
    
    # 模拟指标数据
    print("\n模拟指标数据...")
    
    # 正常CPU使用率
    for i in range(5):
        engine.record_metric("cpu_usage", 70.0 + i)
        time.sleep(0.1)
    
    # 高CPU使用率（触发告警）
    for i in range(5):
        engine.record_metric("cpu_usage", 85.0 + i)
        time.sleep(0.1)
    
    # 高错误率（触发告警）
    for i in range(3):
        engine.record_metric("error_rate", 8.0 + i)
        time.sleep(0.1)
    
    # 等待处理
    time.sleep(2)
    
    # 查看活跃告警
    active_alerts = engine.get_active_alerts()
    print(f"\n活跃告警数量: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"  {alert.title}: {alert.message} [{alert.severity.value}]")
    
    # 查看统计信息
    stats = engine.get_alert_statistics()
    print(f"\n告警统计:")
    print(f"  总告警数: {stats['total_alerts']}")
    print(f"  活跃告警: {stats['active_alerts']}")
    print(f"  已确认告警: {stats['acknowledged_alerts']}")
    print(f"  已解决告警: {stats['resolved_alerts']}")
    print(f"  最近24小时: {stats['recent_24h']}")
    
    # 确认第一个告警
    if active_alerts:
        engine.acknowledge_alert(active_alerts[0].id, "test_user")
        print(f"\n已确认告警: {active_alerts[0].title}")
    
    # 停止引擎
    engine.stop()
    
    print("\n告警系统测试完成!")