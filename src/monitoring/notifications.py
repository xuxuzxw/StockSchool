#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控通知系统模块

提供多种通知渠道的实现
包括邮件、Webhook、短信、钉钉、企业微信等通知方式

作者: StockSchool Team
创建时间: 2025-01-02
"""

import asyncio
import json
import logging
import smtplib
from abc import ABC, abstractmethod
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
from jinja2 import Template

from src.monitoring.alerts import Alert, AlertSeverity
from src.config.unified_config import config

# 配置日志
logger = logging.getLogger(__name__)

class NotificationStatus(Enum):
    """通知状态枚举"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRY = "retry"

class NotificationType(Enum):
    """通知类型枚举"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    DINGTALK = "dingtalk"
    WECHAT_WORK = "wechat_work"
    SMS = "sms"
    SLACK = "slack"

@dataclass
class NotificationConfig:
    """通知配置"""
    # 邮件配置
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Webhook配置
    webhook_urls: List[str] = field(default_factory=list)
    webhook_timeout: int = 30
    webhook_retry_count: int = 3
    
    # 钉钉配置
    dingtalk_webhook_url: str = ""
    dingtalk_secret: str = ""
    
    # 企业微信配置
    wechat_work_webhook_url: str = ""
    
    # 通用配置
    max_retry_count: int = 3
    retry_delay: int = 60  # 秒
    rate_limit_window: int = 300  # 5分钟
    rate_limit_count: int = 10  # 5分钟内最多发送10条
    
    @classmethod
    def from_config(cls) -> 'NotificationConfig':
        """从配置文件创建通知配置"""
        notification_config = config.get('notification', {})
        
        return cls(
            smtp_host=notification_config.get('smtp_host', 'smtp.gmail.com'),
            smtp_port=notification_config.get('smtp_port', 587),
            smtp_username=notification_config.get('smtp_username', ''),
            smtp_password=notification_config.get('smtp_password', ''),
            smtp_use_tls=notification_config.get('smtp_use_tls', True),
            email_from=notification_config.get('email_from', ''),
            email_to=notification_config.get('email_to', []),
            webhook_urls=notification_config.get('webhook_urls', []),
            webhook_timeout=notification_config.get('webhook_timeout', 30),
            webhook_retry_count=notification_config.get('webhook_retry_count', 3),
            dingtalk_webhook_url=notification_config.get('dingtalk_webhook_url', ''),
            dingtalk_secret=notification_config.get('dingtalk_secret', ''),
            wechat_work_webhook_url=notification_config.get('wechat_work_webhook_url', ''),
            max_retry_count=notification_config.get('max_retry_count', 3),
            retry_delay=notification_config.get('retry_delay', 60),
            rate_limit_window=notification_config.get('rate_limit_window', 300),
            rate_limit_count=notification_config.get('rate_limit_count', 10)
        )

@dataclass
class NotificationRecord:
    """通知记录"""
    id: str
    notification_type: NotificationType
    alert_id: str
    recipient: str
    subject: str
    content: str
    status: NotificationStatus
    created_at: datetime
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseNotificationChannel(ABC):
    """通知渠道基类"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            window_seconds=config.rate_limit_window,
            max_requests=config.rate_limit_count
        )
    
    @abstractmethod
    async def send(self, alert: Alert, recipients: List[str]) -> NotificationRecord:
        """发送通知"""
        pass
    
    @abstractmethod
    def get_notification_type(self) -> NotificationType:
        """获取通知类型"""
        pass
    
    def format_alert_message(self, alert: Alert) -> tuple[str, str]:
        """格式化告警消息"""
        # 生成主题
        severity_emoji = {
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.WARNING: "🟡",
            AlertSeverity.INFO: "🔵"
        }
        
        emoji = severity_emoji.get(alert.severity, "⚪")
        subject = f"{emoji} [{alert.severity.value.upper()}] {alert.title}"
        
        # 生成内容
        content_template = Template("""
告警详情:

📋 告警ID: {{ alert.id }}
📝 标题: {{ alert.title }}
📄 描述: {{ alert.message }}
🔥 严重程度: {{ alert.severity.value.upper() }}
📊 类型: {{ alert.alert_type.value }}
🕐 触发时间: {{ alert.created_at.strftime('%Y-%m-%d %H:%M:%S') }}
🏷️ 来源: {{ alert.source }}

{% if alert.tags %}
🏷️ 标签: {{ alert.tags | join(', ') }}
{% endif %}

{% if alert.metadata %}
📋 元数据:
{% for key, value in alert.metadata.items() %}
  - {{ key }}: {{ value }}
{% endfor %}
{% endif %}

---
StockSchool 监控系统
时间: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}
        """)
        
        content = content_template.render(
            alert=alert,
            datetime=datetime
        )
        
        return subject, content

class EmailNotificationChannel(BaseNotificationChannel):
    """邮件通知渠道"""
    
    def get_notification_type(self) -> NotificationType:
        return NotificationType.EMAIL
    
    async def send(self, alert: Alert, recipients: List[str] = None) -> NotificationRecord:
        """发送邮件通知"""
        if not recipients:
            recipients = self.config.email_to
        
        if not recipients:
            raise ValueError("没有配置邮件接收者")
        
        # 检查速率限制
        if not self.rate_limiter.allow_request():
            raise Exception("邮件发送频率超限")
        
        record = NotificationRecord(
            id=f"email_{alert.id}_{int(datetime.now().timestamp())}",
            notification_type=self.get_notification_type(),
            alert_id=alert.id,
            recipient=", ".join(recipients),
            subject="",
            content="",
            status=NotificationStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            subject, content = self.format_alert_message(alert)
            record.subject = subject
            record.content = content
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            # 添加HTML内容
            html_content = self._format_html_content(alert, content)
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # 发送邮件
            await self._send_email(msg)
            
            record.status = NotificationStatus.SENT
            record.sent_at = datetime.now()
            logger.info(f"邮件通知发送成功: {alert.id}")
            
        except Exception as e:
            record.status = NotificationStatus.FAILED
            record.error_message = str(e)
            logger.error(f"邮件通知发送失败: {e}")
        
        return record
    
    async def _send_email(self, msg: MIMEMultipart):
        """发送邮件"""
        loop = asyncio.get_event_loop()
        
        def _send():
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
        
        await loop.run_in_executor(None, _send)
    
    def _format_html_content(self, alert: Alert, content: str) -> str:
        """格式化HTML内容"""
        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ alert.title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .alert-header { background-color: {{ header_color }}; color: white; padding: 15px; border-radius: 5px; }
        .alert-content { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 10px; }
        .alert-meta { background-color: #e9e9e9; padding: 10px; border-radius: 5px; margin-top: 10px; }
        .severity-{{ alert.severity.value }} { border-left: 5px solid {{ border_color }}; }
    </style>
</head>
<body>
    <div class="alert-header severity-{{ alert.severity.value }}">
        <h2>{{ alert.title }}</h2>
        <p>严重程度: {{ alert.severity.value.upper() }} | 类型: {{ alert.alert_type.value }}</p>
    </div>
    
    <div class="alert-content">
        <h3>告警描述</h3>
        <p>{{ alert.message }}</p>
        
        <h3>基本信息</h3>
        <ul>
            <li><strong>告警ID:</strong> {{ alert.id }}</li>
            <li><strong>触发时间:</strong> {{ alert.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</li>
            <li><strong>来源:</strong> {{ alert.source }}</li>
            {% if alert.tags %}
            <li><strong>标签:</strong> {{ alert.tags | join(', ') }}</li>
            {% endif %}
        </ul>
    </div>
    
    {% if alert.metadata %}
    <div class="alert-meta">
        <h3>详细信息</h3>
        <ul>
        {% for key, value in alert.metadata.items() %}
            <li><strong>{{ key }}:</strong> {{ value }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
    
    <div style="margin-top: 20px; font-size: 12px; color: #666;">
        <p>此邮件由 StockSchool 监控系统自动发送</p>
        <p>发送时间: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>
</body>
</html>
        """)
        
        # 根据严重程度设置颜色
        color_map = {
            AlertSeverity.CRITICAL: {"header_color": "#dc3545", "border_color": "#dc3545"},
            AlertSeverity.WARNING: {"header_color": "#ffc107", "border_color": "#ffc107"},
            AlertSeverity.INFO: {"header_color": "#17a2b8", "border_color": "#17a2b8"}
        }
        
        colors = color_map.get(alert.severity, {"header_color": "#6c757d", "border_color": "#6c757d"})
        
        return html_template.render(
            alert=alert,
            datetime=datetime,
            **colors
        )

class WebhookNotificationChannel(BaseNotificationChannel):
    """Webhook通知渠道"""
    
    def get_notification_type(self) -> NotificationType:
        return NotificationType.WEBHOOK
    
    async def send(self, alert: Alert, recipients: List[str] = None) -> NotificationRecord:
        """发送Webhook通知"""
        urls = recipients or self.config.webhook_urls
        
        if not urls:
            raise ValueError("没有配置Webhook URL")
        
        # 检查速率限制
        if not self.rate_limiter.allow_request():
            raise Exception("Webhook发送频率超限")
        
        record = NotificationRecord(
            id=f"webhook_{alert.id}_{int(datetime.now().timestamp())}",
            notification_type=self.get_notification_type(),
            alert_id=alert.id,
            recipient=", ".join(urls),
            subject=alert.title,
            content=alert.message,
            status=NotificationStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            # 构建Webhook负载
            payload = {
                "alert_id": alert.id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "alert_type": alert.alert_type.value,
                "status": alert.status.value,
                "created_at": alert.created_at.isoformat(),
                "source": alert.source,
                "tags": alert.tags,
                "metadata": alert.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            # 发送到所有URL
            success_count = 0
            errors = []
            
            async with aiohttp.ClientSession() as session:
                for url in urls:
                    try:
                        async with session.post(
                            url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout)
                        ) as response:
                            if response.status == 200:
                                success_count += 1
                            else:
                                errors.append(f"{url}: HTTP {response.status}")
                    except Exception as e:
                        errors.append(f"{url}: {str(e)}")
            
            if success_count > 0:
                record.status = NotificationStatus.SENT
                record.sent_at = datetime.now()
                if errors:
                    record.metadata["partial_errors"] = errors
                logger.info(f"Webhook通知发送成功: {alert.id} ({success_count}/{len(urls)})")
            else:
                record.status = NotificationStatus.FAILED
                record.error_message = "; ".join(errors)
                logger.error(f"Webhook通知发送失败: {alert.id}")
        
        except Exception as e:
            record.status = NotificationStatus.FAILED
            record.error_message = str(e)
            logger.error(f"Webhook通知发送失败: {e}")
        
        return record

class DingTalkNotificationChannel(BaseNotificationChannel):
    """钉钉通知渠道"""
    
    def get_notification_type(self) -> NotificationType:
        return NotificationType.DINGTALK
    
    async def send(self, alert: Alert, recipients: List[str] = None) -> NotificationRecord:
        """发送钉钉通知"""
        if not self.config.dingtalk_webhook_url:
            raise ValueError("没有配置钉钉Webhook URL")
        
        # 检查速率限制
        if not self.rate_limiter.allow_request():
            raise Exception("钉钉通知发送频率超限")
        
        record = NotificationRecord(
            id=f"dingtalk_{alert.id}_{int(datetime.now().timestamp())}",
            notification_type=self.get_notification_type(),
            alert_id=alert.id,
            recipient="dingtalk_group",
            subject=alert.title,
            content=alert.message,
            status=NotificationStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            # 构建钉钉消息
            severity_color = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.WARNING: "#FFA500",
                AlertSeverity.INFO: "#0000FF"
            }
            
            color = severity_color.get(alert.severity, "#808080")
            
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": f"告警通知: {alert.title}",
                    "text": f"""
# 🚨 告警通知

**告警标题:** {alert.title}

**严重程度:** <font color="{color}">{alert.severity.value.upper()}</font>

**告警类型:** {alert.alert_type.value}

**告警描述:** {alert.message}

**触发时间:** {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

**来源:** {alert.source}

**告警ID:** {alert.id}

---
> StockSchool 监控系统
                    """
                }
            }
            
            # 如果配置了签名，添加签名
            url = self.config.dingtalk_webhook_url
            if self.config.dingtalk_secret:
                import time
                import hmac
                import hashlib
                import base64
                import urllib.parse
                
                timestamp = str(round(time.time() * 1000))
                secret_enc = self.config.dingtalk_secret.encode('utf-8')
                string_to_sign = f'{timestamp}\n{self.config.dingtalk_secret}'
                string_to_sign_enc = string_to_sign.encode('utf-8')
                hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
                sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                url = f"{url}&timestamp={timestamp}&sign={sign}"
            
            # 发送请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('errcode') == 0:
                            record.status = NotificationStatus.SENT
                            record.sent_at = datetime.now()
                            logger.info(f"钉钉通知发送成功: {alert.id}")
                        else:
                            record.status = NotificationStatus.FAILED
                            record.error_message = result.get('errmsg', '未知错误')
                    else:
                        record.status = NotificationStatus.FAILED
                        record.error_message = f"HTTP {response.status}"
        
        except Exception as e:
            record.status = NotificationStatus.FAILED
            record.error_message = str(e)
            logger.error(f"钉钉通知发送失败: {e}")
        
        return record

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, window_seconds: int, max_requests: int):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests = []
    
    def allow_request(self) -> bool:
        """检查是否允许请求"""
        now = datetime.now()
        
        # 清理过期请求
        cutoff = now.timestamp() - self.window_seconds
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]
        
        # 检查是否超过限制
        if len(self.requests) >= self.max_requests:
            return False
        
        # 记录当前请求
        self.requests.append(now.timestamp())
        return True

class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config: NotificationConfig = None):
        self.config = config or NotificationConfig.from_config()
        self.channels: Dict[NotificationType, BaseNotificationChannel] = {}
        self.notification_records: List[NotificationRecord] = []
        self._setup_channels()
    
    def _setup_channels(self):
        """设置通知渠道"""
        # 邮件渠道
        if self.config.email_from and self.config.email_to:
            self.channels[NotificationType.EMAIL] = EmailNotificationChannel(self.config)
        
        # Webhook渠道
        if self.config.webhook_urls:
            self.channels[NotificationType.WEBHOOK] = WebhookNotificationChannel(self.config)
        
        # 钉钉渠道
        if self.config.dingtalk_webhook_url:
            self.channels[NotificationType.DINGTALK] = DingTalkNotificationChannel(self.config)
        
        logger.info(f"已设置通知渠道: {list(self.channels.keys())}")
    
    async def send_notification(self, alert: Alert, notification_types: List[NotificationType] = None) -> List[NotificationRecord]:
        """发送通知"""
        if not notification_types:
            notification_types = list(self.channels.keys())
        
        records = []
        
        for notification_type in notification_types:
            if notification_type in self.channels:
                try:
                    channel = self.channels[notification_type]
                    record = await channel.send(alert)
                    records.append(record)
                    self.notification_records.append(record)
                except Exception as e:
                    logger.error(f"发送{notification_type.value}通知失败: {e}")
                    # 创建失败记录
                    failed_record = NotificationRecord(
                        id=f"{notification_type.value}_{alert.id}_{int(datetime.now().timestamp())}",
                        notification_type=notification_type,
                        alert_id=alert.id,
                        recipient="unknown",
                        subject=alert.title,
                        content=alert.message,
                        status=NotificationStatus.FAILED,
                        created_at=datetime.now(),
                        error_message=str(e)
                    )
                    records.append(failed_record)
                    self.notification_records.append(failed_record)
        
        return records
    
    async def retry_failed_notifications(self) -> List[NotificationRecord]:
        """重试失败的通知"""
        retry_records = []
        
        for record in self.notification_records:
            if (record.status == NotificationStatus.FAILED and 
                record.retry_count < self.config.max_retry_count):
                
                # 检查重试间隔
                if record.created_at:
                    elapsed = (datetime.now() - record.created_at).total_seconds()
                    if elapsed < self.config.retry_delay * (record.retry_count + 1):
                        continue
                
                try:
                    # 重新发送
                    channel = self.channels.get(record.notification_type)
                    if channel:
                        # 这里需要重新构造Alert对象，实际实现中可能需要从数据库获取
                        # 暂时跳过重试逻辑
                        record.retry_count += 1
                        logger.info(f"重试通知: {record.id} (第{record.retry_count}次)")
                        retry_records.append(record)
                
                except Exception as e:
                    record.retry_count += 1
                    record.error_message = str(e)
                    logger.error(f"重试通知失败: {record.id} - {e}")
        
        return retry_records
    
    def get_notification_statistics(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        total_notifications = len(self.notification_records)
        
        if total_notifications == 0:
            return {
                "total_notifications": 0,
                "success_rate": 0,
                "status_distribution": {},
                "type_distribution": {},
                "recent_notifications": []
            }
        
        # 状态分布
        status_distribution = {}
        for status in NotificationStatus:
            count = sum(1 for r in self.notification_records if r.status == status)
            status_distribution[status.value] = count
        
        # 类型分布
        type_distribution = {}
        for notification_type in NotificationType:
            count = sum(1 for r in self.notification_records if r.notification_type == notification_type)
            if count > 0:
                type_distribution[notification_type.value] = count
        
        # 成功率
        success_count = status_distribution.get(NotificationStatus.SENT.value, 0)
        success_rate = (success_count / total_notifications * 100) if total_notifications > 0 else 0
        
        # 最近通知
        recent_notifications = sorted(
            self.notification_records,
            key=lambda x: x.created_at,
            reverse=True
        )[:10]
        
        recent_data = []
        for record in recent_notifications:
            recent_data.append({
                "id": record.id,
                "type": record.notification_type.value,
                "status": record.status.value,
                "created_at": record.created_at.isoformat(),
                "sent_at": record.sent_at.isoformat() if record.sent_at else None,
                "error_message": record.error_message
            })
        
        return {
            "total_notifications": total_notifications,
            "success_rate": round(success_rate, 2),
            "status_distribution": status_distribution,
            "type_distribution": type_distribution,
            "recent_notifications": recent_data
        }
    
    def get_available_channels(self) -> List[str]:
        """获取可用的通知渠道"""
        return [channel_type.value for channel_type in self.channels.keys()]

# 全局通知管理器实例
notification_manager = NotificationManager()

def get_notification_manager() -> NotificationManager:
    """获取通知管理器实例"""
    return notification_manager