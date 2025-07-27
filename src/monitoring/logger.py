import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger
from pathlib import Path
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
from dataclasses import dataclass
from enum import Enum
import threading
import time
from queue import Queue
import traceback
from src.utils.config_loader import config

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertLevel(Enum):
    """告警级别枚举"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    """告警信息数据类"""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata or {}
        }

class LoggerConfig:
    """日志配置类"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """初始化日志配置"""
        self.config = config_dict or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'log_dir': 'logs',
            'log_level': 'INFO',
            'max_file_size': '10 MB',
            'backup_count': 5,
            'log_format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
            'console_output': True,
            'file_output': True,
            'json_output': False,
            'rotation': 'daily',
            'compression': 'gz'
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self.config.get(key, default)

class EmailNotifier:
    """邮件通知器"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, use_tls: bool = True):
        """初始化邮件通知器"""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
    
    def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """发送告警邮件"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.level.value}] {alert.title}"
            
            # 邮件正文
            body = f"""
告警级别: {alert.level.value}
告警时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
告警来源: {alert.source}

告警信息:
{alert.message}

详细信息:
{json.dumps(alert.metadata or {}, indent=2, ensure_ascii=False)}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
        
        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            return False

class WebhookNotifier:
    """Webhook通知器"""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        """初始化Webhook通知器"""
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def send_alert(self, alert: Alert) -> bool:
        """发送告警到Webhook"""
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            response.raise_for_status()
            return True
        
        except Exception as e:
            logger.error(f"发送Webhook失败: {e}")
            return False

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        """初始化告警管理器"""
        self.notifiers = []
        self.alert_queue = Queue()
        self.alert_history = []
        self.max_history = config.get('monitoring.max_alert_history', 1000)
        self.alert_rules = {}
        self.suppression_rules = {}
        self._running = False
        self._worker_thread = None
    
    def add_notifier(self, notifier):
        """添加通知器"""
        self.notifiers.append(notifier)
    
    def add_alert_rule(self, rule_name: str, condition: callable, 
                      alert_level: AlertLevel, title: str, message: str):
        """添加告警规则"""
        self.alert_rules[rule_name] = {
            'condition': condition,
            'level': alert_level,
            'title': title,
            'message': message,
            'last_triggered': None
        }
    
    def add_suppression_rule(self, rule_name: str, duration_minutes: int):
        """添加告警抑制规则"""
        self.suppression_rules[rule_name] = duration_minutes
    
    def send_alert(self, alert: Alert):
        """发送告警"""
        # 检查抑制规则
        if self._is_suppressed(alert):
            logger.debug(f"告警被抑制: {alert.title}")
            return
        
        # 添加到队列
        self.alert_queue.put(alert)
        
        # 添加到历史记录
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """检查告警是否被抑制"""
        # 简单的抑制逻辑：相同标题的告警在指定时间内只发送一次
        suppression_duration = self.suppression_rules.get(alert.title, 0)
        if suppression_duration == 0:
            return False
        
        # 检查历史记录
        cutoff_time = datetime.now().timestamp() - (suppression_duration * 60)
        for hist_alert in reversed(self.alert_history):
            if hist_alert.timestamp.timestamp() < cutoff_time:
                break
            if hist_alert.title == alert.title:
                return True
        
        return False
    
    def start(self):
        """启动告警管理器"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("告警管理器已启动")
    
    def stop(self):
        """停止告警管理器"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("告警管理器已停止")
    
    def _worker(self):
        """工作线程"""
        while self._running:
            try:
                # 处理告警队列
                if not self.alert_queue.empty():
                    alert = self.alert_queue.get(timeout=1)
                    self._process_alert(alert)
                else:
                    time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"告警处理异常: {e}")
                time.sleep(1)
    
    def _process_alert(self, alert: Alert):
        """处理单个告警"""
        logger.info(f"处理告警: {alert.title} [{alert.level.value}]")
        
        # 发送到所有通知器
        for notifier in self.notifiers:
            try:
                if hasattr(notifier, 'send_alert'):
                    if isinstance(notifier, EmailNotifier):
                        # 邮件通知器需要收件人列表
                        recipients = alert.metadata.get('email_recipients', [])
                        if recipients:
                            notifier.send_alert(alert, recipients)
                    else:
                        notifier.send_alert(alert)
            
            except Exception as e:
                logger.error(f"通知器发送失败: {e}")
    
    def check_rules(self, data: Dict[str, Any]):
        """检查告警规则"""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](data):
                    # 检查是否需要抑制
                    last_triggered = rule.get('last_triggered')
                    suppression_duration = self.suppression_rules.get(rule_name, 0)
                    
                    if (last_triggered is None or 
                        (datetime.now() - last_triggered).total_seconds() > suppression_duration * 60):
                        
                        alert = Alert(
                            level=rule['level'],
                            title=rule['title'],
                            message=rule['message'],
                            timestamp=datetime.now(),
                            source=f"rule:{rule_name}",
                            metadata=data
                        )
                        
                        self.send_alert(alert)
                        rule['last_triggered'] = datetime.now()
            
            except Exception as e:
                logger.error(f"检查规则 {rule_name} 失败: {e}")

class StockSchoolLogger:
    """StockSchool专用日志器"""
    
    def __init__(self, config: LoggerConfig = None):
        """初始化日志器"""
        self.config = config or LoggerConfig()
        self.alert_manager = AlertManager()
        self._setup_logger()
        self._setup_alert_manager()
    
    def _setup_logger(self):
        """设置日志器"""
        # 移除默认处理器
        logger.remove()
        
        # 日志格式
        log_format = self.config.get('log_format')
        
        # 控制台输出
        if self.config.get('console_output', True):
            logger.add(
                sys.stdout,
                format=log_format,
                level=self.config.get('log_level', 'INFO'),
                colorize=True
            )
        
        # 文件输出
        if self.config.get('file_output', True):
            log_dir = Path(self.config.get('log_dir', 'logs'))
            log_dir.mkdir(exist_ok=True)
            
            # 主日志文件
            logger.add(
                log_dir / "stockschool_{time:YYYY-MM-DD}.log",
                format=log_format,
                level=self.config.get('log_level', 'INFO'),
                rotation=self.config.get('rotation', 'daily'),
                retention=f"{self.config.get('backup_count', 5)} days",
                compression=self.config.get('compression', 'gz'),
                encoding='utf-8'
            )
            
            # 错误日志文件
            logger.add(
                log_dir / "error_{time:YYYY-MM-DD}.log",
                format=log_format,
                level="ERROR",
                rotation="daily",
                retention="30 days",
                compression="gz",
                encoding='utf-8'
            )
        
        # JSON格式输出
        if self.config.get('json_output', False):
            log_dir = Path(self.config.get('log_dir', 'logs'))
            log_dir.mkdir(exist_ok=True)
            
            logger.add(
                log_dir / "stockschool_{time:YYYY-MM-DD}.json",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
                level=self.config.get('log_level', 'INFO'),
                rotation="daily",
                retention="7 days",
                serialize=True,
                encoding='utf-8'
            )
    
    def _setup_alert_manager(self):
        """设置告警管理器"""
        # 添加默认告警规则
        error_threshold = config.get('monitoring.error_threshold', 10)
        self.alert_manager.add_alert_rule(
            'high_error_rate',
            lambda data: data.get('error_count', 0) > error_threshold,
            AlertLevel.HIGH,
            '错误率过高',
            '系统错误数量超过阈值'
        )
        
        self.alert_manager.add_alert_rule(
            'data_sync_failure',
            lambda data: data.get('sync_status') == 'failed',
            AlertLevel.CRITICAL,
            '数据同步失败',
            '数据同步过程发生严重错误'
        )
        
        self.alert_manager.add_alert_rule(
            'database_connection_error',
            lambda data: 'database' in data.get('error_type', '').lower(),
            AlertLevel.HIGH,
            '数据库连接错误',
            '数据库连接出现问题'
        )
        
        # 添加抑制规则（分钟）
        self.alert_manager.add_suppression_rule('错误率过高', config.get('monitoring.suppression.error_rate', 30))
        self.alert_manager.add_suppression_rule('数据同步失败', config.get('monitoring.suppression.sync_failure', 60))
        self.alert_manager.add_suppression_rule('数据库连接错误', config.get('monitoring.suppression.db_error', 15))
        
        # 启动告警管理器
        self.alert_manager.start()
    
    def add_email_notifier(self, smtp_server: str, smtp_port: int, 
                          username: str, password: str, use_tls: bool = True):
        """添加邮件通知器"""
        notifier = EmailNotifier(smtp_server, smtp_port, username, password, use_tls)
        self.alert_manager.add_notifier(notifier)
    
    def add_webhook_notifier(self, webhook_url: str, headers: Dict[str, str] = None):
        """添加Webhook通知器"""
        notifier = WebhookNotifier(webhook_url, headers)
        self.alert_manager.add_notifier(notifier)
    
    def log_with_alert(self, level: LogLevel, message: str, 
                      alert_level: AlertLevel = None, 
                      alert_title: str = None,
                      metadata: Dict[str, Any] = None):
        """记录日志并可能发送告警"""
        # 记录日志
        getattr(logger, level.value.lower())(message)
        
        # 发送告警
        if alert_level and alert_title:
            alert = Alert(
                level=alert_level,
                title=alert_title,
                message=message,
                timestamp=datetime.now(),
                source='manual',
                metadata=metadata
            )
            self.alert_manager.send_alert(alert)
    
    def log_exception(self, exc: Exception, context: str = None, 
                     send_alert: bool = True):
        """记录异常"""
        exc_info = {
            'exception_type': type(exc).__name__,
            'exception_message': str(exc),
            'traceback': traceback.format_exc(),
            'context': context
        }
        
        message = f"异常发生: {type(exc).__name__}: {str(exc)}"
        if context:
            message = f"{context} - {message}"
        
        logger.error(message)
        
        if send_alert:
            alert = Alert(
                level=AlertLevel.HIGH,
                title=f"系统异常: {type(exc).__name__}",
                message=message,
                timestamp=datetime.now(),
                source='exception_handler',
                metadata=exc_info
            )
            self.alert_manager.send_alert(alert)
    
    def log_performance(self, operation: str, duration: float, 
                       threshold: float = None):
        """记录性能指标"""
        message = f"操作 '{operation}' 耗时: {duration:.3f}秒"
        
        if threshold and duration > threshold:
            logger.warning(f"性能告警: {message} (超过阈值 {threshold}秒)")
            
            alert = Alert(
                level=AlertLevel.MEDIUM,
                title=f"性能告警: {operation}",
                message=f"操作耗时 {duration:.3f}秒，超过阈值 {threshold}秒",
                timestamp=datetime.now(),
                source='performance_monitor',
                metadata={
                    'operation': operation,
                    'duration': duration,
                    'threshold': threshold
                }
            )
            self.alert_manager.send_alert(alert)
        else:
            logger.info(message)
    
    def check_system_health(self, metrics: Dict[str, Any]):
        """检查系统健康状态"""
        logger.info(f"系统健康检查: {metrics}")
        self.alert_manager.check_rules(metrics)
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警历史"""
        return [alert.to_dict() for alert in self.alert_manager.alert_history[-limit:]]
    
    def shutdown(self):
        """关闭日志器"""
        self.alert_manager.stop()
        logger.info("日志系统已关闭")

# 全局日志器实例
_global_logger = None

def get_logger(config: LoggerConfig = None) -> StockSchoolLogger:
    """获取全局日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = StockSchoolLogger(config)
    return _global_logger

def setup_logging(config_dict: Dict[str, Any] = None) -> StockSchoolLogger:
    """设置日志系统"""
    config = LoggerConfig(config_dict)
    return get_logger(config)

if __name__ == '__main__':
    # 测试代码
    print("测试日志和告警系统...")
    
    # 创建日志器
    log_config = {
        'log_level': 'DEBUG',
        'console_output': True,
        'file_output': True,
        'log_dir': 'test_logs'
    }
    
    stock_logger = setup_logging(log_config)
    
    # 测试基本日志
    logger.info("系统启动")
    logger.debug("调试信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    
    # 测试异常日志
    try:
        raise ValueError("测试异常")
    except Exception as e:
        stock_logger.log_exception(e, "测试异常处理")
    
    # 测试性能日志
    stock_logger.log_performance("数据同步", 2.5, threshold=2.0)
    
    # 测试系统健康检查
    health_metrics = {
        'error_count': 15,  # 触发告警
        'sync_status': 'success',
        'cpu_usage': 75.5,
        'memory_usage': 60.2
    }
    stock_logger.check_system_health(health_metrics)
    
    # 测试手动告警
    stock_logger.log_with_alert(
        LogLevel.ERROR,
        "手动测试告警",
        AlertLevel.MEDIUM,
        "测试告警",
        {'test': True}
    )
    
    # 等待告警处理
    time.sleep(2)
    
    # 获取告警历史
    alert_history = stock_logger.get_alert_history()
    print(f"\n告警历史记录数量: {len(alert_history)}")
    for alert in alert_history:
        print(f"  {alert['timestamp']}: {alert['title']} [{alert['level']}]")
    
    # 关闭系统
    stock_logger.shutdown()
    
    print("\n日志和告警系统测试完成!")