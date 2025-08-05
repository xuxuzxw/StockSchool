import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统配置模块

提供监控系统的配置管理功能，包括：
- 监控参数配置
- 告警规则配置
- 通知渠道配置
- 性能监控配置
"""


@dataclass
class MonitoringConfig:
    """监控系统配置"""

    # 系统监控配置
    system_check_interval: int = 30  # 系统检查间隔（秒）
    health_check_timeout: int = 10  # 健康检查超时（秒）

    # 数据库监控配置
    db_connection_timeout: int = 5  # 数据库连接超时（秒）
    db_query_timeout: int = 30  # 数据库查询超时（秒）

    # Redis监控配置
    redis_connection_timeout: int = 5  # Redis连接超时（秒）
    redis_command_timeout: int = 10  # Redis命令超时（秒）

    # API监控配置
    api_timeout: int = 30  # API超时（秒）
    api_check_endpoints: List[str] = field(
        default_factory=lambda: ["/api/v1/health", "/api/v1/stocks/basic", "/api/v1/factors/list"]
    )

    # Celery监控配置
    celery_broker_timeout: int = 10  # Celery broker超时（秒）

    # 告警配置
    alert_check_interval: int = 60  # 告警检查间隔（秒）
    alert_batch_size: int = 100  # 告警批处理大小
    alert_retention_days: int = 30  # 告警保留天数

    # 性能监控配置
    performance_collection_interval: int = 60  # 性能数据收集间隔（秒）
    performance_retention_days: int = 7  # 性能数据保留天数

    # WebSocket配置
    websocket_heartbeat_interval: int = 30  # WebSocket心跳间隔（秒）
    websocket_connection_timeout: int = 300  # WebSocket连接超时（秒）

    # 通知配置
    notification_retry_count: int = 3  # 通知重试次数
    notification_retry_delay: int = 60  # 通知重试延迟（秒）
    notification_rate_limit: int = 10  # 通知频率限制（每分钟）

    # 数据质量监控配置
    data_quality_check_interval: int = 300  # 数据质量检查间隔（秒）
    data_quality_db_path: str = "data/monitoring/data_quality.db"  # 数据质量数据库路径
    data_quality_retention_days: int = 90  # 数据质量记录保留天数
    data_quality_batch_size: int = 1000  # 数据质量检查批处理大小

    # 阈值配置
    thresholds: Dict[str, Any] = field(
        default_factory=lambda: {
            "cpu_usage": 80.0,  # CPU使用率阈值（%）
            "memory_usage": 85.0,  # 内存使用率阈值（%）
            "disk_usage": 90.0,  # 磁盘使用率阈值（%）
            "response_time": 5000.0,  # 响应时间阈值（毫秒）
            "error_rate": 5.0,  # 错误率阈值（%）
            "db_connections": 80,  # 数据库连接数阈值
            "queue_size": 1000,  # 队列大小阈值
            # 数据质量阈值
            "data_quality_score": 80.0,  # 数据质量总分阈值
            "completeness_score": 95.0,  # 完整性分数阈值
            "accuracy_score": 90.0,  # 准确性分数阈值
            "timeliness_score": 85.0,  # 时效性分数阈值
            "consistency_score": 90.0,  # 一致性分数阈值
            "uniqueness_score": 95.0,  # 唯一性分数阈值
        }
    )

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_rotation: str = "1 day"
    log_retention: str = "30 days"

    @classmethod
    def from_config_file(cls, config_path: Optional[str] = None) -> "MonitoringConfig":
        """从配置文件加载配置"""
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yml"

        if not os.path.exists(config_path):
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return cls()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # 提取监控相关配置
            monitoring_config = config_data.get("monitoring", {})

            return cls(
                system_check_interval=monitoring_config.get("system_check_interval", 30),
                health_check_timeout=monitoring_config.get("health_check_timeout", 10),
                db_connection_timeout=monitoring_config.get("db_connection_timeout", 5),
                db_query_timeout=monitoring_config.get("db_query_timeout", 30),
                redis_connection_timeout=monitoring_config.get("redis_connection_timeout", 5),
                redis_command_timeout=monitoring_config.get("redis_command_timeout", 10),
                api_timeout=monitoring_config.get("api_timeout", 30),
                api_check_endpoints=monitoring_config.get(
                    "api_check_endpoints", ["/api/v1/health", "/api/v1/stocks/basic", "/api/v1/factors/list"]
                ),
                celery_broker_timeout=monitoring_config.get("celery_broker_timeout", 10),
                alert_check_interval=monitoring_config.get("alert_check_interval", 60),
                alert_batch_size=monitoring_config.get("alert_batch_size", 100),
                alert_retention_days=monitoring_config.get("alert_retention_days", 30),
                performance_collection_interval=monitoring_config.get("performance_collection_interval", 60),
                performance_retention_days=monitoring_config.get("performance_retention_days", 7),
                websocket_heartbeat_interval=monitoring_config.get("websocket_heartbeat_interval", 30),
                websocket_connection_timeout=monitoring_config.get("websocket_connection_timeout", 300),
                notification_retry_count=monitoring_config.get("notification_retry_count", 3),
                notification_retry_delay=monitoring_config.get("notification_retry_delay", 60),
                notification_rate_limit=monitoring_config.get("notification_rate_limit", 10),
                data_quality_check_interval=monitoring_config.get("data_quality_check_interval", 300),
                data_quality_db_path=monitoring_config.get("data_quality_db_path", "data/monitoring/data_quality.db"),
                data_quality_retention_days=monitoring_config.get("data_quality_retention_days", 90),
                data_quality_batch_size=monitoring_config.get("data_quality_batch_size", 1000),
                thresholds=monitoring_config.get(
                    "thresholds",
                    {
                        "cpu_usage": 80.0,
                        "memory_usage": 85.0,
                        "disk_usage": 90.0,
                        "response_time": 5000.0,
                        "error_rate": 5.0,
                        "db_connections": 80,
                        "queue_size": 1000,
                        "data_quality_score": 80.0,
                        "completeness_score": 95.0,
                        "accuracy_score": 90.0,
                        "timeliness_score": 85.0,
                        "consistency_score": 90.0,
                        "uniqueness_score": 95.0,
                    },
                ),
                log_level=monitoring_config.get("log_level", "INFO"),
                log_file=monitoring_config.get("log_file"),
                log_rotation=monitoring_config.get("log_rotation", "1 day"),
                log_retention=monitoring_config.get("log_retention", "30 days"),
            )

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "system_check_interval": self.system_check_interval,
            "health_check_timeout": self.health_check_timeout,
            "db_connection_timeout": self.db_connection_timeout,
            "db_query_timeout": self.db_query_timeout,
            "redis_connection_timeout": self.redis_connection_timeout,
            "redis_command_timeout": self.redis_command_timeout,
            "api_timeout": self.api_timeout,
            "api_check_endpoints": self.api_check_endpoints,
            "celery_broker_timeout": self.celery_broker_timeout,
            "alert_check_interval": self.alert_check_interval,
            "alert_batch_size": self.alert_batch_size,
            "alert_retention_days": self.alert_retention_days,
            "performance_collection_interval": self.performance_collection_interval,
            "performance_retention_days": self.performance_retention_days,
            "websocket_heartbeat_interval": self.websocket_heartbeat_interval,
            "websocket_connection_timeout": self.websocket_connection_timeout,
            "notification_retry_count": self.notification_retry_count,
            "notification_retry_delay": self.notification_retry_delay,
            "notification_rate_limit": self.notification_rate_limit,
            "data_quality_check_interval": self.data_quality_check_interval,
            "data_quality_db_path": self.data_quality_db_path,
            "data_quality_retention_days": self.data_quality_retention_days,
            "data_quality_batch_size": self.data_quality_batch_size,
            "thresholds": self.thresholds,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "log_rotation": self.log_rotation,
            "log_retention": self.log_retention,
        }


# 全局配置实例
monitoring_config = MonitoringConfig.from_config_file()


def get_monitoring_config() -> MonitoringConfig:
    """获取监控配置"""
    return monitoring_config


def update_monitoring_config(new_config: Dict[str, Any]) -> None:
    """更新监控配置"""
    global monitoring_config

    # 更新配置
    for key, value in new_config.items():
        if hasattr(monitoring_config, key):
            setattr(monitoring_config, key, value)
        else:
            logger.warning(f"未知的配置项: {key}")

    logger.info("监控配置已更新")


if __name__ == "__main__":
    # 测试配置加载
    config = get_monitoring_config()
    print("监控配置:")
    print(yaml.dump(config.to_dict(), default_flow_style=False, allow_unicode=True))
