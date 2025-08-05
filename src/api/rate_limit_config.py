from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.config.unified_config import config

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API限流配置管理
支持动态配置更新和角色限流策略管理
"""


class RateLimitRule(BaseModel):
    """限流规则配置"""

    requests_per_minute: int = Field(default=60, description="每分钟请求数限制")
    requests_per_hour: int = Field(default=1000, description="每小时请求数限制")
    burst_limit: int = Field(default=10, description="突发请求限制")
    enabled: bool = Field(default=True, description="是否启用限流")
    whitelist: bool = Field(default=False, description="是否为白名单（不受限制）")


class RateLimitConfig(BaseModel):
    """API限流配置"""

    # 角色限流配置
    role_limits: Dict[str, RateLimitRule] = Field(
        default_factory=lambda: {
            "guest": RateLimitRule(requests_per_minute=10, requests_per_hour=100, burst_limit=5, enabled=True),
            "user": RateLimitRule(requests_per_minute=60, requests_per_hour=1000, burst_limit=10, enabled=True),
            "developer": RateLimitRule(requests_per_minute=120, requests_per_hour=2000, burst_limit=20, enabled=True),
            "enterprise": RateLimitRule(requests_per_minute=300, requests_per_hour=5000, burst_limit=50, enabled=True),
            "admin": RateLimitRule(requests_per_minute=1000, requests_per_hour=10000, burst_limit=100, enabled=True),
        }
    )

    # IP级限流配置
    ip_limit: RateLimitRule = Field(
        default_factory=lambda: RateLimitRule(
            requests_per_minute=30, requests_per_hour=500, burst_limit=15, enabled=True
        )
    )

    # 全局限流配置
    global_limit: RateLimitRule = Field(
        default_factory=lambda: RateLimitRule(
            requests_per_minute=1000, requests_per_hour=10000, burst_limit=200, enabled=True
        )
    )

    # 重试配置
    retry_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 300.0,
            "exponential_base": 2.0,
            "jitter": True,
        }
    )

    # 白名单配置
    whitelist_ips: list = Field(default_factory=lambda: ["127.0.0.1", "::1"])

    whitelist_users: list = Field(default_factory=lambda: ["admin", "system"])

    # 监控配置
    monitoring: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "log_level": "INFO",
            "alert_threshold": {
                "rate_limit_exceeded": 10,  # 10秒内超过10次限流触发告警
                "retry_attempts": 5,  # 单个用户5次重试后触发告警
            },
            "metrics_retention_days": 7,
        }
    )

    @classmethod
    def from_config(cls) -> "RateLimitConfig":
        """从统一配置加载限流配置"""
        try:
            rate_limit_config = config.get("api", {}).get("rate_limit", {})
            return cls(**rate_limit_config)
        except Exception as e:
            # 如果配置加载失败，使用默认值
            return cls()

    def get_role_limit(self, role: str) -> Optional[RateLimitRule]:
        """获取指定角色的限流配置"""
        return self.role_limits.get(role)

    def update_role_limit(self, role: str, rule: RateLimitRule):
        """更新角色限流配置"""
        self.role_limits[role] = rule

    def is_whitelisted(self, identifier: str, identifier_type: str = "ip") -> bool:
        """检查是否在白名单中"""
        if identifier_type == "ip":
            return identifier in self.whitelist_ips
        elif identifier_type == "user":
            return identifier in self.whitelist_users
        return False

    def get_retry_config(self) -> Dict[str, Any]:
        """获取重试配置"""
        return self.retry_config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "role_limits": {role: rule.dict() for role, rule in self.role_limits.items()},
            "ip_limit": self.ip_limit.dict(),
            "global_limit": self.global_limit.dict(),
            "retry_config": self.retry_config,
            "whitelist_ips": self.whitelist_ips,
            "whitelist_users": self.whitelist_users,
            "monitoring": self.monitoring,
        }


# 全局配置实例
rate_limit_config = RateLimitConfig.from_config()


class RateLimitManager:
    """限流配置管理器"""

    def __init__(self):
        """方法描述"""
        self._config_listeners = []

    def update_config(self, new_config: Dict[str, Any]):
        """更新限流配置"""
        try:
            self.config = RateLimitConfig(**new_config)
            self._notify_config_change()
            logger.info("限流配置已更新")
        except Exception as e:
            logger.error(f"更新限流配置失败: {e}")
            raise

    def add_config_listener(self, callback: Callable):
        """添加配置变更监听器"""
        self._config_listeners.append(callback)

    def _notify_config_change(self):
        """通知配置变更"""
        for callback in self._config_listeners:
            try:
                callback(self.config)
            except Exception as e:
                logger.error(f"通知配置变更失败: {e}")

    def get_config(self) -> RateLimitConfig:
        """获取当前配置"""
        return self.config

    def reload_config(self):
        """重新加载配置"""
        self.config = RateLimitConfig.from_config()
        self._notify_config_change()


# 全局配置管理器
rate_limit_manager = RateLimitManager()
