#!/usr/bin/env python3
"""
StockSchool 监控系统测试脚本

这个脚本用于测试监控系统的各个组件是否正常工作。
"""

import sys
import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.performance import PerformanceMonitor
from src.monitoring.alerts import AlertEngine, AlertRule, AlertSeverity
from src.monitoring.health import HealthChecker
from src.monitoring.notifications import NotificationManager, NotificationConfig
from src.monitoring.config import MonitoringConfig
from src.monitoring import MonitoringSystem

class TestPerformanceMonitor:
    """性能监控器测试"""
    
    @pytest.fixture
    def monitor(self):
        return PerformanceMonitor()
        
    def test_record_request(self, monitor):
        """测试记录请求"""
        # 记录请求
        monitor.record_request("/api/test", "GET", 200, 150.5)
        
        # 获取统计信息
        stats = monitor.get_stats()
        
        assert stats["total_requests"] == 1
        assert stats["avg_response_time"] == 150.5
        assert stats["error_rate"] == 0.0
        
    def test_record_error(self, monitor):
        """测试记录错误"""
        # 记录正常请求和错误请求
        monitor.record_request("/api/test", "GET", 200, 100)
        monitor.record_request("/api/test", "GET", 500, 200)
        
        stats = monitor.get_stats()
        
        assert stats["total_requests"] == 2
        assert stats["error_rate"] == 50.0  # 1 error out of 2 requests
        
    def test_get_endpoint_stats(self, monitor):
        """测试获取端点统计"""
        # 记录不同端点的请求
        monitor.record_request("/api/users", "GET", 200, 100)
        monitor.record_request("/api/orders", "POST", 201, 200)
        monitor.record_request("/api/users", "GET", 200, 150)
        
        endpoint_stats = monitor.get_endpoint_stats()
        
        assert len(endpoint_stats) == 2
        assert endpoint_stats["/api/users"]["count"] == 2
        assert endpoint_stats["/api/users"]["avg_response_time"] == 125.0
        assert endpoint_stats["/api/orders"]["count"] == 1

class TestAlertEngine:
    """告警引擎测试"""
    
    @pytest.fixture
    def alert_engine(self):
        return AlertEngine()
        
    def test_add_rule(self, alert_engine):
        """测试添加告警规则"""
        rule = AlertRule(
            name="test_rule",
            metric="cpu_usage",
            operator=">",
            threshold=80,
            severity=AlertSeverity.WARNING,
            duration=300
        )
        
        alert_engine.add_rule(rule)
        
        assert len(alert_engine.rules) == 1
        assert alert_engine.rules[0].name == "test_rule"
        
    def test_check_rules(self, alert_engine):
        """测试检查告警规则"""
        # 添加规则
        rule = AlertRule(
            name="high_cpu",
            metric="cpu_usage",
            operator=">",
            threshold=80,
            severity=AlertSeverity.WARNING,
            duration=0  # 立即触发
        )
        alert_engine.add_rule(rule)
        
        # 检查规则（CPU使用率超过阈值）
        metrics = {"cpu_usage": 85}
        alerts = alert_engine.check_rules(metrics)
        
        assert len(alerts) == 1
        assert alerts[0].rule_name == "high_cpu"
        assert alerts[0].severity == AlertSeverity.WARNING
        
    def test_check_rules_no_trigger(self, alert_engine):
        """测试不触发告警的情况"""
        rule = AlertRule(
            name="high_cpu",
            metric="cpu_usage",
            operator=">",
            threshold=80,
            severity=AlertSeverity.WARNING,
            duration=0
        )
        alert_engine.add_rule(rule)
        
        # CPU使用率未超过阈值
        metrics = {"cpu_usage": 75}
        alerts = alert_engine.check_rules(metrics)
        
        assert len(alerts) == 0
        
    def test_get_active_alerts(self, alert_engine):
        """测试获取活跃告警"""
        rule = AlertRule(
            name="test_rule",
            metric="memory_usage",
            operator=">",
            threshold=90,
            severity=AlertSeverity.CRITICAL,
            duration=0
        )
        alert_engine.add_rule(rule)
        
        # 触发告警
        metrics = {"memory_usage": 95}
        alert_engine.check_rules(metrics)
        
        active_alerts = alert_engine.get_active_alerts()
        assert len(active_alerts) > 0

class TestHealthChecker:
    """健康检查器测试"""
    
    @pytest.fixture
    def health_checker(self):
        return HealthChecker()
        
    @pytest.mark.asyncio
    async def test_check_system_health(self, health_checker):
        """测试系统健康检查"""
        health = await health_checker.check_system_health()
        
        assert "status" in health
        assert "cpu_usage" in health
        assert "memory_usage" in health
        assert "disk_usage" in health
        assert health["status"] in ["healthy", "warning", "error"]
        
    @pytest.mark.asyncio
    async def test_check_database_health(self, health_checker):
        """测试数据库健康检查"""
        # Mock数据库连接
        with patch('src.monitoring.health.get_database') as mock_db:
            mock_db.return_value = Mock()
            
            health = await health_checker.check_database_health()
            
            assert "status" in health
            assert "response_time" in health
            
    @pytest.mark.asyncio
    async def test_check_redis_health(self, health_checker):
        """测试Redis健康检查"""
        # Mock Redis连接
        with patch('redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_instance.ping.return_value = True
            mock_instance.info.return_value = {"used_memory": 1024000}
            mock_redis.return_value = mock_instance
            
            health = await health_checker.check_redis_health()
            
            assert "status" in health
            assert "response_time" in health
            assert "memory_usage" in health

class TestNotificationManager:
    """通知管理器测试"""
    
    @pytest.fixture
    def notification_config(self):
        return NotificationConfig(
            email_enabled=False,
            webhook_enabled=False,
            dingtalk_enabled=False
        )
        
    @pytest.fixture
    def notification_manager(self, notification_config):
        return NotificationManager(notification_config)
        
    @pytest.mark.asyncio
    async def test_send_notification_disabled(self, notification_manager):
        """测试发送通知（所有渠道禁用）"""
        result = await notification_manager.send_notification(
            "Test Alert",
            "This is a test alert",
            "warning"
        )
        
        # 所有渠道都禁用，应该返回空结果
        assert len(result) == 0
        
    def test_get_statistics(self, notification_manager):
        """测试获取统计信息"""
        stats = notification_manager.get_statistics()
        
        assert "total_sent" in stats
        assert "total_failed" in stats
        assert "channels" in stats

class TestMonitoringConfig:
    """监控配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MonitoringConfig()
        
        assert config.system_check_interval == 30
        assert config.health_check_timeout == 10
        assert config.database_enabled is True
        assert config.redis_enabled is True
        
    def test_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {
            "system_check_interval": 60,
            "database_enabled": False,
            "redis_check_interval": 45
        }
        
        config = MonitoringConfig.from_dict(config_dict)
        
        assert config.system_check_interval == 60
        assert config.database_enabled is False
        assert config.redis_check_interval == 45

class TestMonitoringSystem:
    """监控系统集成测试"""
    
    @pytest.fixture
    def monitoring_config(self):
        return MonitoringConfig(
            system_check_interval=1,  # 快速测试
            database_enabled=False,  # 避免真实数据库连接
            redis_enabled=False,     # 避免真实Redis连接
            alerting_enabled=True
        )
        
    @pytest.fixture
    def monitoring_system(self, monitoring_config):
        return MonitoringSystem(monitoring_config)
        
    @pytest.mark.asyncio
    async def test_start_stop(self, monitoring_system):
        """测试启动和停止监控系统"""
        # 启动系统
        await monitoring_system.start()
        assert monitoring_system.running is True
        
        # 停止系统
        await monitoring_system.stop()
        assert monitoring_system.running is False
        
    @pytest.mark.asyncio
    async def test_get_health_status(self, monitoring_system):
        """测试获取健康状态"""
        await monitoring_system.start()
        
        try:
            health = await monitoring_system.get_health_status()
            
            assert "system" in health
            assert "overall_status" in health
            
        finally:
            await monitoring_system.stop()
            
    def test_get_performance_metrics(self, monitoring_system):
        """测试获取性能指标"""
        metrics = monitoring_system.get_performance_metrics()
        
        assert "total_requests" in metrics
        assert "avg_response_time" in metrics
        assert "error_rate" in metrics
        
    def test_get_alerts(self, monitoring_system):
        """测试获取告警"""
        alerts = monitoring_system.get_alerts()
        
        assert isinstance(alerts, list)

def run_tests():
    """运行所有测试"""
    print("开始运行监控系统测试...")
    
    # 运行pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])
    
    if exit_code == 0:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败！")
        
    return exit_code

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)