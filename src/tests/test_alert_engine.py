from datetime import datetime, timedelta

from src.monitoring.alert_engine import (
    AlertEngine,
    AlertEvent,
    AlertLevel,
    AlertRule,
    AlertStatus,
    AlertSuppressionManager,
    AsyncMock,
    EmailNotificationChannel,
    LogNotificationChannel,
    Mock,
    WebhookNotificationChannel,
    asyncio,
    create_alert_engine,
    create_default_notification_config,
    create_default_rules,
    patch,
    pytest,
    unittest.mock,
)


class TestAlertRule:
    """测试告警规则"""

    def test_alert_rule_creation(self):
        """测试告警规则创建"""
        rule = AlertRule(
            rule_id="test_rule",
            name="测试规则",
            description="这是一个测试规则",
            metric_name="cpu_usage",
            threshold=80.0,
            condition=">",
            severity=AlertLevel.WARNING
        )

        assert rule.rule_id == "test_rule"
        assert rule.name == "测试规则"
        assert rule.threshold == 80.0
        assert rule.condition == ">"
        assert rule.severity == AlertLevel.WARNING
        assert rule.enabled is True
        assert "log" in rule.notification_channels

    def test_alert_rule_evaluation(self):
        """测试告警规则评估"""
        rule = AlertRule(
            rule_id="test_rule",
            name="测试规则",
            description="测试规则",
            metric_name="cpu_usage",
            threshold=80.0,
            condition=">"
        )

        # 测试不同条件
        assert rule.evaluate(90.0) is True   # 90 > 80
        assert rule.evaluate(70.0) is False  # 70 > 80

        # 测试其他操作符
        rule.condition = "<"
        assert rule.evaluate(70.0) is True   # 70 < 80
        assert rule.evaluate(90.0) is False  # 90 < 80

        rule.condition = ">="
        assert rule.evaluate(80.0) is True   # 80 >= 80
        assert rule.evaluate(79.0) is False  # 79 >= 80

        rule.condition = "<="
        assert rule.evaluate(80.0) is True   # 80 <= 80
        assert rule.evaluate(81.0) is False  # 81 <= 80

        rule.condition = "=="
        assert rule.evaluate(80.0) is True   # 80 == 80
        assert rule.evaluate(81.0) is False  # 81 == 80

        rule.condition = "!="
        assert rule.evaluate(81.0) is True   # 81 != 80
        assert rule.evaluate(80.0) is False  # 80 != 80

    def test_alert_rule_to_dict(self):
        """测试告警规则转换为字典"""
        rule = AlertRule(
            rule_id="test_rule",
            name="测试规则",
            description="测试规则",
            metric_name="cpu_usage",
            threshold=80.0
        )

        rule_dict = rule.to_dict()

        assert isinstance(rule_dict, dict)
        assert rule_dict["rule_id"] == "test_rule"
        assert rule_dict["name"] == "测试规则"
        assert rule_dict["threshold"] == 80.0


class TestAlertEvent:
    """测试告警事件"""

    def test_alert_event_creation(self):
        """测试告警事件创建"""
        event = AlertEvent(
            alert_id="ALERT_001",
            rule_id="test_rule",
            metric_name="cpu_usage",
            source_component="server1",
            severity=AlertLevel.WARNING,
            title="CPU使用率过高",
            description="CPU使用率超过阈值",
            threshold_value=80.0,
            actual_value=90.0,
            timestamp=datetime.now()
        )

        assert event.alert_id == "ALERT_001"
        assert event.severity == AlertLevel.WARNING
        assert event.threshold_value == 80.0
        assert event.actual_value == 90.0
        assert event.status == AlertStatus.ACTIVE

    def test_alert_event_to_dict(self):
        """测试告警事件转换为字典"""
        event = AlertEvent(
            alert_id="ALERT_001",
            rule_id="test_rule",
            metric_name="cpu_usage",
            source_component="server1",
            severity=AlertLevel.WARNING,
            title="CPU使用率过高",
            description="CPU使用率超过阈值",
            threshold_value=80.0,
            actual_value=90.0,
            timestamp=datetime.now()
        )

        event_dict = event.to_dict()

        assert isinstance(event_dict, dict)
        assert event_dict["alert_id"] == "ALERT_001"
        assert "timestamp" in event_dict
        assert isinstance(event_dict["timestamp"], str)


class TestNotificationChannels:
    """测试通知渠道"""

    @pytest.mark.asyncio
    async def test_log_notification_channel(self):
        """测试日志通知渠道"""
        channel = LogNotificationChannel({})

        assert channel.is_available() is True

        # 创建测试告警
        alert = AlertEvent(
            alert_id="TEST_001",
            rule_id="test_rule",
            metric_name="test_metric",
            source_component="test_component",
            severity=AlertLevel.WARNING,
            title="测试告警",
            description="测试描述",
            threshold_value=100.0,
            actual_value=150.0,
            timestamp=datetime.now()
        )

        # 测试发送通知
        result = await channel.send_notification(alert)
        assert result is True

    def test_email_notification_channel_availability(self):
        """测试邮件通知渠道可用性"""
        # 不完整配置
        incomplete_config = {'smtp_host': 'smtp.gmail.com'}
        channel = EmailNotificationChannel(incomplete_config)
        assert channel.is_available() is False

        # 完整配置
        complete_config = {
            'smtp_host': 'smtp.gmail.com',
            'from_email': 'test@example.com',
            'to_emails': ['admin@example.com']
        }
        channel = EmailNotificationChannel(complete_config)
        assert channel.is_available() is True

    def test_webhook_notification_channel_availability(self):
        """测试Webhook通知渠道可用性"""
        # 不完整配置
        incomplete_config = {}
        channel = WebhookNotificationChannel(incomplete_config)
        assert channel.is_available() is False

        # 完整配置
        complete_config = {'webhook_url': 'https://example.com/webhook'}
        channel = WebhookNotificationChannel(complete_config)
        assert channel.is_available() is True


class TestAlertSuppressionManager:
    """测试告警抑制管理器"""

    def test_suppression_manager_creation(self):
        """测试抑制管理器创建"""
        manager = AlertSuppressionManager()

        assert isinstance(manager.suppression_rules, dict)
        assert isinstance(manager.alert_history, dict)
        assert isinstance(manager.last_alert_time, dict)

    def test_cooldown_suppression(self):
        """测试冷却时间抑制"""
        manager = AlertSuppressionManager()

        # 创建测试告警和规则
        alert = AlertEvent(
            alert_id="TEST_001",
            rule_id="test_rule",
            metric_name="cpu_usage",
            source_component="server1",
            severity=AlertLevel.WARNING,
            title="测试告警",
            description="测试描述",
            threshold_value=80.0,
            actual_value=90.0,
            timestamp=datetime.now()
        )

        rule = AlertRule(
            rule_id="test_rule",
            name="测试规则",
            description="测试规则",
            metric_name="cpu_usage",
            cooldown_minutes=10
        )

        # 第一次告警不应该被抑制
        assert manager.should_suppress_alert(alert, rule) is False

        # 记录告警
        manager.record_alert(alert)

        # 立即再次检查应该被抑制
        assert manager.should_suppress_alert(alert, rule) is True

    def test_frequency_limit_suppression(self):
        """测试频率限制抑制"""
        manager = AlertSuppressionManager()

        alert = AlertEvent(
            alert_id="TEST_001",
            rule_id="test_rule",
            metric_name="cpu_usage",
            source_component="server1",
            severity=AlertLevel.WARNING,
            title="测试告警",
            description="测试描述",
            threshold_value=80.0,
            actual_value=90.0,
            timestamp=datetime.now()
        )

        rule = AlertRule(
            rule_id="test_rule",
            name="测试规则",
            description="测试规则",
            metric_name="cpu_usage",
            cooldown_minutes=0,  # 禁用冷却时间
            max_alerts_per_hour=3
        )

        # 记录多个告警
        for _ in range(3):
            manager.record_alert(alert)

        # 第4个告警应该被抑制
        assert manager.should_suppress_alert(alert, rule) is True


class TestAlertEngine:
    """测试告警引擎"""

    @pytest.fixture
    def alert_engine(self):
        """创建告警引擎实例"""
        return AlertEngine()

    @pytest.mark.asyncio
    async def test_alert_engine_initialization(self, alert_engine):
        """测试告警引擎初始化"""
        config = {
            'rules': create_default_rules(),
            'notifications': {}
        }

        await alert_engine.initialize(config)

        assert alert_engine.running is True
        assert len(alert_engine.rules) > 0
        assert 'log' in alert_engine.notification_channels

        await alert_engine.close()
        assert alert_engine.running is False

    @pytest.mark.asyncio
    async def test_add_remove_rules(self, alert_engine):
        """测试添加和移除规则"""
        await alert_engine.initialize({'rules': [], 'notifications': {}})

        # 添加规则
        rule = AlertRule(
            rule_id="test_rule",
            name="测试规则",
            description="测试规则",
            metric_name="cpu_usage",
            threshold=80.0
        )

        await alert_engine.add_rule(rule)
        assert "test_rule" in alert_engine.rules
        assert alert_engine.stats['rules_loaded'] == 1

        # 移除规则
        await alert_engine.remove_rule("test_rule")
        assert "test_rule" not in alert_engine.rules
        assert alert_engine.stats['rules_loaded'] == 0

        await alert_engine.close()

    @pytest.mark.asyncio
    async def test_metric_evaluation(self, alert_engine):
        """测试指标评估"""
        # 创建规则
        rule = AlertRule(
            rule_id="cpu_high",
            name="CPU使用率过高",
            description="CPU使用率超过阈值",
            metric_name="cpu_usage",
            source_component="server1",
            threshold=80.0,
            condition=">"
        )

        config = {
            'rules': [rule.to_dict()],
            'notifications': {}
        }

        await alert_engine.initialize(config)

        # 测试触发告警的指标
        metric = {
            'metric_name': 'cpu_usage',
            'source_component': 'server1',
            'metric_value': 90.0,
            'timestamp': datetime.now().isoformat()
        }

        initial_alerts = alert_engine.stats['alerts_triggered']
        await alert_engine.evaluate_metric(metric)

        # 验证告警被触发
        assert alert_engine.stats['alerts_triggered'] > initial_alerts

        await alert_engine.close()

    @pytest.mark.asyncio
    async def test_notification_channel_testing(self, alert_engine):
        """测试通知渠道测试功能"""
        await alert_engine.initialize({'rules': [], 'notifications': {}})

        results = await alert_engine.test_notification_channels()

        assert isinstance(results, dict)
        assert 'log' in results
        assert results['log'] is True  # 日志通知应该总是成功

        await alert_engine.close()

    def test_get_stats(self, alert_engine):
        """测试获取统计信息"""
        stats = alert_engine.get_stats()

        required_fields = [
            'rules_loaded', 'alerts_triggered', 'alerts_suppressed',
            'notifications_sent', 'notifications_failed', 'running',
            'active_rules', 'total_rules', 'notification_channels'
        ]

        for field in required_fields:
            assert field in stats

    def test_get_rules(self, alert_engine):
        """测试获取规则列表"""
        rules = alert_engine.get_rules()
        assert isinstance(rules, list)


class TestConvenienceFunctions:
    """测试便捷函数"""

    @pytest.mark.asyncio
    async def test_create_alert_engine(self):
        """测试创建告警引擎便捷函数"""
        config = {
            'rules': create_default_rules(),
            'notifications': create_default_notification_config()
        }

        engine = await create_alert_engine(config)

        assert isinstance(engine, AlertEngine)
        assert engine.running is True
        assert len(engine.rules) > 0

        await engine.close()

    def test_create_default_rules(self):
        """测试创建默认规则"""
        rules = create_default_rules()

        assert isinstance(rules, list)
        assert len(rules) > 0

        for rule_config in rules:
            assert 'rule_id' in rule_config
            assert 'name' in rule_config
            assert 'metric_name' in rule_config
            assert 'threshold' in rule_config

    def test_create_default_notification_config(self):
        """测试创建默认通知配置"""
        config = create_default_notification_config()

        assert isinstance(config, dict)
        assert 'email' in config
        assert 'webhook' in config

        # 验证邮件配置结构
        email_config = config['email']
        required_email_fields = ['smtp_host', 'smtp_port', 'from_email', 'to_emails']
        for field in required_email_fields:
            assert field in email_config


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])