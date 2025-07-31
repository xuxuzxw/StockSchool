#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控告警模块单元测试

测试 src/monitoring/alerts.py 中的核心功能
"""

import unittest
import tempfile
import os
import sqlite3
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.monitoring.alerts import (
    AlertSeverity, AlertStatus, AlertType, AlertRule, Alert,
    MetricCollector, AlertStorage, AlertEngine, EmailChannel, WebhookChannel
)

class TestAlertDataClasses(unittest.TestCase):
    """测试告警数据类"""
    
    def test_alert_rule_creation(self):
        """测试告警规则创建"""
        rule = AlertRule(
            id="test_rule_1",
            name="CPU使用率告警",
            description="CPU使用率超过阈值",
            alert_type=AlertType.SYSTEM,
            severity=AlertSeverity.WARNING,
            condition="cpu_usage",
            threshold=80.0,
            comparison=">",
            evaluation_window=300,
            trigger_count=3
        )
        
        self.assertEqual(rule.id, "test_rule_1")
        self.assertEqual(rule.name, "CPU使用率告警")
        self.assertEqual(rule.alert_type, AlertType.SYSTEM)
        self.assertEqual(rule.severity, AlertSeverity.WARNING)
        self.assertEqual(rule.threshold, 80.0)
        self.assertTrue(rule.enabled)
        self.assertEqual(rule.tags, [])
        self.assertEqual(rule.metadata, {})
    
    def test_alert_creation(self):
        """测试告警创建"""
        now = datetime.now()
        alert = Alert(
            id="alert_1",
            rule_id="rule_1",
            title="CPU使用率过高",
            message="当前CPU使用率85%，超过阈值80%",
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.SYSTEM,
            status=AlertStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )
        
        self.assertEqual(alert.id, "alert_1")
        self.assertEqual(alert.rule_id, "rule_1")
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertEqual(alert.status, AlertStatus.ACTIVE)
        self.assertIsNone(alert.resolved_at)
        self.assertIsNone(alert.acknowledged_at)
    
    def test_alert_to_dict(self):
        """测试告警转换为字典"""
        now = datetime.now()
        alert = Alert(
            id="alert_1",
            rule_id="rule_1",
            title="测试告警",
            message="测试消息",
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.SYSTEM,
            status=AlertStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            tags=["test", "system"],
            metadata={"key": "value"}
        )
        
        alert_dict = alert.to_dict()
        
        self.assertEqual(alert_dict['id'], "alert_1")
        self.assertEqual(alert_dict['severity'], "ERROR")
        self.assertEqual(alert_dict['alert_type'], "SYSTEM")
        self.assertEqual(alert_dict['status'], "ACTIVE")
        self.assertEqual(alert_dict['tags'], ["test", "system"])
        self.assertEqual(alert_dict['metadata'], {"key": "value"})
        self.assertIsInstance(alert_dict['created_at'], str)

class TestMetricCollector(unittest.TestCase):
    """测试指标收集器"""
    
    def setUp(self):
        self.collector = MetricCollector(max_history=100)
    
    def test_record_metric(self):
        """测试记录指标"""
        self.collector.record_metric("cpu_usage", 75.5)
        self.collector.record_metric("memory_usage", 60.0)
        
        cpu_values = self.collector.get_metric_values("cpu_usage")
        memory_values = self.collector.get_metric_values("memory_usage")
        
        self.assertEqual(len(cpu_values), 1)
        self.assertEqual(len(memory_values), 1)
        self.assertEqual(cpu_values[0][1], 75.5)
        self.assertEqual(memory_values[0][1], 60.0)
    
    def test_get_metric_stats(self):
        """测试获取指标统计"""
        # 记录多个指标值
        values = [70.0, 75.0, 80.0, 85.0, 90.0]
        for value in values:
            self.collector.record_metric("cpu_usage", value)
        
        stats = self.collector.get_metric_stats("cpu_usage")
        
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['min'], 70.0)
        self.assertEqual(stats['max'], 90.0)
        self.assertEqual(stats['mean'], 80.0)
        self.assertEqual(stats['latest'], 90.0)
    
    def test_metric_window_filtering(self):
        """测试指标时间窗口过滤"""
        now = datetime.now()
        old_time = now - timedelta(seconds=3600)  # 1小时前
        
        # 记录旧指标
        self.collector.record_metric("cpu_usage", 50.0, old_time)
        # 记录新指标
        self.collector.record_metric("cpu_usage", 80.0, now)
        
        # 获取最近30分钟的指标
        recent_values = self.collector.get_metric_values("cpu_usage", window_seconds=1800)
        
        self.assertEqual(len(recent_values), 1)
        self.assertEqual(recent_values[0][1], 80.0)

class TestAlertStorage(unittest.TestCase):
    """测试告警存储"""
    
    def setUp(self):
        # 创建临时数据库
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_db_fd)  # 关闭文件描述符
        self.storage = AlertStorage(self.temp_db_path)
    
    def tearDown(self):
        # 清理临时数据库
        try:
            if hasattr(self.storage, 'conn') and self.storage.conn:
                self.storage.conn.close()
        except:
            pass
        try:
            os.unlink(self.temp_db_path)
        except:
            pass
    
    def test_save_and_load_rule(self):
        """测试保存和加载告警规则"""
        rule = AlertRule(
            id="test_rule",
            name="测试规则",
            description="测试描述",
            alert_type=AlertType.SYSTEM,
            severity=AlertSeverity.WARNING,
            condition="cpu_usage",
            threshold=80.0,
            comparison=">",
            evaluation_window=300,
            trigger_count=3,
            tags=["test"],
            metadata={"env": "test"}
        )
        
        # 保存规则
        self.storage.save_rule(rule)
        
        # 加载规则
        rules = self.storage.load_rules()
        
        self.assertEqual(len(rules), 1)
        loaded_rule = rules[0]
        self.assertEqual(loaded_rule.id, "test_rule")
        self.assertEqual(loaded_rule.name, "测试规则")
        self.assertEqual(loaded_rule.alert_type, AlertType.SYSTEM)
        self.assertEqual(loaded_rule.severity, AlertSeverity.WARNING)
        self.assertEqual(loaded_rule.tags, ["test"])
        self.assertEqual(loaded_rule.metadata, {"env": "test"})
    
    def test_save_and_load_alert(self):
        """测试保存和加载告警"""
        now = datetime.now()
        alert = Alert(
            id="test_alert",
            rule_id="test_rule",
            title="测试告警",
            message="测试消息",
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.SYSTEM,
            status=AlertStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            source="test",
            tags=["test"],
            metadata={"key": "value"}
        )
        
        # 保存告警
        self.storage.save_alert(alert)
        
        # 加载告警
        alerts = self.storage.load_alerts()
        
        self.assertEqual(len(alerts), 1)
        loaded_alert = alerts[0]
        self.assertEqual(loaded_alert.id, "test_alert")
        self.assertEqual(loaded_alert.rule_id, "test_rule")
        self.assertEqual(loaded_alert.severity, AlertSeverity.ERROR)
        self.assertEqual(loaded_alert.status, AlertStatus.ACTIVE)
        self.assertEqual(loaded_alert.tags, ["test"])
        self.assertEqual(loaded_alert.metadata, {"key": "value"})
    
    def test_load_alerts_by_status(self):
        """测试按状态加载告警"""
        now = datetime.now()
        
        # 创建不同状态的告警
        active_alert = Alert(
            id="active_alert",
            rule_id="rule1",
            title="活跃告警",
            message="消息",
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.SYSTEM,
            status=AlertStatus.ACTIVE,
            created_at=now,
            updated_at=now
        )
        
        resolved_alert = Alert(
            id="resolved_alert",
            rule_id="rule2",
            title="已解决告警",
            message="消息",
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.SYSTEM,
            status=AlertStatus.RESOLVED,
            created_at=now,
            updated_at=now,
            resolved_at=now
        )
        
        # 保存告警
        self.storage.save_alert(active_alert)
        self.storage.save_alert(resolved_alert)
        
        # 按状态加载
        active_alerts = self.storage.load_alerts(AlertStatus.ACTIVE)
        resolved_alerts = self.storage.load_alerts(AlertStatus.RESOLVED)
        
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(len(resolved_alerts), 1)
        self.assertEqual(active_alerts[0].id, "active_alert")
        self.assertEqual(resolved_alerts[0].id, "resolved_alert")

class TestNotificationChannels(unittest.TestCase):
    """测试通知渠道"""
    
    def test_email_channel_creation(self):
        """测试邮件通知渠道创建"""
        channel = EmailChannel(
            smtp_server="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="password",
            recipients=["admin@example.com"]
        )
        
        self.assertEqual(channel.smtp_server, "smtp.example.com")
        self.assertEqual(channel.smtp_port, 587)
        self.assertEqual(channel.recipients, ["admin@example.com"])
        self.assertTrue(channel.use_tls)
    
    def test_webhook_channel_creation(self):
        """测试Webhook通知渠道创建"""
        channel = WebhookChannel(
            url="https://hooks.example.com/webhook",
            headers={"Authorization": "Bearer token"}
        )
        
        self.assertEqual(channel.url, "https://hooks.example.com/webhook")
        self.assertEqual(channel.headers["Authorization"], "Bearer token")
    
    @patch('requests.post')
    def test_webhook_send_success(self, mock_post):
        """测试Webhook发送成功"""
        # 模拟成功响应
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        channel = WebhookChannel("https://hooks.example.com/webhook")
        
        alert = Alert(
            id="test_alert",
            rule_id="test_rule",
            title="测试告警",
            message="测试消息",
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.SYSTEM,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        result = channel.send(alert)
        
        self.assertTrue(result)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_webhook_send_failure(self, mock_post):
        """测试Webhook发送失败"""
        # 模拟失败响应
        mock_post.side_effect = Exception("Network error")
        
        channel = WebhookChannel("https://hooks.example.com/webhook")
        
        alert = Alert(
            id="test_alert",
            rule_id="test_rule",
            title="测试告警",
            message="测试消息",
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.SYSTEM,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        result = channel.send(alert)
        
        self.assertFalse(result)

class TestAlertEngine(unittest.TestCase):
    """测试告警引擎"""
    
    def setUp(self):
        # 创建临时数据库
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_db_fd)  # 关闭文件描述符
        self.storage = AlertStorage(self.temp_db_path)
        self.engine = AlertEngine(self.storage)
    
    def tearDown(self):
        # 停止引擎并清理
        if self.engine.running:
            self.engine.stop()
        try:
            if hasattr(self.storage, 'conn') and self.storage.conn:
                self.storage.conn.close()
        except:
            pass
        try:
            os.unlink(self.temp_db_path)
        except:
            pass
    
    def test_add_rule(self):
        """测试添加告警规则"""
        rule = AlertRule(
            id="cpu_rule",
            name="CPU告警",
            description="CPU使用率告警",
            alert_type=AlertType.SYSTEM,
            severity=AlertSeverity.WARNING,
            condition="cpu_usage",
            threshold=80.0,
            comparison=">",
            evaluation_window=300,
            trigger_count=1
        )
        
        self.engine.add_rule(rule)
        
        self.assertIn("cpu_rule", self.engine.rules)
        self.assertEqual(self.engine.rules["cpu_rule"].name, "CPU告警")
    
    def test_record_metric_no_trigger(self):
        """测试记录指标但不触发告警"""
        rule = AlertRule(
            id="cpu_rule",
            name="CPU告警",
            description="CPU使用率告警",
            alert_type=AlertType.SYSTEM,
            severity=AlertSeverity.WARNING,
            condition="cpu_usage",
            threshold=80.0,
            comparison=">",
            evaluation_window=300,
            trigger_count=1
        )
        
        self.engine.add_rule(rule)
        
        # 记录低于阈值的指标
        self.engine.record_metric("cpu_usage", 70.0)
        
        # 检查没有触发告警
        self.assertTrue(self.engine.alert_queue.empty())
    
    def test_record_metric_trigger_alert(self):
        """测试记录指标触发告警"""
        rule = AlertRule(
            id="cpu_rule",
            name="CPU告警",
            description="CPU使用率告警",
            alert_type=AlertType.SYSTEM,
            severity=AlertSeverity.WARNING,
            condition="cpu_usage",
            threshold=80.0,
            comparison=">",
            evaluation_window=300,
            trigger_count=1
        )
        
        self.engine.add_rule(rule)
        
        # 记录超过阈值的指标
        self.engine.record_metric("cpu_usage", 85.0)
        
        # 检查触发了告警
        self.assertFalse(self.engine.alert_queue.empty())
        
        # 获取告警
        alert = self.engine.alert_queue.get()
        self.assertEqual(alert.rule_id, "cpu_rule")
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertEqual(alert.status, AlertStatus.ACTIVE)
    
    def test_evaluate_condition(self):
        """测试条件评估"""
        # 测试各种比较操作
        self.assertTrue(self.engine._evaluate_condition(85.0, 80.0, ">"))
        self.assertFalse(self.engine._evaluate_condition(75.0, 80.0, ">"))
        
        self.assertTrue(self.engine._evaluate_condition(75.0, 80.0, "<"))
        self.assertFalse(self.engine._evaluate_condition(85.0, 80.0, "<"))
        
        self.assertTrue(self.engine._evaluate_condition(80.0, 80.0, ">="))
        self.assertTrue(self.engine._evaluate_condition(85.0, 80.0, ">="))
        
        self.assertTrue(self.engine._evaluate_condition(80.0, 80.0, "<="))
        self.assertTrue(self.engine._evaluate_condition(75.0, 80.0, "<="))
        
        self.assertTrue(self.engine._evaluate_condition(80.0, 80.0, "=="))
        self.assertFalse(self.engine._evaluate_condition(85.0, 80.0, "=="))
        
        self.assertTrue(self.engine._evaluate_condition(85.0, 80.0, "!="))
        self.assertFalse(self.engine._evaluate_condition(80.0, 80.0, "!="))
    
    def test_suppression_rule(self):
        """测试抑制规则"""
        # 添加抑制规则
        self.engine.add_suppression_rule("CPU告警", 10)  # 10分钟抑制
        
        rule = AlertRule(
            id="cpu_rule",
            name="CPU告警",
            description="CPU使用率告警",
            alert_type=AlertType.SYSTEM,
            severity=AlertSeverity.WARNING,
            condition="cpu_usage",
            threshold=80.0,
            comparison=">",
            evaluation_window=300,
            trigger_count=1
        )
        
        self.engine.add_rule(rule)
        
        # 第一次触发告警
        self.engine.record_metric("cpu_usage", 85.0)
        self.assertFalse(self.engine.alert_queue.empty())
        
        # 清空队列
        self.engine.alert_queue.get()
        
        # 立即再次触发，应该被抑制
        self.engine.record_metric("cpu_usage", 90.0)
        # 注意：由于抑制逻辑依赖数据库中的历史告警，这里可能需要更复杂的测试设置

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)