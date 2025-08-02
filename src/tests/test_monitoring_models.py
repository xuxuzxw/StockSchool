"""
监控系统数据模型单元测试

测试监控相关数据库模型的正确性和约束条件
包含模型创建、数据验证、关系约束等测试

作者: StockSchool Team
创建时间: 2025-01-02
"""

import pytest
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from src.models.monitoring import (
    Base, MonitoringMetric, SystemHealthStatus, 
    AlertRecord, MonitoringConfig
)


class TestMonitoringModels:
    """监控模型测试类"""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """创建测试数据库引擎"""
        # 使用内存SQLite数据库进行测试
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def session(self, engine):
        """创建数据库会话"""
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.rollback()
        session.close()
    
    def test_monitoring_metric_creation(self, session):
        """测试监控指标模型创建"""
        # 创建监控指标
        metric = MonitoringMetric(
            metric_name="cpu_usage",
            metric_type="gauge",
            metric_value=Decimal("75.5"),
            metric_unit="percent",
            labels={"host": "server1", "region": "us-east-1"},
            source_component="system_monitor"
        )
        
        session.add(metric)
        session.commit()
        
        # 验证数据
        saved_metric = session.query(MonitoringMetric).filter_by(
            metric_name="cpu_usage"
        ).first()
        
        assert saved_metric is not None
        assert saved_metric.metric_name == "cpu_usage"
        assert saved_metric.metric_type == "gauge"
        assert saved_metric.metric_value == Decimal("75.5")
        assert saved_metric.metric_unit == "percent"
        assert saved_metric.labels["host"] == "server1"
        assert saved_metric.source_component == "system_monitor"
        assert saved_metric.timestamp is not None
        assert isinstance(saved_metric.id, uuid.UUID)
    
    def test_monitoring_metric_to_dict(self, session):
        """测试监控指标转换为字典"""
        metric = MonitoringMetric(
            metric_name="memory_usage",
            metric_type="gauge",
            metric_value=Decimal("1024.0"),
            metric_unit="MB",
            labels={"process": "python"},
            source_component="memory_monitor"
        )
        
        session.add(metric)
        session.commit()
        
        metric_dict = metric.to_dict()
        
        assert isinstance(metric_dict, dict)
        assert metric_dict["metric_name"] == "memory_usage"
        assert metric_dict["metric_type"] == "gauge"
        assert metric_dict["metric_value"] == 1024.0
        assert metric_dict["metric_unit"] == "MB"
        assert metric_dict["labels"]["process"] == "python"
        assert metric_dict["source_component"] == "memory_monitor"
        assert "timestamp" in metric_dict
        assert "id" in metric_dict
    
    def test_monitoring_metric_required_fields(self, session):
        """测试监控指标必填字段约束"""
        # 测试缺少metric_name
        with pytest.raises(IntegrityError):
            metric = MonitoringMetric(
                metric_type="gauge",
                source_component="test"
            )
            session.add(metric)
            session.commit()
        
        session.rollback()
        
        # 测试缺少metric_type
        with pytest.raises(IntegrityError):
            metric = MonitoringMetric(
                metric_name="test_metric",
                source_component="test"
            )
            session.add(metric)
            session.commit()
        
        session.rollback()
        
        # 测试缺少source_component
        with pytest.raises(IntegrityError):
            metric = MonitoringMetric(
                metric_name="test_metric",
                metric_type="gauge"
            )
            session.add(metric)
            session.commit()
    
    def test_system_health_status_creation(self, session):
        """测试系统健康状态模型创建"""
        health_status = SystemHealthStatus(
            component_name="database",
            status="healthy",
            details={
                "connection_count": 10,
                "response_time_ms": 50,
                "last_error": None
            },
            last_check=datetime.now()
        )
        
        session.add(health_status)
        session.commit()
        
        # 验证数据
        saved_status = session.query(SystemHealthStatus).filter_by(
            component_name="database"
        ).first()
        
        assert saved_status is not None
        assert saved_status.component_name == "database"
        assert saved_status.status == "healthy"
        assert saved_status.details["connection_count"] == 10
        assert saved_status.details["response_time_ms"] == 50
        assert saved_status.last_check is not None
        assert saved_status.timestamp is not None
    
    def test_system_health_status_to_dict(self, session):
        """测试系统健康状态转换为字典"""
        health_status = SystemHealthStatus(
            component_name="redis",
            status="warning",
            details={"memory_usage": "85%"},
            last_check=datetime.now()
        )
        
        session.add(health_status)
        session.commit()
        
        status_dict = health_status.to_dict()
        
        assert isinstance(status_dict, dict)
        assert status_dict["component_name"] == "redis"
        assert status_dict["status"] == "warning"
        assert status_dict["details"]["memory_usage"] == "85%"
        assert "timestamp" in status_dict
        assert "last_check" in status_dict
        assert "id" in status_dict
    
    def test_alert_record_creation(self, session):
        """测试告警记录模型创建"""
        alert = AlertRecord(
            alert_id="ALERT_001",
            alert_level="warning",
            alert_type="system_health",
            title="高CPU使用率告警",
            description="服务器CPU使用率超过80%",
            source_component="cpu_monitor",
            metric_name="cpu_usage",
            threshold_value=Decimal("80.0"),
            actual_value=Decimal("85.5")
        )
        
        session.add(alert)
        session.commit()
        
        # 验证数据
        saved_alert = session.query(AlertRecord).filter_by(
            alert_id="ALERT_001"
        ).first()
        
        assert saved_alert is not None
        assert saved_alert.alert_id == "ALERT_001"
        assert saved_alert.alert_level == "warning"
        assert saved_alert.alert_type == "system_health"
        assert saved_alert.title == "高CPU使用率告警"
        assert saved_alert.source_component == "cpu_monitor"
        assert saved_alert.threshold_value == Decimal("80.0")
        assert saved_alert.actual_value == Decimal("85.5")
        assert saved_alert.status == "active"  # 默认状态
        assert saved_alert.created_at is not None
    
    def test_alert_record_acknowledge(self, session):
        """测试告警确认功能"""
        alert = AlertRecord(
            alert_id="ALERT_002",
            alert_level="error",
            alert_type="database",
            title="数据库连接失败",
            description="无法连接到主数据库"
        )
        
        session.add(alert)
        session.commit()
        
        # 确认告警
        alert.acknowledge("admin_user")
        session.commit()
        
        # 验证确认状态
        assert alert.status == "acknowledged"
        assert alert.acknowledged_by == "admin_user"
        assert alert.acknowledged_at is not None
    
    def test_alert_record_resolve(self, session):
        """测试告警解决功能"""
        alert = AlertRecord(
            alert_id="ALERT_003",
            alert_level="critical",
            alert_type="system",
            title="系统宕机",
            description="系统无响应"
        )
        
        session.add(alert)
        session.commit()
        
        # 解决告警
        alert.resolve()
        session.commit()
        
        # 验证解决状态
        assert alert.status == "resolved"
        assert alert.resolved_at is not None
    
    def test_alert_record_unique_constraint(self, session):
        """测试告警记录唯一约束"""
        # 创建第一个告警
        alert1 = AlertRecord(
            alert_id="UNIQUE_ALERT",
            alert_level="info",
            alert_type="test",
            title="测试告警1"
        )
        session.add(alert1)
        session.commit()
        
        # 尝试创建相同alert_id的告警
        with pytest.raises(IntegrityError):
            alert2 = AlertRecord(
                alert_id="UNIQUE_ALERT",
                alert_level="warning",
                alert_type="test",
                title="测试告警2"
            )
            session.add(alert2)
            session.commit()
    
    def test_monitoring_config_creation(self, session):
        """测试监控配置模型创建"""
        config = MonitoringConfig(
            config_key="alert_thresholds",
            config_value={
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0
            },
            description="系统告警阈值配置"
        )
        
        session.add(config)
        session.commit()
        
        # 验证数据
        saved_config = session.query(MonitoringConfig).filter_by(
            config_key="alert_thresholds"
        ).first()
        
        assert saved_config is not None
        assert saved_config.config_key == "alert_thresholds"
        assert saved_config.config_value["cpu_usage"] == 80.0
        assert saved_config.description == "系统告警阈值配置"
        assert saved_config.created_at is not None
        assert saved_config.updated_at is not None
    
    def test_monitoring_config_unique_constraint(self, session):
        """测试监控配置唯一约束"""
        # 创建第一个配置
        config1 = MonitoringConfig(
            config_key="test_config",
            config_value={"value": 1},
            description="测试配置1"
        )
        session.add(config1)
        session.commit()
        
        # 尝试创建相同config_key的配置
        with pytest.raises(IntegrityError):
            config2 = MonitoringConfig(
                config_key="test_config",
                config_value={"value": 2},
                description="测试配置2"
            )
            session.add(config2)
            session.commit()
    
    def test_monitoring_config_to_dict(self, session):
        """测试监控配置转换为字典"""
        config = MonitoringConfig(
            config_key="notification_settings",
            config_value={
                "enabled": True,
                "channels": ["email", "sms"]
            },
            description="通知设置"
        )
        
        session.add(config)
        session.commit()
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["config_key"] == "notification_settings"
        assert config_dict["config_value"]["enabled"] is True
        assert config_dict["config_value"]["channels"] == ["email", "sms"]
        assert config_dict["description"] == "通知设置"
        assert "created_at" in config_dict
        assert "updated_at" in config_dict
        assert "id" in config_dict
    
    def test_model_relationships_and_queries(self, session):
        """测试模型关系和复杂查询"""
        # 创建相关数据
        metric = MonitoringMetric(
            metric_name="response_time",
            metric_type="gauge",
            metric_value=Decimal("2.5"),
            metric_unit="seconds",
            source_component="api_monitor"
        )
        
        health_status = SystemHealthStatus(
            component_name="api_service",
            status="warning",
            details={"avg_response_time": "2.5s"}
        )
        
        alert = AlertRecord(
            alert_id="API_SLOW_001",
            alert_level="warning",
            alert_type="performance",
            title="API响应时间过慢",
            source_component="api_monitor",
            metric_name="response_time",
            threshold_value=Decimal("2.0"),
            actual_value=Decimal("2.5")
        )
        
        session.add_all([metric, health_status, alert])
        session.commit()
        
        # 测试复杂查询
        # 查询特定组件的所有告警
        api_alerts = session.query(AlertRecord).filter_by(
            source_component="api_monitor"
        ).all()
        assert len(api_alerts) == 1
        assert api_alerts[0].alert_id == "API_SLOW_001"
        
        # 查询特定状态的系统健康记录
        warning_statuses = session.query(SystemHealthStatus).filter_by(
            status="warning"
        ).all()
        assert len(warning_statuses) == 1
        assert warning_statuses[0].component_name == "api_service"
        
        # 查询特定指标类型的监控数据
        gauge_metrics = session.query(MonitoringMetric).filter_by(
            metric_type="gauge"
        ).all()
        assert len(gauge_metrics) == 1
        assert gauge_metrics[0].metric_name == "response_time"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])