from datetime import datetime

from src.schemas.monitoring_schemas import (""",  # !/usr/bin/env python3
                                            __file__, import, os,
                                            os.path.abspath, os.path.dirname,
                                            sys, sys.path.append,
                                            简单测试监控数据模型的正确性)

    MetricType, AlertLevel, SystemStatus,
    MonitoringMetricSchema, SystemHealthMetrics,
    DatabaseHealthMetrics, RedisHealthMetrics,
    CeleryHealthMetrics, APIHealthMetrics,
    AlertRecordSchema, MonitoringConfigSchema
)

def test_monitoring_metric():
    """测试监控指标模型"""
    print("测试监控指标模型...")

    try:
        metric = MonitoringMetricSchema(
            timestamp=datetime.now(),
            metric_name="cpu_usage",
            metric_type=MetricType.GAUGE,
            metric_value=75.5,
            metric_unit="percent",
            labels={"host": "server1", "region": "us-east-1"},
            source_component="system_monitor"
        )

        print(f"✅ 监控指标创建成功: {metric.metric_name} = {metric.metric_value}{metric.metric_unit}")

        # 测试序列化
        metric_dict = metric.dict()
        print(f"✅ 字典序列化成功: {len(metric_dict)} 个字段")

        metric_json = metric.json()
        print(f"✅ JSON序列化成功: {len(metric_json)} 字符")

    except Exception as e:
        print(f"❌ 监控指标测试失败: {e}")
        return False

    return True

def test_system_health():
    """测试系统健康模型"""
    print("\n测试系统健康模型...")

    try:
        # 创建各组件健康指标
        db_metrics = DatabaseHealthMetrics(
            connection_status=SystemStatus.HEALTHY,
            connection_count=10,
            active_connections=5,
            query_avg_time_ms=50.0
        )

        redis_metrics = RedisHealthMetrics(
            connection_status=SystemStatus.WARNING,
            memory_usage_mb=512.0,
            memory_usage_percent=85.0,
            connected_clients=20,
            cache_hit_rate=95.5
        )

        celery_metrics = CeleryHealthMetrics(
            connection_status=SystemStatus.HEALTHY,
            active_tasks=5,
            pending_tasks=2,
            success_rate=98.0
        )

        api_metrics = APIHealthMetrics(
            status=SystemStatus.HEALTHY,
            response_time_ms=150.0,
            request_count_1h=1000,
            error_count_1h=10,
            error_rate=1.0
        )

        # 创建系统健康指标
        system_health = SystemHealthMetrics(
            database=db_metrics,
            redis=redis_metrics,
            celery=celery_metrics,
            api=api_metrics,
            overall_status=SystemStatus.HEALTHY
        )

        print(f"✅ 系统健康指标创建成功，整体状态: {system_health.overall_status}")
        print(f"  - 数据库状态: {system_health.database.connection_status}")
        print(f"  - Redis状态: {system_health.redis.connection_status}")
        print(f"  - Celery状态: {system_health.celery.connection_status}")
        print(f"  - API状态: {system_health.api.status}")

    except Exception as e:
        print(f"❌ 系统健康测试失败: {e}")
        return False

    return True

def test_alert_record():
    """测试告警记录模型"""
    print("\n测试告警记录模型...")

    try:
        alert = AlertRecordSchema(
            alert_id="ALERT_001",
            alert_level=AlertLevel.WARNING,
            alert_type="system_health",
            title="高CPU使用率告警",
            description="服务器CPU使用率超过80%",
            source_component="cpu_monitor",
            metric_name="cpu_usage",
            threshold_value=80.0,
            actual_value=85.5
        )

        print(f"✅ 告警记录创建成功: {alert.alert_id} - {alert.title}")
        print(f"  - 告警级别: {alert.alert_level}")
        print(f"  - 阈值: {alert.threshold_value}, 实际值: {alert.actual_value}")
        print(f"  - 状态: {alert.status}")

    except Exception as e:
        print(f"❌ 告警记录测试失败: {e}")
        return False

    return True

def test_monitoring_config():
    """测试监控配置模型"""
    print("\n测试监控配置模型...")

    try:
        config = MonitoringConfigSchema(
            config_key="alert_thresholds",
            config_value={
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0
            },
            description="系统告警阈值配置"
        )

        print(f"✅ 监控配置创建成功: {config.config_key}")
        print(f"  - 配置项数量: {len(config.config_value)}")
        print(f"  - 描述: {config.description}")

    except Exception as e:
        print(f"❌ 监控配置测试失败: {e}")
        return False

    return True

def test_validation_errors():
    """测试数据验证错误"""
    print("\n测试数据验证错误...")

    # 测试空指标名称
    try:
        MonitoringMetricSchema(
            timestamp=datetime.now(),
            metric_name="",  # 空名称
            metric_type=MetricType.GAUGE,
            source_component="test"
        )
        print("❌ 应该抛出验证错误但没有")
        return False
    except Exception:
        print("✅ 空指标名称验证错误正确抛出")

    # 测试活跃连接数超过总连接数
    try:
        DatabaseHealthMetrics(
            connection_status=SystemStatus.HEALTHY,
            connection_count=5,
            active_connections=10,  # 超过总连接数
            query_avg_time_ms=50.0
        )
        print("❌ 应该抛出验证错误但没有")
        return False
    except Exception:
        print("✅ 活跃连接数验证错误正确抛出")

    return True

def main():
    """主测试函数"""
    print("🚀 开始测试监控数据模型...")

    tests = [
        test_monitoring_metric,
        test_system_health,
        test_alert_record,
        test_monitoring_config,
        test_validation_errors
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1

    print(f"\n📊 测试结果: {passed}/{total} 个测试通过")

    if passed == total:
        print("🎉 所有测试通过！监控数据模型工作正常")
        return True
    else:
        print("❌ 部分测试失败，请检查模型定义")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)