"""
监控系统Pydantic模型单元测试

测试监控相关数据验证和序列化模型的正确性
包含数据验证、序列化、反序列化等测试

作者: StockSchool Team
创建时间: 2025-01-02
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from pydantic import ValidationError

from src.schemas.monitoring_schemas import (
    MetricType, AlertLevel, SystemStatus, AlertStatus,
    MonitoringMetricSchema, SystemHealthMetrics, DataSyncMetrics,
    FactorComputeMetrics, AIModelMetrics, AlertRecordSchema,
    MonitoringConfigSchema, DatabaseHealthMetrics, RedisHealthMetrics,
    CeleryHealthMetrics, APIHealthMetrics, DataSyncTaskInfo,
    DataQualityMetrics, APIQuotaInfo, FactorTaskInfo, PerformanceMetrics,
    ModelInfo, TrainingMetrics, PredictionResult, AlertQueryRequest,
    MetricQueryRequest, AlertAcknowledgeRequest
)


class TestEnums:
    """测试枚举类型"""
    
    def test_metric_type_enum(self):
        """测试指标类型枚举"""
        assert MetricType.GAUGE == "gauge"
        assert MetricType.COUNTER == "counter"
        assert MetricType.HISTOGRAM == "histogram"
    
    def test_alert_level_enum(self):
        """测试告警级别枚举"""
        assert AlertLevel.INFO == "info"
        assert AlertLevel.WARNING == "warning"
        assert AlertLevel.ERROR == "error"
        assert AlertLevel.CRITICAL == "critical"
    
    def test_system_status_enum(self):
        """测试系统状态枚举"""
        assert SystemStatus.HEALTHY == "healthy"
        assert SystemStatus.WARNING == "warning"
        assert SystemStatus.CRITICAL == "critical"


class TestMonitoringMetricSchema:
    """测试监控指标数据模型"""
    
    def test_valid_metric_creation(self):
        """测试有效指标创建"""
        metric = MonitoringMetricSchema(
            timestamp=datetime.now(),
            metric_name="cpu_usage",
            metric_type=MetricType.GAUGE,
            metric_value=75.5,
            metric_unit="percent",
            labels={"host": "server1", "region": "us-east-1"},
            source_component="system_monitor"
        )
        
        assert metric.metric_name == "cpu_usage"
        assert metric.metric_type == MetricType.GAUGE
        assert metric.metric_value == 75.5
        assert metric.metric_unit == "percent"
        assert metric.labels["host"] == "server1"
        assert metric.source_component == "system_monitor"
    
    def test_metric_validation_errors(self):
        """测试指标验证错误"""
        # 测试空指标名称
        with pytest.raises(ValidationError):
            MonitoringMetricSchema(
                timestamp=datetime.now(),
                metric_name="",
                metric_type=MetricType.GAUGE,
                source_component="test"
            )
        
        # 测试指标值超出范围
        with pytest.raises(ValidationError):
            MonitoringMetricSchema(
                timestamp=datetime.now(),
                metric_name="test_metric",
                metric_type=MetricType.GAUGE,
                metric_value=1e10,  # 超出范围
                source_component="test"
            )
        
        # 测试无效标签格式
        with pytest.raises(ValidationError):
            MonitoringMetricSchema(
                timestamp=datetime.now(),
                metric_name="test_metric",
                metric_type=MetricType.GAUGE,
                labels={"key": 123},  # 值不是字符串
                source_component="test"
            )
    
    def test_metric_serialization(self):
        """测试指标序列化"""
        metric = MonitoringMetricSchema(
            timestamp=datetime.now(),
            metric_name="memory_usage",
            metric_type=MetricType.GAUGE,
            metric_value=1024.0,
            metric_unit="MB",
            labels={"process": "python"},
            source_component="memory_monitor"
        )
        
        # 测试转换为字典
        metric_dict = metric.dict()
        assert isinstance(metric_dict, dict)
        assert metric_dict["metric_name"] == "memory_usage"
        assert metric_dict["metric_value"] == 1024.0
        
        # 测试JSON序列化
        metric_json = metric.json()
        assert isinstance(metric_json, str)
        assert "memory_usage" in metric_json


class TestSystemHealthMetrics:
    """测试系统健康指标模型"""
    
    def test_database_health_metrics(self):
        """测试数据库健康指标"""
        db_metrics = DatabaseHealthMetrics(
            connection_status=SystemStatus.HEALTHY,
            connection_count=10,
            active_connections=5,
            query_avg_time_ms=50.0,
            slow_queries_count=2
        )
        
        assert db_metrics.connection_status == SystemStatus.HEALTHY
        assert db_metrics.connection_count == 10
        assert db_metrics.active_connections == 5
        assert db_metrics.query_avg_time_ms == 50.0
    
    def test_database_health_validation(self):
        """测试数据库健康指标验证"""
        # 测试活跃连接数超过总连接数
        with pytest.raises(ValidationError):
            DatabaseHealthMetrics(
                connection_status=SystemStatus.HEALTHY,
                connection_count=5,
                active_connections=10,  # 超过总连接数
                query_avg_time_ms=50.0
            )
    
    def test_redis_health_metrics(self):
        """测试Redis健康指标"""
        redis_metrics = RedisHealthMetrics(
            connection_status=SystemStatus.HEALTHY,
            memory_usage_mb=512.0,
            memory_usage_percent=75.0,
            connected_clients=20,
            cache_hit_rate=95.5,
            keys_count=1000
        )
        
        assert redis_metrics.connection_status == SystemStatus.HEALTHY
        assert redis_metrics.memory_usage_mb == 512.0
        assert redis_metrics.cache_hit_rate == 95.5
    
    def test_system_health_overall_status(self):
        """测试系统整体健康状态计算"""
        # 创建各组件健康指标
        db_metrics = DatabaseHealthMetrics(
            connection_status=SystemStatus.HEALTHY,
            connection_count=10,
            active_connections=5,
            query_avg_time_ms=50.0
        )
        
        redis_metrics = RedisHealthMetrics(
            connection_status=SystemStatus.WARNING,  # 警告状态
            memory_usage_mb=512.0,
            memory_usage_percent=85.0,  # 高内存使用
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
            overall_status=SystemStatus.HEALTHY  # 会被自动计算覆盖
        )
        
        # 验证整体状态被正确计算为WARNING（因为Redis是WARNING状态）
        assert system_health.overall_status == SystemStatus.WARNING


class TestDataSyncMetrics:
    """测试数据同步指标模型"""
    
    def test_data_sync_task_info(self):
        """测试数据同步任务信息"""
        task_info = DataSyncTaskInfo(
            task_id="sync_001",
            task_name="股票基础数据同步",
            status="failed",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            error_message="API限流",
            records_processed=1000
        )
        
        assert task_info.task_id == "sync_001"
        assert task_info.status == "failed"
        assert task_info.records_processed == 1000
    
    def test_api_quota_info(self):
        """测试API配额信息"""
        quota_info = APIQuotaInfo(
            total_quota=10000,
            used_quota=8000,
            remaining_quota=2000,
            usage_percent=80.0,
            is_limited=False
        )
        
        assert quota_info.total_quota == 10000
        assert quota_info.used_quota == 8000
        assert quota_info.remaining_quota == 2000
        assert quota_info.usage_percent == 80.0
    
    def test_api_quota_validation(self):
        """测试API配额验证"""
        # 测试已使用配额超过总配额
        with pytest.raises(ValidationError):
            APIQuotaInfo(
                total_quota=10000,
                used_quota=12000,  # 超过总配额
                remaining_quota=0,
                usage_percent=120.0
            )
        
        # 测试剩余配额计算错误
        with pytest.raises(ValidationError):
            APIQuotaInfo(
                total_quota=10000,
                used_quota=8000,
                remaining_quota=3000,  # 应该是2000
                usage_percent=80.0
            )
    
    def test_data_quality_metrics(self):
        """测试数据质量指标"""
        quality_metrics = DataQualityMetrics(
            completeness_score=95.0,
            accuracy_score=98.0,
            consistency_score=92.0,
            timeliness_score=88.0,
            overall_score=93.25,
            anomaly_count=5,
            missing_data_count=10
        )
        
        assert quality_metrics.completeness_score == 95.0
        assert quality_metrics.overall_score == 93.25
        assert quality_metrics.anomaly_count == 5


class TestFactorComputeMetrics:
    """测试因子计算指标模型"""
    
    def test_factor_task_info(self):
        """测试因子计算任务信息"""
        task_info = FactorTaskInfo(
            task_id="factor_001",
            factor_name="RSI",
            status="running",
            progress=75.0,
            start_time=datetime.now() - timedelta(minutes=30),
            estimated_completion=datetime.now() + timedelta(minutes=10),
            stocks_processed=750,
            total_stocks=1000
        )
        
        assert task_info.factor_name == "RSI"
        assert task_info.progress == 75.0
        assert task_info.stocks_processed == 750
    
    def test_performance_metrics(self):
        """测试性能指标"""
        perf_metrics = PerformanceMetrics(
            cpu_usage_percent=85.0,
            memory_usage_mb=2048.0,
            memory_usage_percent=75.0,
            disk_io_read_mb=100.0,
            disk_io_write_mb=50.0,
            network_in_mb=25.0,
            network_out_mb=15.0
        )
        
        assert perf_metrics.cpu_usage_percent == 85.0
        assert perf_metrics.memory_usage_mb == 2048.0
        assert perf_metrics.memory_usage_percent == 75.0
    
    def test_factor_compute_metrics_validation(self):
        """测试因子计算指标验证"""
        perf_metrics = PerformanceMetrics(
            cpu_usage_percent=85.0,
            memory_usage_mb=2048.0,
            memory_usage_percent=75.0
        )
        
        # 测试任务列表过长
        long_task_list = [
            FactorTaskInfo(
                task_id=f"task_{i}",
                factor_name=f"factor_{i}",
                status="running",
                progress=50.0
            ) for i in range(60)  # 超过50个任务
        ]
        
        with pytest.raises(ValidationError):
            FactorComputeMetrics(
                current_tasks=long_task_list,
                completion_progress=50.0,
                performance_metrics=perf_metrics,
                success_rate=95.0,
                compute_status=SystemStatus.HEALTHY
            )


class TestAIModelMetrics:
    """测试AI模型指标模型"""
    
    def test_model_info(self):
        """测试模型信息"""
        model_info = ModelInfo(
            model_id="model_001",
            model_name="股价预测模型",
            version="v1.2.0",
            algorithm="LSTM",
            created_at=datetime.now() - timedelta(days=30),
            last_updated=datetime.now()
        )
        
        assert model_info.model_name == "股价预测模型"
        assert model_info.algorithm == "LSTM"
        assert model_info.version == "v1.2.0"
    
    def test_training_metrics(self):
        """测试训练指标"""
        training_metrics = TrainingMetrics(
            status="training",
            progress=75.0,
            current_epoch=75,
            total_epochs=100,
            loss=0.025,
            accuracy=92.5,
            validation_loss=0.030,
            validation_accuracy=90.0,
            learning_rate=0.001
        )
        
        assert training_metrics.progress == 75.0
        assert training_metrics.current_epoch == 75
        assert training_metrics.accuracy == 92.5
    
    def test_training_metrics_validation(self):
        """测试训练指标验证"""
        # 测试当前轮次超过总轮次
        with pytest.raises(ValidationError):
            TrainingMetrics(
                status="training",
                progress=75.0,
                current_epoch=150,  # 超过总轮次
                total_epochs=100,
                loss=0.025,
                accuracy=92.5
            )
    
    def test_prediction_result(self):
        """测试预测结果"""
        prediction = PredictionResult(
            prediction_id="pred_001",
            stock_code="000001.SZ",
            prediction_value=15.25,
            confidence=85.0,
            prediction_time=datetime.now(),
            actual_value=15.10,
            accuracy=98.0
        )
        
        assert prediction.stock_code == "000001.SZ"
        assert prediction.prediction_value == 15.25
        assert prediction.confidence == 85.0


class TestAlertModels:
    """测试告警相关模型"""
    
    def test_alert_record_schema(self):
        """测试告警记录模型"""
        alert = AlertRecordSchema(
            alert_id="ALERT_001",
            alert_level=AlertLevel.WARNING,
            alert_type="system_health",
            title="高CPU使用率告警",
            description="服务器CPU使用率超过80%",
            source_component="cpu_monitor",
            metric_name="cpu_usage",
            threshold_value=80.0,
            actual_value=85.5,
            status=AlertStatus.ACTIVE
        )
        
        assert alert.alert_id == "ALERT_001"
        assert alert.alert_level == AlertLevel.WARNING
        assert alert.title == "高CPU使用率告警"
        assert alert.threshold_value == 80.0
    
    def test_alert_time_validation(self):
        """测试告警时间验证"""
        now = datetime.now()
        
        # 测试确认时间早于创建时间
        with pytest.raises(ValidationError):
            AlertRecordSchema(
                alert_id="ALERT_002",
                alert_level=AlertLevel.ERROR,
                alert_type="database",
                title="数据库连接失败",
                created_at=now,
                acknowledged_at=now - timedelta(minutes=10)  # 早于创建时间
            )
        
        # 测试解决时间早于创建时间
        with pytest.raises(ValidationError):
            AlertRecordSchema(
                alert_id="ALERT_003",
                alert_level=AlertLevel.CRITICAL,
                alert_type="system",
                title="系统宕机",
                created_at=now,
                resolved_at=now - timedelta(minutes=5)  # 早于创建时间
            )
    
    def test_monitoring_config_schema(self):
        """测试监控配置模型"""
        config = MonitoringConfigSchema(
            config_key="alert_thresholds",
            config_value={
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0
            },
            description="系统告警阈值配置"
        )
        
        assert config.config_key == "alert_thresholds"
        assert config.config_value["cpu_usage"] == 80.0
        assert config.description == "系统告警阈值配置"
    
    def test_config_value_validation(self):
        """测试配置值验证"""
        # 测试非字典格式的配置值
        with pytest.raises(ValidationError):
            MonitoringConfigSchema(
                config_key="test_config",
                config_value="not_a_dict",  # 不是字典
                description="测试配置"
            )


class TestRequestModels:
    """测试请求模型"""
    
    def test_alert_query_request(self):
        """测试告警查询请求"""
        query_request = AlertQueryRequest(
            level=AlertLevel.WARNING,
            status=AlertStatus.ACTIVE,
            source_component="cpu_monitor",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            page=1,
            page_size=20
        )
        
        assert query_request.level == AlertLevel.WARNING
        assert query_request.status == AlertStatus.ACTIVE
        assert query_request.page == 1
        assert query_request.page_size == 20
    
    def test_alert_query_time_validation(self):
        """测试告警查询时间验证"""
        now = datetime.now()
        
        # 测试结束时间早于开始时间
        with pytest.raises(ValidationError):
            AlertQueryRequest(
                start_time=now,
                end_time=now - timedelta(hours=1),  # 早于开始时间
                page=1,
                page_size=20
            )
    
    def test_metric_query_request(self):
        """测试指标查询请求"""
        query_request = MetricQueryRequest(
            metric_names=["cpu_usage", "memory_usage"],
            source_components=["system_monitor"],
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            aggregation="avg",
            interval="5m"
        )
        
        assert len(query_request.metric_names) == 2
        assert "cpu_usage" in query_request.metric_names
        assert query_request.aggregation == "avg"
    
    def test_metric_query_validation(self):
        """测试指标查询验证"""
        now = datetime.now()
        
        # 测试查询时间范围超过30天
        with pytest.raises(ValidationError):
            MetricQueryRequest(
                metric_names=["cpu_usage"],
                start_time=now - timedelta(days=35),  # 超过30天
                end_time=now
            )
        
        # 测试指标数量超过限制
        with pytest.raises(ValidationError):
            MetricQueryRequest(
                metric_names=[f"metric_{i}" for i in range(25)],  # 超过20个
                start_time=now - timedelta(hours=1),
                end_time=now
            )
    
    def test_alert_acknowledge_request(self):
        """测试告警确认请求"""
        ack_request = AlertAcknowledgeRequest(
            acknowledged_by="admin_user",
            comment="已确认，正在处理"
        )
        
        assert ack_request.acknowledged_by == "admin_user"
        assert ack_request.comment == "已确认，正在处理"


class TestModelSerialization:
    """测试模型序列化"""
    
    def test_json_serialization(self):
        """测试JSON序列化"""
        metric = MonitoringMetricSchema(
            timestamp=datetime.now(),
            metric_name="test_metric",
            metric_type=MetricType.GAUGE,
            metric_value=100.0,
            source_component="test"
        )
        
        # 测试转换为JSON字符串
        json_str = metric.json()
        assert isinstance(json_str, str)
        assert "test_metric" in json_str
        
        # 测试从JSON字符串解析
        parsed_metric = MonitoringMetricSchema.parse_raw(json_str)
        assert parsed_metric.metric_name == "test_metric"
        assert parsed_metric.metric_value == 100.0
    
    def test_dict_serialization(self):
        """测试字典序列化"""
        alert = AlertRecordSchema(
            alert_id="TEST_001",
            alert_level=AlertLevel.INFO,
            alert_type="test",
            title="测试告警"
        )
        
        # 测试转换为字典
        alert_dict = alert.dict()
        assert isinstance(alert_dict, dict)
        assert alert_dict["alert_id"] == "TEST_001"
        assert alert_dict["alert_level"] == "info"
        
        # 测试从字典解析
        parsed_alert = AlertRecordSchema.parse_obj(alert_dict)
        assert parsed_alert.alert_id == "TEST_001"
        assert parsed_alert.alert_level == AlertLevel.INFO


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])