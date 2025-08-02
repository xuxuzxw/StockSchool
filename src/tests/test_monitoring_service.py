"""
监控数据服务单元测试

测试监控数据的缓存、存储和查询功能
包含缓存管理器、指标存储、告警管理等测试

作者: StockSchool Team
创建时间: 2025-01-02
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.services.monitoring_service import (
    CacheConfig, StorageConfig, CacheManager, MetricsStorage,
    AlertManager, MonitoringService, create_monitoring_service
)
from src.schemas.monitoring_schemas import (
    MonitoringMetricSchema, MetricType, AlertRecordSchema, AlertLevel, AlertStatus
)


class TestCacheConfig:
    """测试缓存配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = CacheConfig()
        
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.metrics_ttl == 300
        assert config.key_prefix == "stockschool:monitoring"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = CacheConfig(
            redis_host="custom-redis",
            redis_port=6380,
            metrics_ttl=600,
            key_prefix="custom:monitoring"
        )
        
        assert config.redis_host == "custom-redis"
        assert config.redis_port == 6380
        assert config.metrics_ttl == 600
        assert config.key_prefix == "custom:monitoring"


class TestStorageConfig:
    """测试存储配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = StorageConfig()
        
        assert "postgresql://" in config.database_url
        assert config.metrics_retention_days == 90
        assert config.batch_size == 1000
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = StorageConfig(
            database_url="postgresql://test:test@localhost:5432/test",
            metrics_retention_days=30,
            batch_size=500
        )
        
        assert config.database_url == "postgresql://test:test@localhost:5432/test"
        assert config.metrics_retention_days == 30
        assert config.batch_size == 500


class TestCacheManager:
    """测试缓存管理器"""
    
    @pytest.fixture
    def cache_config(self):
        """创建测试缓存配置"""
        return CacheConfig()
    
    @pytest.fixture
    def cache_manager(self, cache_config):
        """创建缓存管理器实例"""
        return CacheManager(cache_config)
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, cache_manager):
        """测试缓存管理器初始化"""
        await cache_manager.initialize()
        
        # 验证初始化状态
        stats = cache_manager.get_stats()
        assert isinstance(stats, dict)
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'redis_available' in stats
        
        await cache_manager.close()
    
    def test_generate_key(self, cache_manager):
        """测试缓存键生成"""
        key = cache_manager._generate_key("test", "key1", "key2")
        expected = "stockschool:monitoring:test:key1:key2"
        assert key == expected
    
    @pytest.mark.asyncio
    async def test_cache_operations_without_redis(self, cache_manager):
        """测试无Redis环境下的缓存操作"""
        await cache_manager.initialize()
        
        # 测试设置缓存（应该失败但不抛异常）
        result = await cache_manager.set("test_key", {"data": "test"}, 300)
        assert result is False
        
        # 测试获取缓存（应该返回None）
        value = await cache_manager.get("test_key")
        assert value is None
        
        # 测试删除缓存（应该失败但不抛异常）
        result = await cache_manager.delete("test_key")
        assert result is False
        
        await cache_manager.close()
    
    @pytest.mark.asyncio
    async def test_get_or_compute(self, cache_manager):
        """测试获取或计算缓存"""
        await cache_manager.initialize()
        
        # 定义计算函数
        async def compute_func(x, y):
            return x + y
        
        # 测试计算并缓存
        result = await cache_manager.get_or_compute(
            "compute_test", compute_func, 300, 10, 20
        )
        assert result == 30
        
        await cache_manager.close()
    
    def test_cache_stats(self, cache_manager):
        """测试缓存统计"""
        stats = cache_manager.get_stats()
        
        required_fields = ['hits', 'misses', 'sets', 'deletes', 'errors', 'total_operations', 'hit_rate_percent', 'redis_available']
        for field in required_fields:
            assert field in stats
        
        assert isinstance(stats['hit_rate_percent'], (int, float))
        assert 0 <= stats['hit_rate_percent'] <= 100


class TestMetricsStorage:
    """测试指标存储管理器"""
    
    @pytest.fixture
    def storage_config(self):
        """创建测试存储配置"""
        return StorageConfig()
    
    @pytest.fixture
    def metrics_storage(self, storage_config):
        """创建指标存储实例"""
        return MetricsStorage(storage_config)
    
    @pytest.mark.asyncio
    async def test_storage_initialization(self, metrics_storage):
        """测试存储初始化"""
        await metrics_storage.initialize()
        
        # 验证初始化状态
        stats = metrics_storage.get_stats()
        assert isinstance(stats, dict)
        assert 'writes' in stats
        assert 'reads' in stats
        assert 'database_available' in stats
        
        await metrics_storage.close()
    
    @pytest.mark.asyncio
    async def test_store_metric_without_database(self, metrics_storage):
        """测试无数据库环境下的指标存储"""
        await metrics_storage.initialize()
        
        # 创建测试指标
        metric = MonitoringMetricSchema(
            timestamp=datetime.now(),
            metric_name="test_metric",
            metric_type=MetricType.GAUGE,
            metric_value=100.0,
            source_component="test"
        )
        
        # 测试存储（应该失败但不抛异常）
        result = await metrics_storage.store_metric(metric)
        assert result is False
        
        await metrics_storage.close()
    
    @pytest.mark.asyncio
    async def test_batch_store_metrics(self, metrics_storage):
        """测试批量存储指标"""
        await metrics_storage.initialize()
        
        # 创建测试指标列表
        metrics = []
        for i in range(5):
            metric = MonitoringMetricSchema(
                timestamp=datetime.now(),
                metric_name=f"test_metric_{i}",
                metric_type=MetricType.GAUGE,
                metric_value=float(i * 10),
                source_component="test"
            )
            metrics.append(metric)
        
        # 测试批量存储
        stored_count = await metrics_storage.store_metrics_batch(metrics)
        # 在无数据库环境下应该返回0
        assert stored_count == 0
        
        await metrics_storage.close()
    
    @pytest.mark.asyncio
    async def test_query_metrics(self, metrics_storage):
        """测试指标查询"""
        await metrics_storage.initialize()
        
        # 测试查询
        results = await metrics_storage.query_metrics(
            metric_names=["test_metric"],
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        # 在无数据库环境下应该返回空列表
        assert isinstance(results, list)
        assert len(results) == 0
        
        await metrics_storage.close()
    
    def test_storage_stats(self, metrics_storage):
        """测试存储统计"""
        stats = metrics_storage.get_stats()
        
        required_fields = ['writes', 'reads', 'errors', 'batch_writes', 'database_available']
        for field in required_fields:
            assert field in stats


class TestMonitoringService:
    """测试监控服务主类"""
    
    @pytest.fixture
    def monitoring_service(self):
        """创建监控服务实例"""
        return MonitoringService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, monitoring_service):
        """测试服务初始化"""
        await monitoring_service.initialize()
        
        assert monitoring_service._initialized is True
        
        # 验证组件初始化
        assert monitoring_service.cache is not None
        assert monitoring_service.storage is not None
        assert monitoring_service.alerts is not None
        
        await monitoring_service.close()
        assert monitoring_service._initialized is False
    
    @pytest.mark.asyncio
    async def test_service_context_manager(self):
        """测试服务上下文管理器"""
        service = MonitoringService()
        
        async with service.service_context() as svc:
            assert svc._initialized is True
            assert svc is service
        
        assert service._initialized is False
    
    @pytest.mark.asyncio
    async def test_store_and_get_system_health(self, monitoring_service):
        """测试系统健康数据存储和获取"""
        await monitoring_service.initialize()
        
        # 创建模拟健康数据
        health_data = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': {'connection_status': 'healthy'},
            'redis': {'connection_status': 'healthy'},
            'celery': {'connection_status': 'healthy'},
            'api': {'status': 'healthy'}
        }
        
        # 由于没有真实的SystemHealthMetrics对象，我们直接测试缓存功能
        cache_key = monitoring_service.cache._generate_key("system_health", "latest")
        await monitoring_service.cache.set(cache_key, health_data, 60)
        
        # 测试获取
        retrieved_data = await monitoring_service.get_system_health()
        # 在无Redis环境下可能返回None
        
        await monitoring_service.close()
    
    @pytest.mark.asyncio
    async def test_metrics_batch_storage(self, monitoring_service):
        """测试批量指标存储"""
        await monitoring_service.initialize()
        
        # 创建测试指标
        metrics = []
        for i in range(3):
            metric = MonitoringMetricSchema(
                timestamp=datetime.now(),
                metric_name=f"batch_metric_{i}",
                metric_type=MetricType.COUNTER,
                metric_value=float(i),
                source_component="batch_test"
            )
            metrics.append(metric)
        
        # 测试批量存储
        stored_count = await monitoring_service.store_metrics_batch(metrics)
        # 在无数据库环境下应该返回0
        assert stored_count == 0
        
        await monitoring_service.close()
    
    @pytest.mark.asyncio
    async def test_alert_operations(self, monitoring_service):
        """测试告警操作"""
        await monitoring_service.initialize()
        
        # 创建测试告警
        alert = AlertRecordSchema(
            alert_id="TEST_ALERT_001",
            alert_level=AlertLevel.WARNING,
            alert_type="test",
            title="测试告警",
            description="这是一个测试告警"
        )
        
        # 测试创建告警
        created = await monitoring_service.create_alert(alert)
        # 在无数据库环境下应该返回False
        assert created is False
        
        # 测试获取活跃告警
        alerts = await monitoring_service.get_active_alerts()
        assert isinstance(alerts, list)
        
        await monitoring_service.close()
    
    def test_service_stats(self, monitoring_service):
        """测试服务统计"""
        stats = monitoring_service.get_service_stats()
        
        required_fields = ['initialized', 'cache_stats', 'storage_stats', 'alert_stats']
        for field in required_fields:
            assert field in stats
        
        assert isinstance(stats['cache_stats'], dict)
        assert isinstance(stats['storage_stats'], dict)
        assert isinstance(stats['alert_stats'], dict)


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    @pytest.mark.asyncio
    async def test_create_monitoring_service(self):
        """测试创建监控服务便捷函数"""
        service = await create_monitoring_service()
        
        assert isinstance(service, MonitoringService)
        assert service._initialized is True
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_create_monitoring_service_with_custom_config(self):
        """测试使用自定义配置创建监控服务"""
        cache_config = CacheConfig(redis_host="custom-host")
        storage_config = StorageConfig(batch_size=500)
        
        service = await create_monitoring_service(cache_config, storage_config)
        
        assert service.cache_config.redis_host == "custom-host"
        assert service.storage_config.batch_size == 500
        
        await service.close()


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """测试完整工作流程"""
        # 创建服务
        async with MonitoringService().service_context() as service:
            # 测试指标存储
            metrics = [
                MonitoringMetricSchema(
                    timestamp=datetime.now(),
                    metric_name="integration_test",
                    metric_type=MetricType.GAUGE,
                    metric_value=50.0,
                    source_component="integration"
                )
            ]
            
            stored_count = await service.store_metrics_batch(metrics)
            # 在模拟环境下验证不会抛异常
            assert isinstance(stored_count, int)
            
            # 测试查询
            results = await service.query_metrics_with_cache(
                metric_names=["integration_test"],
                start_time=datetime.now() - timedelta(minutes=5),
                end_time=datetime.now()
            )
            assert isinstance(results, list)
            
            # 测试统计信息
            stats = service.get_service_stats()
            assert stats['initialized'] is True


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])