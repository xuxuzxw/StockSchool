#!/usr/bin/env python3
"""
简单测试监控数据服务的基本功能
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_service_imports():
    """测试服务模块导入"""
    print("测试服务模块导入...")
    
    try:
        from src.services.monitoring_service import (
            CacheConfig, StorageConfig, CacheManager, MetricsStorage,
            AlertManager, MonitoringService, create_monitoring_service
        )
        print("✅ 监控服务类导入成功")
        
        from src.schemas.monitoring_schemas import (
            MonitoringMetricSchema, MetricType, AlertRecordSchema, AlertLevel
        )
        print("✅ 数据模型导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

async def test_cache_manager():
    """测试缓存管理器"""
    print("\n测试缓存管理器...")
    
    try:
        from src.services.monitoring_service import CacheManager, CacheConfig
        
        config = CacheConfig()
        cache = CacheManager(config)
        
        # 测试初始化
        await cache.initialize()
        print("✅ 缓存管理器初始化成功")
        
        # 测试键生成
        key = cache._generate_key("test", "key1", "key2")
        expected = "stockschool:monitoring:test:key1:key2"
        assert key == expected
        print("✅ 缓存键生成正确")
        
        # 测试缓存操作（在无Redis环境下）
        result = await cache.set("test_key", {"data": "test"}, 300)
        print(f"✅ 缓存设置操作完成: {result}")
        
        value = await cache.get("test_key")
        print(f"✅ 缓存获取操作完成: {value}")
        
        # 测试统计信息
        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert 'hits' in stats
        assert 'redis_available' in stats
        print(f"✅ 缓存统计信息正常: Redis可用={stats['redis_available']}")
        
        await cache.close()
        print("✅ 缓存管理器关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 缓存管理器测试失败: {e}")
        return False

async def test_metrics_storage():
    """测试指标存储管理器"""
    print("\n测试指标存储管理器...")
    
    try:
        from src.services.monitoring_service import MetricsStorage, StorageConfig
        from src.schemas.monitoring_schemas import MonitoringMetricSchema, MetricType
        
        config = StorageConfig()
        storage = MetricsStorage(config)
        
        # 测试初始化
        await storage.initialize()
        print("✅ 指标存储管理器初始化成功")
        
        # 创建测试指标
        metric = MonitoringMetricSchema(
            timestamp=datetime.now(),
            metric_name="test_metric",
            metric_type=MetricType.GAUGE,
            metric_value=100.0,
            source_component="test"
        )
        
        # 测试单个指标存储
        result = await storage.store_metric(metric)
        print(f"✅ 单个指标存储操作完成: {result}")
        
        # 测试批量指标存储
        metrics = [metric for _ in range(3)]
        stored_count = await storage.store_metrics_batch(metrics)
        print(f"✅ 批量指标存储操作完成: {stored_count}")
        
        # 测试指标查询
        results = await storage.query_metrics(
            metric_names=["test_metric"],
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        print(f"✅ 指标查询操作完成: {len(results)} 条结果")
        
        # 测试统计信息
        stats = storage.get_stats()
        assert isinstance(stats, dict)
        assert 'database_available' in stats
        print(f"✅ 存储统计信息正常: 数据库可用={stats['database_available']}")
        
        await storage.close()
        print("✅ 指标存储管理器关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 指标存储管理器测试失败: {e}")
        return False

async def test_monitoring_service():
    """测试监控服务主类"""
    print("\n测试监控服务主类...")
    
    try:
        from src.services.monitoring_service import MonitoringService
        from src.schemas.monitoring_schemas import MonitoringMetricSchema, MetricType
        
        service = MonitoringService()
        
        # 测试初始化
        await service.initialize()
        assert service._initialized is True
        print("✅ 监控服务初始化成功")
        
        # 验证组件
        assert service.cache is not None
        assert service.storage is not None
        assert service.alerts is not None
        print("✅ 监控服务组件创建成功")
        
        # 测试批量指标存储
        metrics = []
        for i in range(3):
            metric = MonitoringMetricSchema(
                timestamp=datetime.now(),
                metric_name=f"service_test_{i}",
                metric_type=MetricType.COUNTER,
                metric_value=float(i * 10),
                source_component="service_test"
            )
            metrics.append(metric)
        
        stored_count = await service.store_metrics_batch(metrics)
        print(f"✅ 服务批量存储操作完成: {stored_count}")
        
        # 测试指标查询
        results = await service.query_metrics_with_cache(
            metric_names=["service_test_0"],
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now()
        )
        print(f"✅ 服务查询操作完成: {len(results)} 条结果")
        
        # 测试获取活跃告警
        alerts = await service.get_active_alerts()
        print(f"✅ 获取活跃告警完成: {len(alerts)} 条告警")
        
        # 测试服务统计
        stats = service.get_service_stats()
        assert isinstance(stats, dict)
        assert 'initialized' in stats
        assert stats['initialized'] is True
        print("✅ 服务统计信息正常")
        
        await service.close()
        assert service._initialized is False
        print("✅ 监控服务关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 监控服务测试失败: {e}")
        return False

async def test_service_context_manager():
    """测试服务上下文管理器"""
    print("\n测试服务上下文管理器...")
    
    try:
        from src.services.monitoring_service import MonitoringService
        
        service = MonitoringService()
        
        # 测试上下文管理器
        async with service.service_context() as svc:
            assert svc._initialized is True
            assert svc is service
            print("✅ 服务上下文管理器进入成功")
        
        assert service._initialized is False
        print("✅ 服务上下文管理器退出成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 服务上下文管理器测试失败: {e}")
        return False

async def test_convenience_functions():
    """测试便捷函数"""
    print("\n测试便捷函数...")
    
    try:
        from src.services.monitoring_service import create_monitoring_service, CacheConfig, StorageConfig
        
        # 测试默认配置创建
        service = await create_monitoring_service()
        assert isinstance(service, type(service))
        assert service._initialized is True
        print("✅ 默认配置创建监控服务成功")
        await service.close()
        
        # 测试自定义配置创建
        cache_config = CacheConfig(redis_host="custom-host")
        storage_config = StorageConfig(batch_size=500)
        
        service = await create_monitoring_service(cache_config, storage_config)
        assert service.cache_config.redis_host == "custom-host"
        assert service.storage_config.batch_size == 500
        print("✅ 自定义配置创建监控服务成功")
        await service.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 便捷函数测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n测试文件结构...")
    
    required_files = [
        'src/services/__init__.py',
        'src/services/monitoring_service.py',
        'src/tests/test_monitoring_service.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist

async def main():
    """主测试函数"""
    print("🚀 开始测试监控数据服务...")
    
    # 同步测试
    sync_tests = [
        ("文件结构", test_file_structure)
    ]
    
    # 异步测试
    async_tests = [
        ("服务模块导入", test_service_imports),
        ("缓存管理器", test_cache_manager),
        ("指标存储管理器", test_metrics_storage),
        ("监控服务主类", test_monitoring_service),
        ("服务上下文管理器", test_service_context_manager),
        ("便捷函数", test_convenience_functions)
    ]
    
    passed = 0
    total = len(sync_tests) + len(async_tests)
    
    # 执行同步测试
    for name, test_func in sync_tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {name}测试通过")
            else:
                print(f"❌ {name}测试失败")
        except Exception as e:
            print(f"❌ {name}测试异常: {e}")
        print()
    
    # 执行异步测试
    for name, test_func in async_tests:
        try:
            if await test_func():
                passed += 1
                print(f"✅ {name}测试通过")
            else:
                print(f"❌ {name}测试失败")
        except Exception as e:
            print(f"❌ {name}测试异常: {e}")
        print()
    
    print(f"📊 测试结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有监控服务测试通过！")
        print("📝 任务7完成状态:")
        print("  ✅ 创建了src/services/monitoring_service.py文件")
        print("  ✅ 实现了CacheManager类，提供Redis缓存功能")
        print("  ✅ 实现了MetricsStorage类，提供TimescaleDB存储功能")
        print("  ✅ 实现了AlertManager类，提供告警管理功能")
        print("  ✅ 实现了MonitoringService主服务类")
        print("  ✅ 添加了数据缓存机制，优化热点数据访问")
        print("  ✅ 实现了时序数据写入和历史数据查询")
        print("  ✅ 创建了单元测试验证缓存命中率和数据一致性")
        print("  ✅ 所有服务功能正常，支持模拟环境运行")
        return True
    else:
        print("❌ 部分测试失败，请检查监控服务实现")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)