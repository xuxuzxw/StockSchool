#!/usr/bin/env python3
"""
简单测试监控数据收集器的基本功能
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_collector_imports():
    """测试收集器模块导入"""
    print("测试收集器模块导入...")
    
    try:
        # 测试基本类导入
        from src.monitoring.collectors import (
            CollectorConfig, BaseCollector, DatabaseHealthCollector,
            RedisHealthCollector, CeleryHealthCollector, APIHealthCollector,
            SystemHealthCollector
        )
        print("✅ 收集器类导入成功")
        
        # 测试便捷函数导入
        from src.monitoring.collectors import (
            collect_system_health, collect_database_health,
            collect_redis_health, collect_celery_health, collect_api_health
        )
        print("✅ 便捷函数导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_collector_config():
    """测试收集器配置"""
    print("\n测试收集器配置...")
    
    try:
        from src.monitoring.collectors import CollectorConfig
        
        # 测试默认配置
        config = CollectorConfig()
        print(f"✅ 默认配置创建成功")
        print(f"  - 数据库URL: {config.database_url}")
        print(f"  - Redis主机: {config.redis_host}:{config.redis_port}")
        print(f"  - 收集间隔: {config.collection_interval}s")
        print(f"  - 最大重试: {config.max_retries}")
        
        # 测试自定义配置
        custom_config = CollectorConfig(
            database_url="postgresql://test:test@localhost:5432/test",
            redis_host="custom-redis",
            collection_interval=60.0
        )
        print(f"✅ 自定义配置创建成功")
        print(f"  - 自定义数据库URL: {custom_config.database_url}")
        print(f"  - 自定义Redis主机: {custom_config.redis_host}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

async def test_mock_collectors():
    """测试模拟收集器功能"""
    print("\n测试模拟收集器功能...")
    
    try:
        from src.monitoring.collectors import (
            CollectorConfig, DatabaseHealthCollector, RedisHealthCollector,
            CeleryHealthCollector, APIHealthCollector
        )
        
        config = CollectorConfig()
        
        # 测试各个收集器
        collectors = [
            ("数据库", DatabaseHealthCollector(config)),
            ("Redis", RedisHealthCollector(config)),
            ("Celery", CeleryHealthCollector(config)),
            ("API", APIHealthCollector(config))
        ]
        
        for name, collector in collectors:
            print(f"  📊 测试{name}收集器...")
            
            # 测试单次收集
            result = await collector.collect()
            
            # 验证基本结构
            assert isinstance(result, dict), f"{name}收集器返回类型错误"
            
            # 验证状态字段
            status_key = 'connection_status' if name != 'API' else 'status'
            assert status_key in result, f"{name}收集器缺少状态字段"
            assert result[status_key] in ['healthy', 'warning', 'critical'], f"{name}收集器状态值无效"
            
            # 验证错误字段
            assert 'last_error' in result, f"{name}收集器缺少错误字段"
            
            print(f"    ✅ {name}收集器基本功能正常")
            print(f"    - 状态: {result[status_key]}")
            
            # 显示一些关键指标
            if name == "数据库":
                print(f"    - 连接数: {result.get('connection_count', 'N/A')}")
                print(f"    - 查询时间: {result.get('query_avg_time_ms', 'N/A')}ms")
            elif name == "Redis":
                print(f"    - 内存使用: {result.get('memory_usage_percent', 'N/A')}%")
                print(f"    - 缓存命中率: {result.get('cache_hit_rate', 'N/A')}%")
            elif name == "Celery":
                print(f"    - 活跃任务: {result.get('active_tasks', 'N/A')}")
                print(f"    - 工作进程: {result.get('worker_count', 'N/A')}")
            elif name == "API":
                print(f"    - 响应时间: {result.get('response_time_ms', 'N/A')}ms")
                print(f"    - 错误率: {result.get('error_rate', 'N/A')}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟收集器测试失败: {e}")
        return False

async def test_system_health_collector():
    """测试系统健康综合收集器"""
    print("\n测试系统健康综合收集器...")
    
    try:
        from src.monitoring.collectors import SystemHealthCollector, CollectorConfig
        
        config = CollectorConfig()
        collector = SystemHealthCollector(config)
        
        # 测试系统健康收集
        result = await collector.collect()
        
        # 验证基本结构
        required_fields = ['database', 'redis', 'celery', 'api', 'overall_status', 'timestamp']
        for field in required_fields:
            assert field in result, f"系统健康收集器缺少字段: {field}"
        
        print("✅ 系统健康收集器基本功能正常")
        print(f"  - 整体状态: {result['overall_status']}")
        print(f"  - 时间戳: {result['timestamp']}")
        
        # 验证各组件状态
        components = ['database', 'redis', 'celery', 'api']
        for component in components:
            comp_data = result[component]
            status_key = 'connection_status' if component != 'api' else 'status'
            status = comp_data.get(status_key, 'unknown')
            print(f"  - {component.capitalize()}状态: {status}")
        
        # 验证收集摘要
        if 'collection_summary' in result:
            summary = result['collection_summary']
            print(f"  - 健康组件: {summary.get('healthy_components', 0)}")
            print(f"  - 警告组件: {summary.get('warning_components', 0)}")
            print(f"  - 严重组件: {summary.get('critical_components', 0)}")
        
        # 测试统计信息
        stats = collector.get_all_stats()
        print(f"✅ 收集器统计信息获取成功，包含 {len(stats)} 个收集器")
        
        # 关闭收集器
        await collector.close()
        print("✅ 收集器关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 系统健康收集器测试失败: {e}")
        return False

async def test_convenience_functions():
    """测试便捷函数"""
    print("\n测试便捷函数...")
    
    try:
        from src.monitoring.collectors import collect_system_health
        
        # 测试系统健康收集便捷函数
        result = await collect_system_health()
        
        assert isinstance(result, dict), "便捷函数返回类型错误"
        assert 'overall_status' in result, "便捷函数缺少整体状态"
        
        print("✅ 系统健康收集便捷函数正常")
        print(f"  - 整体状态: {result['overall_status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 便捷函数测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n测试文件结构...")
    
    required_files = [
        'src/monitoring/__init__.py',
        'src/monitoring/collectors.py',
        'src/tests/test_monitoring_collectors.py'
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
    print("🚀 开始测试监控数据收集器...")
    
    # 同步测试
    sync_tests = [
        ("文件结构", test_file_structure),
        ("模块导入", test_collector_imports),
        ("收集器配置", test_collector_config)
    ]
    
    # 异步测试
    async_tests = [
        ("模拟收集器", test_mock_collectors),
        ("系统健康收集器", test_system_health_collector),
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
        print("🎉 所有测试通过！监控数据收集器工作正常")
        print("📝 任务3完成状态:")
        print("  ✅ 创建了src/monitoring/collectors.py文件")
        print("  ✅ 实现了SystemHealthCollector类")
        print("  ✅ 实现了数据库、Redis、Celery、API服务状态收集")
        print("  ✅ 添加了异常处理和重试机制")
        print("  ✅ 创建了集成测试文件")
        print("  ✅ 所有收集器功能正常，支持模拟数据")
        return True
    else:
        print("❌ 部分测试失败，请检查收集器实现")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)