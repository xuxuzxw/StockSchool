#!/usr/bin/env python3
"""
测试扩展的监控数据收集器
包括数据同步、因子计算、AI模型监控收集器
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_data_sync_collector():
    """测试数据同步收集器"""
    print("测试数据同步收集器...")
    
    try:
        from src.monitoring.collectors import DataSyncCollector, CollectorConfig
        
        config = CollectorConfig()
        collector = DataSyncCollector(config)
        
        result = await collector.collect()
        
        # 验证基本结构
        required_fields = [
            'last_sync_time', 'sync_progress', 'failed_tasks',
            'data_quality_score', 'api_quota_remaining', 'api_calls_today',
            'sync_status', 'last_error'
        ]
        
        for field in required_fields:
            assert field in result, f"数据同步收集器缺少字段: {field}"
        
        print("✅ 数据同步收集器基本功能正常")
        print(f"  - 同步状态: {result['sync_status']}")
        print(f"  - 同步进度: {result['sync_progress']}%")
        print(f"  - 数据质量评分: {result['data_quality_score']}")
        print(f"  - API配额剩余: {result['api_quota_remaining']}")
        print(f"  - 失败任务数: {len(result['failed_tasks'])}")
        
        # 测试带重试的收集
        retry_result = await collector.collect_with_retry()
        assert 'collection_time_ms' in retry_result
        print("✅ 数据同步收集器重试机制正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据同步收集器测试失败: {e}")
        return False

async def test_factor_compute_collector():
    """测试因子计算收集器"""
    print("\n测试因子计算收集器...")
    
    try:
        from src.monitoring.collectors import FactorComputeCollector, CollectorConfig
        
        config = CollectorConfig()
        collector = FactorComputeCollector(config)
        
        result = await collector.collect()
        
        # 验证基本结构
        required_fields = [
            'current_tasks', 'completion_progress', 'performance_metrics',
            'success_rate', 'failed_factors', 'compute_status', 'last_error'
        ]
        
        for field in required_fields:
            assert field in result, f"因子计算收集器缺少字段: {field}"
        
        print("✅ 因子计算收集器基本功能正常")
        print(f"  - 计算状态: {result['compute_status']}")
        print(f"  - 完成进度: {result['completion_progress']}%")
        print(f"  - 成功率: {result['success_rate']}%")
        print(f"  - 当前任务数: {len(result['current_tasks'])}")
        print(f"  - 失败因子数: {len(result['failed_factors'])}")
        
        # 验证性能指标
        perf_metrics = result['performance_metrics']
        assert isinstance(perf_metrics, dict)
        assert 'cpu_usage_percent' in perf_metrics
        assert 'memory_usage_percent' in perf_metrics
        print(f"  - CPU使用率: {perf_metrics['cpu_usage_percent']}%")
        print(f"  - 内存使用率: {perf_metrics['memory_usage_percent']}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 因子计算收集器测试失败: {e}")
        return False

async def test_ai_model_collector():
    """测试AI模型收集器"""
    print("\n测试AI模型收集器...")
    
    try:
        from src.monitoring.collectors import AIModelCollector, CollectorConfig
        
        config = CollectorConfig()
        collector = AIModelCollector(config)
        
        result = await collector.collect()
        
        # 验证基本结构
        required_fields = [
            'model_info', 'prediction_accuracy', 'latest_predictions',
            'model_metrics', 'model_status', 'last_error'
        ]
        
        for field in required_fields:
            assert field in result, f"AI模型收集器缺少字段: {field}"
        
        print("✅ AI模型收集器基本功能正常")
        print(f"  - 模型状态: {result['model_status']}")
        print(f"  - 预测准确率: {result['prediction_accuracy']}%")
        
        # 验证模型信息
        model_info = result['model_info']
        assert isinstance(model_info, dict)
        print(f"  - 模型名称: {model_info.get('model_name', 'N/A')}")
        print(f"  - 模型版本: {model_info.get('version', 'N/A')}")
        print(f"  - 算法类型: {model_info.get('algorithm', 'N/A')}")
        
        # 验证预测结果
        predictions = result['latest_predictions']
        assert isinstance(predictions, list)
        print(f"  - 最新预测数量: {len(predictions)}")
        
        # 验证训练指标
        training_metrics = result.get('training_metrics')
        if training_metrics:
            print(f"  - 训练状态: {training_metrics.get('status', 'N/A')}")
            if training_metrics.get('status') == 'training':
                print(f"  - 训练进度: {training_metrics.get('progress', 0)}%")
        
        return True
        
    except Exception as e:
        print(f"❌ AI模型收集器测试失败: {e}")
        return False

async def test_extended_system_collector():
    """测试扩展的系统收集器"""
    print("\n测试扩展的系统收集器...")
    
    try:
        from src.monitoring.collectors import SystemHealthCollector, CollectorConfig
        
        config = CollectorConfig()
        collector = SystemHealthCollector(config)
        
        result = await collector.collect()
        
        # 验证基本结构
        required_fields = [
            'database', 'redis', 'celery', 'api',
            'data_sync', 'factor_compute', 'ai_model',
            'overall_status', 'timestamp', 'collection_summary'
        ]
        
        for field in required_fields:
            assert field in result, f"扩展系统收集器缺少字段: {field}"
        
        print("✅ 扩展系统收集器基本功能正常")
        print(f"  - 整体状态: {result['overall_status']}")
        
        # 验证各组件状态
        components = {
            'database': 'connection_status',
            'redis': 'connection_status', 
            'celery': 'connection_status',
            'api': 'status',
            'data_sync': 'sync_status',
            'factor_compute': 'compute_status',
            'ai_model': 'model_status'
        }
        
        for component, status_key in components.items():
            comp_data = result[component]
            status = comp_data.get(status_key, 'unknown')
            print(f"  - {component.replace('_', ' ').title()}状态: {status}")
        
        # 验证收集摘要
        summary = result['collection_summary']
        assert summary['total_components'] == 7
        print(f"  - 总组件数: {summary['total_components']}")
        print(f"  - 健康组件: {summary['healthy_components']}")
        print(f"  - 警告组件: {summary['warning_components']}")
        print(f"  - 严重组件: {summary['critical_components']}")
        
        # 验证组件数量总和
        total = (summary['healthy_components'] + 
                summary['warning_components'] + 
                summary['critical_components'])
        assert total == 7, f"组件数量总和错误: {total} != 7"
        
        # 测试统计信息
        stats = collector.get_all_stats()
        expected_collectors = [
            'database_collector', 'redis_collector', 'celery_collector',
            'api_collector', 'data_sync_collector', 'factor_compute_collector',
            'ai_model_collector', 'system_collector'
        ]
        
        for collector_name in expected_collectors:
            assert collector_name in stats, f"缺少收集器统计: {collector_name}"
        
        print(f"✅ 收集器统计信息正常，包含 {len(stats)} 个收集器")
        
        # 关闭收集器
        await collector.close()
        print("✅ 收集器关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 扩展系统收集器测试失败: {e}")
        return False

async def test_convenience_functions():
    """测试便捷函数"""
    print("\n测试便捷函数...")
    
    try:
        from src.monitoring.collectors import (
            collect_data_sync_status, collect_factor_compute_status,
            collect_ai_model_status, collect_system_health
        )
        
        # 测试数据同步便捷函数
        sync_result = await collect_data_sync_status()
        assert isinstance(sync_result, dict)
        assert 'sync_status' in sync_result
        print("✅ 数据同步便捷函数正常")
        
        # 测试因子计算便捷函数
        factor_result = await collect_factor_compute_status()
        assert isinstance(factor_result, dict)
        assert 'compute_status' in factor_result
        print("✅ 因子计算便捷函数正常")
        
        # 测试AI模型便捷函数
        model_result = await collect_ai_model_status()
        assert isinstance(model_result, dict)
        assert 'model_status' in model_result
        print("✅ AI模型便捷函数正常")
        
        # 测试系统健康便捷函数
        system_result = await collect_system_health()
        assert isinstance(system_result, dict)
        assert 'overall_status' in system_result
        assert 'data_sync' in system_result
        assert 'factor_compute' in system_result
        assert 'ai_model' in system_result
        print("✅ 系统健康便捷函数正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 便捷函数测试失败: {e}")
        return False

async def test_status_determination_logic():
    """测试状态判断逻辑"""
    print("\n测试状态判断逻辑...")
    
    try:
        from src.monitoring.collectors import (
            DataSyncCollector, FactorComputeCollector, 
            AIModelCollector, CollectorConfig
        )
        
        config = CollectorConfig()
        
        # 测试多次收集，验证状态逻辑的一致性
        collectors = [
            ("数据同步", DataSyncCollector(config)),
            ("因子计算", FactorComputeCollector(config)),
            ("AI模型", AIModelCollector(config))
        ]
        
        for name, collector in collectors:
            print(f"  📊 测试{name}状态判断逻辑...")
            
            # 多次收集验证状态逻辑
            statuses = []
            for _ in range(3):
                result = await collector.collect()
                
                if name == "数据同步":
                    status = result['sync_status']
                elif name == "因子计算":
                    status = result['compute_status']
                else:  # AI模型
                    status = result['model_status']
                
                statuses.append(status)
                assert status in ['healthy', 'warning', 'critical'], f"无效状态: {status}"
            
            print(f"    ✅ {name}状态判断逻辑正常，状态范围: {set(statuses)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 状态判断逻辑测试失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("🚀 开始测试扩展的监控数据收集器...")
    
    tests = [
        ("数据同步收集器", test_data_sync_collector),
        ("因子计算收集器", test_factor_compute_collector),
        ("AI模型收集器", test_ai_model_collector),
        ("扩展系统收集器", test_extended_system_collector),
        ("便捷函数", test_convenience_functions),
        ("状态判断逻辑", test_status_determination_logic)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
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
        print("🎉 所有扩展收集器测试通过！")
        print("📝 任务4完成状态:")
        print("  ✅ 扩展了src/monitoring/collectors.py文件")
        print("  ✅ 实现了DataSyncCollector类")
        print("  ✅ 实现了FactorComputeCollector类")
        print("  ✅ 实现了AIModelCollector类")
        print("  ✅ 更新了SystemHealthCollector类")
        print("  ✅ 添加了数据同步进度跟踪功能")
        print("  ✅ 添加了数据质量检查逻辑")
        print("  ✅ 添加了API配额监控功能")
        print("  ✅ 所有收集器功能正常，支持模拟数据")
        return True
    else:
        print("❌ 部分测试失败，请检查扩展收集器实现")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)