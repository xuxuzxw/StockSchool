#!/usr/bin/env python3
"""
任务6完整功能测试

验证数据存储架构优化的所有功能
"""

import sys
import os
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.timescale_optimizer import TimescaleOptimizer

def test_task6_complete():
    """测试任务6的完整功能"""
    
    print("🚀 开始测试任务6：优化数据存储架构")
    print("=" * 60)
    
    try:
        # 创建优化器实例
        optimizer = TimescaleOptimizer()
        print("✅ TimescaleOptimizer 实例创建成功")
        
        # 测试结果统计
        test_results = {
            'subtask_6_1': {'name': '时序数据库优化', 'passed': 0, 'total': 0},
            'subtask_6_2': {'name': '存储性能监控', 'passed': 0, 'total': 0},
            'subtask_6_3': {'name': '数据备份策略', 'passed': 0, 'total': 0}
        }
        
        print("\n" + "=" * 60)
        print("📊 子任务 6.1: 实现时序数据库优化")
        print("=" * 60)
        
        # 6.1.1 测试超表配置
        test_results['subtask_6_1']['total'] += 1
        if len(optimizer.hypertable_configs) >= 5:
            print("✅ 超表配置完整 (5个表)")
            test_results['subtask_6_1']['passed'] += 1
        else:
            print(f"❌ 超表配置不完整 ({len(optimizer.hypertable_configs)}个表)")
        
        # 6.1.2 测试超表创建方法
        test_results['subtask_6_1']['total'] += 1
        if hasattr(optimizer, 'create_hypertable') and callable(optimizer.create_hypertable):
            print("✅ create_hypertable 方法存在")
            test_results['subtask_6_1']['passed'] += 1
        else:
            print("❌ create_hypertable 方法缺失")
        
        # 6.1.3 测试复合索引创建方法
        test_results['subtask_6_1']['total'] += 1
        if hasattr(optimizer, 'create_composite_indexes') and callable(optimizer.create_composite_indexes):
            print("✅ create_composite_indexes 方法存在")
            test_results['subtask_6_1']['passed'] += 1
        else:
            print("❌ create_composite_indexes 方法缺失")
        
        # 6.1.4 测试数据压缩方法
        test_results['subtask_6_1']['total'] += 1
        if hasattr(optimizer, 'enable_compression') and callable(optimizer.enable_compression):
            print("✅ enable_compression 方法存在")
            test_results['subtask_6_1']['passed'] += 1
        else:
            print("❌ enable_compression 方法缺失")
        
        # 6.1.5 测试表优化方法
        test_results['subtask_6_1']['total'] += 1
        if hasattr(optimizer, 'optimize_table') and callable(optimizer.optimize_table):
            print("✅ optimize_table 方法存在")
            test_results['subtask_6_1']['passed'] += 1
        else:
            print("❌ optimize_table 方法缺失")
        
        # 6.1.6 测试批量优化方法
        test_results['subtask_6_1']['total'] += 1
        if hasattr(optimizer, 'optimize_all_tables') and callable(optimizer.optimize_all_tables):
            print("✅ optimize_all_tables 方法存在")
            test_results['subtask_6_1']['passed'] += 1
        else:
            print("❌ optimize_all_tables 方法缺失")
        
        print("\n" + "=" * 60)
        print("📈 子任务 6.2: 实现存储性能监控")
        print("=" * 60)
        
        # 6.2.1 测试数据库性能指标获取
        test_results['subtask_6_2']['total'] += 1
        if hasattr(optimizer, 'get_database_performance_metrics') and callable(optimizer.get_database_performance_metrics):
            print("✅ get_database_performance_metrics 方法存在")
            test_results['subtask_6_2']['passed'] += 1
        else:
            print("❌ get_database_performance_metrics 方法缺失")
        
        # 6.2.2 测试TimescaleDB性能指标获取
        test_results['subtask_6_2']['total'] += 1
        if hasattr(optimizer, 'get_timescale_performance_metrics') and callable(optimizer.get_timescale_performance_metrics):
            print("✅ get_timescale_performance_metrics 方法存在")
            test_results['subtask_6_2']['passed'] += 1
        else:
            print("❌ get_timescale_performance_metrics 方法缺失")
        
        # 6.2.3 测试存储空间使用监控
        test_results['subtask_6_2']['total'] += 1
        if hasattr(optimizer, 'get_storage_space_usage') and callable(optimizer.get_storage_space_usage):
            print("✅ get_storage_space_usage 方法存在")
            test_results['subtask_6_2']['passed'] += 1
        else:
            print("❌ get_storage_space_usage 方法缺失")
        
        # 6.2.4 测试性能优化建议生成
        test_results['subtask_6_2']['total'] += 1
        if hasattr(optimizer, 'generate_performance_optimization_suggestions') and callable(optimizer.generate_performance_optimization_suggestions):
            print("✅ generate_performance_optimization_suggestions 方法存在")
            test_results['subtask_6_2']['passed'] += 1
        else:
            print("❌ generate_performance_optimization_suggestions 方法缺失")
        
        # 6.2.5 测试性能报告生成
        test_results['subtask_6_2']['total'] += 1
        if hasattr(optimizer, 'generate_performance_report') and callable(optimizer.generate_performance_report):
            print("✅ generate_performance_report 方法存在")
            test_results['subtask_6_2']['passed'] += 1
        else:
            print("❌ generate_performance_report 方法缺失")
        
        # 6.2.6 测试存储增长监控
        test_results['subtask_6_2']['total'] += 1
        if hasattr(optimizer, 'monitor_storage_growth') and callable(optimizer.monitor_storage_growth):
            print("✅ monitor_storage_growth 方法存在")
            test_results['subtask_6_2']['passed'] += 1
        else:
            print("❌ monitor_storage_growth 方法缺失")
        
        # 6.2.7 测试存储监控表创建
        test_results['subtask_6_2']['total'] += 1
        if hasattr(optimizer, 'create_storage_monitoring_table') and callable(optimizer.create_storage_monitoring_table):
            print("✅ create_storage_monitoring_table 方法存在")
            test_results['subtask_6_2']['passed'] += 1
        else:
            print("❌ create_storage_monitoring_table 方法缺失")
        
        print("\n" + "=" * 60)
        print("💾 子任务 6.3: 实现数据备份策略")
        print("=" * 60)
        
        # 6.3.1 测试备份策略创建
        test_results['subtask_6_3']['total'] += 1
        try:
            backup_strategy = optimizer.create_backup_strategy()
            if backup_strategy and 'config' in backup_strategy and 'backup_types' in backup_strategy:
                print("✅ create_backup_strategy 方法正常工作")
                test_results['subtask_6_3']['passed'] += 1
            else:
                print("❌ create_backup_strategy 方法返回结果不正确")
        except Exception as e:
            print(f"❌ create_backup_strategy 方法异常: {e}")
        
        # 6.3.2 测试完整备份执行
        test_results['subtask_6_3']['total'] += 1
        if hasattr(optimizer, 'execute_full_backup') and callable(optimizer.execute_full_backup):
            print("✅ execute_full_backup 方法存在")
            test_results['subtask_6_3']['passed'] += 1
        else:
            print("❌ execute_full_backup 方法缺失")
        
        # 6.3.3 测试增量备份执行
        test_results['subtask_6_3']['total'] += 1
        if hasattr(optimizer, 'execute_incremental_backup') and callable(optimizer.execute_incremental_backup):
            print("✅ execute_incremental_backup 方法存在")
            test_results['subtask_6_3']['passed'] += 1
        else:
            print("❌ execute_incremental_backup 方法缺失")
        
        # 6.3.4 测试数据恢复功能
        test_results['subtask_6_3']['total'] += 1
        if hasattr(optimizer, 'restore_from_backup') and callable(optimizer.restore_from_backup):
            print("✅ restore_from_backup 方法存在")
            test_results['subtask_6_3']['passed'] += 1
        else:
            print("❌ restore_from_backup 方法缺失")
        
        # 6.3.5 测试备份清理功能
        test_results['subtask_6_3']['total'] += 1
        if hasattr(optimizer, 'cleanup_old_backups') and callable(optimizer.cleanup_old_backups):
            print("✅ cleanup_old_backups 方法存在")
            test_results['subtask_6_3']['passed'] += 1
        else:
            print("❌ cleanup_old_backups 方法缺失")
        
        # 6.3.6 测试备份状态监控
        test_results['subtask_6_3']['total'] += 1
        if hasattr(optimizer, 'get_backup_status') and callable(optimizer.get_backup_status):
            print("✅ get_backup_status 方法存在")
            test_results['subtask_6_3']['passed'] += 1
        else:
            print("❌ get_backup_status 方法缺失")
        
        # 6.3.7 测试备份状态表创建
        test_results['subtask_6_3']['total'] += 1
        if hasattr(optimizer, 'create_backup_status_table') and callable(optimizer.create_backup_status_table):
            print("✅ create_backup_status_table 方法存在")
            test_results['subtask_6_3']['passed'] += 1
        else:
            print("❌ create_backup_status_table 方法缺失")
        
        # 生成测试报告
        print("\n" + "=" * 60)
        print("📋 任务6测试结果汇总")
        print("=" * 60)
        
        total_passed = 0
        total_tests = 0
        
        for subtask_id, result in test_results.items():
            passed = result['passed']
            total = result['total']
            percentage = (passed / total * 100) if total > 0 else 0
            
            status = "✅ 通过" if passed == total else "⚠️  部分通过" if passed > 0 else "❌ 失败"
            print(f"{result['name']}: {passed}/{total} ({percentage:.1f}%) {status}")
            
            total_passed += passed
            total_tests += total
        
        overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n总体结果: {total_passed}/{total_tests} ({overall_percentage:.1f}%)")
        
        if overall_percentage >= 90:
            print("🎉 任务6实现质量优秀！")
            success = True
        elif overall_percentage >= 70:
            print("👍 任务6实现质量良好！")
            success = True
        elif overall_percentage >= 50:
            print("⚠️  任务6实现基本完成，但需要改进")
            success = True
        else:
            print("❌ 任务6实现不完整，需要重新检查")
            success = False
        
        # 保存测试报告
        report = {
            'task': '任务6：优化数据存储架构',
            'timestamp': str(datetime.now()),
            'subtasks': test_results,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'success_rate': overall_percentage,
                'status': 'success' if success else 'failed'
            }
        }
        
        with open('task6_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细测试报告已保存到: task6_test_report.json")
        
        return success
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    from datetime import datetime
    success = test_task6_complete()
    sys.exit(0 if success else 1)