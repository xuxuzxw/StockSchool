#!/usr/bin/env python3
"""
阶段九：集成测试验收实现运行脚本
执行端到端集成测试、多用户并发测试、故障恢复测试等
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.acceptance.phases.integration_test_e2e import IntegrationTestE2E
    # 导入测试状态类
    class TestStatus:
        PASSED = "PASSED"
        FAILED = "FAILED"
        SKIPPED = "SKIPPED"
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，并且所有依赖都已安装")
    sys.exit(1)

def run_integration_acceptance_tests():
    """运行集成验收测试"""
    print("=" * 80)
    print("StockSchool 阶段九：集成测试验收实现")
    print("=" * 80)
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 测试配置
    test_config = {
        'test_stocks': ['000001.SZ', '000002.SZ', '600000.SH'],
        'test_date_range': {
            'start': '2024-01-01',
            'end': '2024-01-31'
        },
        'timeout_seconds': 300,
        'concurrent_config': {
            'concurrent_users': 5,
            'concurrent_operations': 10,
            'max_error_rate': 0.1
        },
        'enable_long_running_test': True
    }
    
    try:
        # 创建集成测试阶段
        integration_phase = IntegrationTestE2E("integration_test_e2e", test_config)
        
        print("🚀 开始执行集成测试验收...")
        print()
        
        # 执行测试
        start_time = time.time()
        test_results = integration_phase._run_tests()
        end_time = time.time()
        
        # 统计测试结果
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in test_results if r.status == TestStatus.FAILED])
        skipped_tests = len([r for r in test_results if r.status == TestStatus.SKIPPED])
        
        # 显示测试结果
        print("\n" + "=" * 80)
        print("集成测试验收结果汇总")
        print("=" * 80)
        
        for result in test_results:
            status_icon = "✅" if result.status == TestStatus.PASSED else "❌" if result.status == TestStatus.FAILED else "⏭️"
            print(f"{status_icon} {result.test_name}: {result.status}")
            if result.error_message:
                print(f"   错误信息: {result.error_message}")
            print(f"   执行时间: {result.execution_time:.2f}秒")
            
            # 显示详细结果
            if result.details:
                print("   详细结果:")
                for key, value in result.details.items():
                    if isinstance(value, bool):
                        icon = "✅" if value else "❌"
                        print(f"     {icon} {key}: {value}")
                    elif isinstance(value, (int, float)):
                        print(f"     📊 {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 3:
                        print(f"     📋 {key}: {len(value)} 项")
            print()
        
        # 总体统计
        print("=" * 80)
        print("测试统计:")
        print(f"  总测试数: {total_tests}")
        print(f"  通过: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"  失败: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"  跳过: {skipped_tests} ({skipped_tests/total_tests*100:.1f}%)")
        print(f"  总执行时间: {end_time - start_time:.2f}秒")
        
        # 测试通过率
        pass_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
        print(f"  通过率: {pass_rate:.1f}%")
        
        # 详细分析
        print("\n" + "=" * 80)
        print("详细分析:")
        print("=" * 80)
        
        for result in test_results:
            if result.details:
                print(f"\n📋 {result.test_name} 详细分析:")
                
                # 分析端到端测试结果
                if result.test_name == "end_to_end_data_flow_test":
                    analyze_e2e_results(result.details)
                
                # 分析并发测试结果
                elif result.test_name == "multi_user_concurrent_test":
                    analyze_concurrent_results(result.details)
                
                # 分析故障恢复测试结果
                elif result.test_name == "fault_recovery_test":
                    analyze_fault_recovery_results(result.details)
                
                # 分析长时间运行测试结果
                elif result.test_name == "long_running_stability_test":
                    analyze_stability_results(result.details)
                
                # 分析业务场景测试结果
                elif result.test_name == "business_scenario_test":
                    analyze_business_results(result.details)
        
        # 保存测试结果
        save_test_results(test_results, {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'pass_rate': pass_rate,
            'execution_time': end_time - start_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # 判断测试是否成功
        if failed_tests == 0:
            print("\n🎉 所有集成测试验收通过！系统已准备好进入下一阶段。")
            return True
        else:
            print(f"\n⚠️ 有 {failed_tests} 个测试失败，请检查相关功能和配置")
            return False
            
    except Exception as e:
        print(f"\n❌ 集成测试验收执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_e2e_results(details):
    """分析端到端测试结果"""
    print("  🔄 端到端数据流分析:")
    if 'e2e_results' in details:
        e2e_results = details['e2e_results']
        
        # 数据同步分析
        if 'data_sync' in e2e_results:
            sync_result = e2e_results['data_sync']
            print(f"    📥 数据同步: {sync_result.get('synced_stocks_count', 0)} 只股票")
            print(f"    ⏱️  同步速度: {sync_result.get('sync_rate_stocks_per_second', 0):.2f} 股票/秒")
        
        # 因子计算分析
        if 'factor_calculation' in e2e_results:
            factor_result = e2e_results['factor_calculation']
            print(f"    🧮 因子计算: {factor_result.get('calculated_factors_count', 0)} 个因子")
            print(f"    ⏱️  计算速度: {factor_result.get('calculation_rate_factors_per_second', 0):.2f} 因子/秒")
        
        # 模型训练分析
        if 'model_training' in e2e_results:
            training_result = e2e_results['model_training']
            print(f"    🤖 模型训练: {training_result.get('trained_models_count', 0)} 个模型")
            print(f"    📊 平均准确率: {training_result.get('average_accuracy', 0):.3f}")
        
        # 预测分析
        if 'prediction' in e2e_results:
            prediction_result = e2e_results['prediction']
            print(f"    🔮 预测生成: {prediction_result.get('predictions_count', 0)} 个预测")
            print(f"    🎯 平均置信度: {prediction_result.get('average_confidence', 0):.3f}")

def analyze_concurrent_results(details):
    """分析并发测试结果"""
    print("  👥 并发测试分析:")
    if 'concurrent_results' in details:
        concurrent_results = details['concurrent_results']
        
        # 并发操作分析
        if 'concurrent_operations' in concurrent_results:
            ops_result = concurrent_results['concurrent_operations']
            print(f"    🔄 并发操作: {ops_result.get('total_operations', 0)} 次操作")
            print(f"    ✅ 成功率: {(1 - ops_result.get('error_rate', 0)) * 100:.1f}%")
            print(f"    ⚡ 操作速度: {ops_result.get('operations_per_second', 0):.2f} 操作/秒")

def analyze_fault_recovery_results(details):
    """分析故障恢复测试结果"""
    print("  🛠️ 故障恢复分析:")
    if 'fault_results' in details:
        fault_results = details['fault_results']
        
        # 数据库故障恢复
        if 'database_fault' in fault_results:
            db_result = fault_results['database_fault']
            print(f"    💾 数据库故障恢复: {db_result.get('recovered_scenarios', 0)}/{db_result.get('scenarios_count', 0)} 场景")
            print(f"    ⏱️  平均恢复时间: {db_result.get('average_recovery_time_seconds', 0):.2f} 秒")
        
        # 服务崩溃恢复
        if 'service_crash' in fault_results:
            service_result = fault_results['service_crash']
            print(f"    🔧 服务崩溃恢复: {service_result.get('recovered_services', 0)}/{service_result.get('services_count', 0)} 服务")

def analyze_stability_results(details):
    """分析稳定性测试结果"""
    print("  📈 稳定性测试分析:")
    if 'stability_results' in details:
        stability_results = details['stability_results']
        
        # 内存监控
        if 'memory_monitoring' in stability_results:
            memory_result = stability_results['memory_monitoring']
            print(f"    🧠 内存使用: {memory_result.get('initial_memory_mb', 0):.1f} → {memory_result.get('final_memory_mb', 0):.1f} MB")
            print(f"    📊 内存趋势: {memory_result.get('memory_trend_mb_per_sample', 0):.3f} MB/采样")

def analyze_business_results(details):
    """分析业务场景测试结果"""
    print("  💼 业务场景分析:")
    if 'business_results' in details:
        business_results = details['business_results']
        
        # 量化工作流
        if 'quant_workflow' in business_results:
            workflow_result = business_results['quant_workflow']
            print(f"    📊 量化工作流: {workflow_result.get('successful_steps', 0)}/{len(workflow_result.get('workflow_steps', []))} 步骤成功")
            print(f"    ⏱️  总耗时: {workflow_result.get('total_duration_seconds', 0):.2f} 秒")

def save_test_results(test_results, summary):
    """保存测试结果到文件"""
    try:
        # 创建测试报告目录
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # 准备测试结果数据
        results_data = {
            'test_type': 'integration_acceptance',
            'summary': summary,
            'test_results': []
        }
        
        for result in test_results:
            results_data['test_results'].append({
                'phase': result.phase,
                'test_name': result.test_name,
                'status': result.status,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'details': result.details
            })
        
        # 保存到JSON文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"integration_acceptance_test_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 测试报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 保存测试报告失败: {e}")

def check_prerequisites():
    """检查测试前提条件"""
    print("🔍 检查测试前提条件...")
    
    issues = []
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        issues.append("Python版本需要3.8或更高")
    
    # 检查必要的目录结构
    required_dirs = [
        'src/acceptance/phases',
        'src/data',
        'src/compute',
        'src/ai'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"缺少必要目录: {dir_path}")
    
    # 检查数据库连接
    try:
        from src.utils.db import get_db_engine
        engine = get_db_engine()
        if engine is None:
            issues.append("数据库连接失败")
        else:
            print("✅ 数据库连接正常")
    except Exception as e:
        issues.append(f"数据库连接检查失败: {e}")
    
    if issues:
        print("❌ 发现以下问题:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("✅ 前提条件检查通过")
        return True

if __name__ == "__main__":
    print("StockSchool 阶段九：集成测试验收实现")
    print("=" * 50)
    
    # 检查前提条件
    if not check_prerequisites():
        print("\n❌ 前提条件检查失败，请解决上述问题后重试")
        sys.exit(1)
    
    print()
    
    # 运行测试
    success = run_integration_acceptance_tests()
    
    # 退出码
    sys.exit(0 if success else 1)