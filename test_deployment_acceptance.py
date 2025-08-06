#!/usr/bin/env python3
"""
部署验收测试运行脚本
测试Docker容器化、CI/CD集成、生产环境部署等功能
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
    from src.acceptance.phases.deployment import DeploymentPhase
    from src.acceptance.core.models import TestResult, TestStatus
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，并且所有依赖都已安装")
    sys.exit(1)

def run_deployment_acceptance_test():
    """运行部署验收测试"""
    print("=" * 80)
    print("StockSchool 部署验收测试")
    print("=" * 80)
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 测试配置
    test_config = {
        'test_timeout': 300,  # 5分钟超时
        'docker_enabled': True,
        'ci_cd_enabled': True,
        'production_checks': True,
        'multi_environment': True
    }
    
    try:
        # 创建部署测试阶段
        deployment_phase = DeploymentPhase("deployment_acceptance", test_config)
        
        print("🚀 开始执行部署验收测试...")
        print()
        
        # 执行测试
        start_time = time.time()
        test_results = deployment_phase._run_tests()
        end_time = time.time()
        
        # 统计测试结果
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in test_results if r.status == TestStatus.FAILED])
        skipped_tests = len([r for r in test_results if r.status == TestStatus.SKIPPED])
        
        # 显示测试结果
        print("\n" + "=" * 80)
        print("部署验收测试结果汇总")
        print("=" * 80)
        
        for result in test_results:
            status_icon = "✅" if result.status == TestStatus.PASSED else "❌" if result.status == TestStatus.FAILED else "⏭️"
            print(f"{status_icon} {result.test_name}: {result.status.value}")
            if result.error_message:
                print(f"   错误信息: {result.error_message}")
            print(f"   执行时间: {result.execution_time:.2f}秒")
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
            print("\n🎉 所有部署验收测试通过！")
            return True
        else:
            print(f"\n⚠️ 有 {failed_tests} 个测试失败，请检查相关配置和环境")
            return False
            
    except Exception as e:
        print(f"\n❌ 部署验收测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_test_results(test_results, summary):
    """保存测试结果到文件"""
    try:
        # 创建测试报告目录
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # 准备测试结果数据
        results_data = {
            'test_type': 'deployment_acceptance',
            'summary': summary,
            'test_results': []
        }
        
        for result in test_results:
            results_data['test_results'].append({
                'phase': result.phase,
                'test_name': result.test_name,
                'status': result.status.value,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'details': result.details
            })
        
        # 保存到JSON文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"deployment_acceptance_test_{timestamp}.json"
        
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
        'src/acceptance/core',
        'config'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"缺少必要目录: {dir_path}")
    
    # 检查配置文件
    config_files = [
        '.env',
        'docker-compose.yml',
        'Dockerfile'
    ]
    
    missing_configs = []
    for config_file in config_files:
        if not os.path.exists(config_file):
            missing_configs.append(config_file)
    
    if missing_configs:
        print(f"⚠️ 缺少配置文件: {', '.join(missing_configs)}")
        print("   这可能影响某些测试的执行，但不会阻止测试运行")
    
    if issues:
        print("❌ 发现以下问题:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("✅ 前提条件检查通过")
        return True

if __name__ == "__main__":
    print("StockSchool 部署验收测试工具")
    print("=" * 50)
    
    # 检查前提条件
    if not check_prerequisites():
        print("\n❌ 前提条件检查失败，请解决上述问题后重试")
        sys.exit(1)
    
    print()
    
    # 运行测试
    success = run_deployment_acceptance_test()
    
    # 退出码
    sys.exit(0 if success else 1)