"""
重构后的数据服务验收测试使用示例
"""

import os
from typing import Dict, Any

from data_service import DataServicePhase
from data_service_constants import DataServiceConstants


def run_data_service_acceptance_test():
    """运行数据服务验收测试的示例"""
    
    # 配置测试参数
    config = {
        'tushare_token': os.getenv('TUSHARE_TOKEN'),
        'database_url': os.getenv('DATABASE_URL'),
        'log_level': 'INFO'
    }
    
    # 创建数据服务验收测试实例
    data_service_phase = DataServicePhase(
        phase_name="data_service_acceptance",
        config=config
    )
    
    try:
        # 执行验收测试
        print("🚀 开始执行数据服务验收测试...")
        test_results = data_service_phase.run()
        
        # 分析测试结果
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.status.value == 'PASSED')
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 测试结果统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过测试: {passed_tests}")
        print(f"   失败测试: {failed_tests}")
        print(f"   成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        # 显示详细结果
        print(f"\n📋 详细测试结果:")
        for result in test_results:
            status_emoji = "✅" if result.status.value == 'PASSED' else "❌"
            print(f"   {status_emoji} {result.test_name}: {result.status.value}")
            if result.error_message:
                print(f"      错误: {result.error_message}")
            print(f"      执行时间: {result.execution_time:.2f}秒")
        
        return test_results
        
    except Exception as e:
        print(f"❌ 验收测试执行失败: {e}")
        return None
    
    finally:
        # 清理资源
        data_service_phase._cleanup_resources()


def demonstrate_validator_usage():
    """演示验证器使用方法"""
    from validators import ValidatorFactory
    import pandas as pd
    
    print("\n🔍 验证器使用示例:")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'],
        'symbol': ['000001', '000002', '600000'],
        'name': ['平安银行', '万科A', '浦发银行'],
        'market': ['主板', '主板', '主板'],
        'list_status': ['L', 'L', 'L']
    })
    
    # 使用股票基础信息验证器
    validator = ValidatorFactory.create_validator('stock_basic')
    result = validator.validate(test_data)
    
    print(f"   验证结果: {'通过' if result.is_valid else '失败'}")
    print(f"   质量评分: {result.score:.2f}")
    print(f"   验证消息: {result.message}")
    
    if result.issues:
        print(f"   发现问题: {'; '.join(result.issues)}")


def demonstrate_constants_usage():
    """演示常量使用方法"""
    print("\n📝 常量配置示例:")
    print(f"   测试股票代码: {DataServiceConstants.TEST_STOCK_CODE}")
    print(f"   数据质量阈值: {DataServiceConstants.MIN_QUALITY_SCORE}")
    print(f"   必需数据库表: {DataServiceConstants.REQUIRED_TABLES}")
    print(f"   股票基础信息必需列: {DataServiceConstants.STOCK_BASIC_REQUIRED_COLUMNS}")


if __name__ == "__main__":
    print("=" * 60)
    print("🏗️  数据服务验收测试重构示例")
    print("=" * 60)
    
    # 演示常量使用
    demonstrate_constants_usage()
    
    # 演示验证器使用
    demonstrate_validator_usage()
    
    # 运行完整的验收测试（需要配置环境变量）
    if os.getenv('TUSHARE_TOKEN'):
        run_data_service_acceptance_test()
    else:
        print("\n⚠️  未配置TUSHARE_TOKEN环境变量，跳过完整测试")
        print("   请设置环境变量后重新运行:")
        print("   export TUSHARE_TOKEN=your_token_here")
    
    print("\n✨ 示例运行完成!")