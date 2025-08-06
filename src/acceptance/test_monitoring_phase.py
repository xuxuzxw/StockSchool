"""
监控告警系统验收阶段测试脚本
用于验证监控告警系统验收阶段是否正常工作
"""
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.acceptance.phases.monitoring import MonitoringPhase
from src.acceptance.core.models import TestStatus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_monitoring_phase():
    """测试监控告警系统验收阶段"""
    logger.info("=== 开始测试监控告警系统验收阶段 ===")
    
    try:
        # 创建监控告警系统验收阶段实例
        config = {
            'config_file': '.env.acceptance',
            'test_timeout': 300,
            'max_concurrent_tests': 3
        }
        
        monitoring_phase = MonitoringPhase(
            phase_name="监控告警系统验收测试",
            config=config
        )
        
        logger.info("监控告警系统验收阶段实例创建成功")
        
        # 执行验收测试
        logger.info("开始执行监控告警系统验收测试...")
        start_time = datetime.now()
        
        results = monitoring_phase.execute()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"监控告警系统验收测试完成，耗时: {execution_time:.2f}秒")
        
        # 分析结果
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        
        logger.info(f"测试结果统计:")
        logger.info(f"  总测试数: {total_tests}")
        logger.info(f"  通过测试: {passed_tests}")
        logger.info(f"  失败测试: {failed_tests}")
        logger.info(f"  跳过测试: {skipped_tests}")
        logger.info(f"  成功率: {passed_tests/total_tests:.1%}" if total_tests > 0 else "  成功率: N/A")
        
        # 打印详细结果
        logger.info("\n=== 详细测试结果 ===")
        for result in results:
            status_icon = {
                TestStatus.PASSED: "✅",
                TestStatus.FAILED: "❌", 
                TestStatus.SKIPPED: "⏭️"
            }.get(result.status, "❓")
            
            logger.info(f"{status_icon} {result.test_name}: {result.status.value} ({result.execution_time:.3f}s)")
            
            if result.error_message:
                logger.info(f"    错误: {result.error_message}")
            
            if result.details:
                # 只显示关键信息，避免日志过长
                key_details = {}
                for key, value in result.details.items():
                    if key.endswith('_score') or key.endswith('_status') or key.endswith('_working'):
                        key_details[key] = value
                
                if key_details:
                    logger.info(f"    关键指标: {key_details}")
        
        # 验证关键测试项
        critical_tests = [
            'system_metrics_collection_test',
            'monitoring_data_storage_test', 
            'alert_rules_trigger_test',
            'alert_notification_test'
        ]
        
        critical_passed = 0
        for result in results:
            if result.test_name in critical_tests and result.status == TestStatus.PASSED:
                critical_passed += 1
        
        logger.info(f"\n=== 关键测试项验证 ===")
        logger.info(f"关键测试通过: {critical_passed}/{len(critical_tests)}")
        
        if critical_passed == len(critical_tests):
            logger.info("🎉 所有关键监控告警系统测试项均通过！")
        else:
            logger.warning("⚠️ 部分关键监控告警系统测试项未通过")
        
        # 总体评估
        overall_success = passed_tests >= total_tests * 0.8  # 80%通过率
        
        if overall_success:
            logger.info("🎉 监控告警系统验收阶段测试总体成功！")
        else:
            logger.warning("⚠️ 监控告警系统验收阶段测试需要改进")
        
        return results
        
    except Exception as e:
        logger.error(f"监控告警系统验收阶段测试失败: {e}")
        raise


def main():
    """主函数"""
    try:
        results = test_monitoring_phase()
        
        # 返回适当的退出码
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        
        if passed_tests >= total_tests * 0.8:  # 80%通过率
            sys.exit(0)  # 成功
        else:
            sys.exit(1)  # 失败
            
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        sys.exit(2)  # 错误


if __name__ == "__main__":
    main()