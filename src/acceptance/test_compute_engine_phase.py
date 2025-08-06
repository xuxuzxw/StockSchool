"""
计算引擎验收阶段测试脚本
用于验证计算引擎验收阶段是否正常工作
"""
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.acceptance.phases.compute_engine import ComputeEnginePhase
from src.acceptance.core.models import TestStatus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_compute_engine_phase():
    """测试计算引擎验收阶段"""
    logger.info("=== 开始测试计算引擎验收阶段 ===")
    
    try:
        # 创建计算引擎验收阶段实例
        config = {
            'config_file': '.env.acceptance',
            'test_timeout': 300,
            'max_concurrent_tests': 3
        }
        
        compute_engine_phase = ComputeEnginePhase(
            phase_name="计算引擎验收测试",
            config=config
        )
        
        logger.info("计算引擎验收阶段实例创建成功")
        
        # 执行验收测试
        logger.info("开始执行计算引擎验收测试...")
        start_time = datetime.now()
        
        results = compute_engine_phase.execute()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"计算引擎验收测试完成，耗时: {execution_time:.2f}秒")
        
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
                    if key in ['calculation_status', 'indicators_count', 'factors_categories', 'validation_score', 'performance_score']:
                        key_details[key] = value
                if key_details:
                    logger.info(f"    详情: {key_details}")
        
        # 清理资源
        compute_engine_phase._cleanup_resources()
        
        return failed_tests == 0
        
    except Exception as e:
        logger.error(f"计算引擎验收阶段测试失败: {e}", exc_info=True)
        return False


def main():
    """主函数"""
    logger.info("StockSchool 计算引擎验收阶段测试")
    
    success = test_compute_engine_phase()
    
    if success:
        logger.info("🎉 计算引擎验收阶段测试成功！")
        sys.exit(0)
    else:
        logger.error("💥 计算引擎验收阶段测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()