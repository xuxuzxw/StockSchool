"""
数据服务验收阶段测试脚本
用于验证数据服务验收阶段是否正常工作
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.acceptance.phases.data_service import DataServicePhase
from src.acceptance.core.models import TestStatus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_service_phase():
    """测试数据服务验收阶段"""
    logger.info("=== 开始测试数据服务验收阶段 ===")
    
    try:
        # 创建数据服务验收阶段实例
        config = {
            'config_file': '.env.acceptance',
            'tushare_token': os.getenv('TUSHARE_TOKEN'),
            'test_timeout': 300,
            'max_concurrent_tests': 3
        }
        
        data_service_phase = DataServicePhase(
            phase_name="数据服务验收测试",
            config=config
        )
        
        logger.info("数据服务验收阶段实例创建成功")
        
        # 执行验收测试
        logger.info("开始执行数据服务验收测试...")
        start_time = datetime.now()
        
        results = data_service_phase.execute()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"数据服务验收测试完成，耗时: {execution_time:.2f}秒")
        
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
                    if key in ['sync_status', 'total_records', 'data_quality_score', 'overall_healthy', 'validation_status']:
                        key_details[key] = value
                if key_details:
                    logger.info(f"    详情: {key_details}")
        
        # 清理资源
        data_service_phase._cleanup_resources()
        
        return failed_tests == 0
        
    except Exception as e:
        logger.error(f"数据服务验收阶段测试失败: {e}", exc_info=True)
        return False

def main():
    """主函数"""
    logger.info("StockSchool 数据服务验收阶段测试")
    
    # 检查必要的环境变量
    if not os.getenv('TUSHARE_TOKEN'):
        logger.error("❌ 未设置TUSHARE_TOKEN环境变量，跳过数据服务测试")
        logger.info("请设置TUSHARE_TOKEN环境变量后重新运行测试")
        sys.exit(1)
    
    success = test_data_service_phase()
    
    if success:
        logger.info("🎉 数据服务验收阶段测试成功！")
        sys.exit(0)
    else:
        logger.error("💥 数据服务验收阶段测试失败！")
        sys.exit(1)

if __name__ == '__main__':
    main()