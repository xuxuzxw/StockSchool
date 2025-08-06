"""
验收测试编排器测试脚本
用于验证验收测试编排器是否正常工作
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.acceptance.orchestrator import AcceptanceTestOrchestrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_orchestrator():
    """测试验收测试编排器"""
    logger.info("=== 开始测试验收测试编排器 ===")
    
    try:
        # 创建编排器实例
        logger.info("创建验收测试编排器实例...")
        
        orchestrator = AcceptanceTestOrchestrator('.env.acceptance', skip_config_validation=True)
        
        logger.info(f"编排器创建成功，会话ID: {orchestrator.session_id}")
        logger.info(f"初始化了 {len(orchestrator.test_phases)} 个测试阶段")
        
        # 获取会话状态
        status = orchestrator.get_session_status()
        logger.info(f"当前会话状态: {status}")
        
        # 只运行基础设施验收阶段（避免运行所有阶段）
        logger.info("开始执行基础设施验收阶段...")
        start_time = datetime.now()
        
        # 只选择基础设施验收阶段
        report = orchestrator.run_acceptance_tests(selected_phases=["基础设施验收"])
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"验收测试完成，耗时: {execution_time:.2f}秒")
        
        # 分析报告
        logger.info(f"=== 验收测试报告摘要 ===")
        logger.info(f"会话ID: {report.test_session_id}")
        logger.info(f"开始时间: {report.start_time}")
        logger.info(f"结束时间: {report.end_time}")
        logger.info(f"总测试数: {report.total_tests}")
        logger.info(f"通过测试: {report.passed_tests}")
        logger.info(f"失败测试: {report.failed_tests}")
        logger.info(f"跳过测试: {report.skipped_tests}")
        logger.info(f"整体结果: {'✅ 通过' if report.overall_result else '❌ 失败'}")
        
        # 显示性能指标
        if report.performance_metrics:
            logger.info(f"=== 性能指标 ===")
            for metric_name, metric_data in report.performance_metrics.items():
                logger.info(f"{metric_name}: {metric_data}")
        
        # 显示改进建议
        if report.recommendations:
            logger.info(f"=== 改进建议 ===")
            for i, recommendation in enumerate(report.recommendations, 1):
                logger.info(f"{i}. {recommendation}")
        
        # 清理资源
        orchestrator.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"验收测试编排器测试失败: {e}", exc_info=True)
        return False

def main():
    """主函数"""
    logger.info("StockSchool 验收测试编排器测试")
    
    success = test_orchestrator()
    
    if success:
        logger.info("🎉 验收测试编排器测试成功！")
        sys.exit(0)
    else:
        logger.error("💥 验收测试编排器测试失败！")
        sys.exit(1)

if __name__ == '__main__':
    main()