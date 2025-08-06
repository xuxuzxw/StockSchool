#!/usr/bin/env python3
"""
StockSchool 验收测试执行脚本
用于执行完整的系统验收测试
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.acceptance.orchestrator import AcceptanceTestOrchestrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='StockSchool 验收测试执行工具')
    parser.add_argument('--config', '-c', default='.env.acceptance', 
                       help='配置文件路径 (默认: .env.acceptance)')
    parser.add_argument('--phases', nargs='+', 
                       help='指定要执行的测试阶段 (如: 基础设施验收 数据服务验收)')
    parser.add_argument('--skip-config-validation', action='store_true',
                       help='跳过配置验证（测试模式）')
    parser.add_argument('--output-dir', default='reports',
                       help='报告输出目录 (默认: reports)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式，只显示关键信息')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细模式，显示所有日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== StockSchool 验收测试开始 ===")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {args.output_dir}")
    
    if args.phases:
        logger.info(f"指定测试阶段: {', '.join(args.phases)}")
    else:
        logger.info("执行所有可用的测试阶段")
    
    try:
        # 检查配置文件
        if not os.path.exists(args.config):
            logger.warning(f"配置文件 {args.config} 不存在，将使用默认配置")
        
        # 创建输出目录
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # 创建验收测试编排器
        logger.info("初始化验收测试编排器...")
        orchestrator = AcceptanceTestOrchestrator(
            config_file=args.config,
            skip_config_validation=args.skip_config_validation
        )
        
        logger.info(f"编排器初始化成功，会话ID: {orchestrator.session_id}")
        logger.info(f"可用测试阶段数: {len(orchestrator.test_phases)}")
        
        # 执行验收测试
        logger.info("开始执行验收测试...")
        start_time = datetime.now()
        
        report = orchestrator.run_acceptance_tests(selected_phases=args.phases)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # 显示测试结果摘要
        logger.info("=== 验收测试结果摘要 ===")
        logger.info(f"会话ID: {report.test_session_id}")
        logger.info(f"执行时间: {total_duration:.2f} 秒")
        logger.info(f"总测试数: {report.total_tests}")
        logger.info(f"通过测试: {report.passed_tests}")
        logger.info(f"失败测试: {report.failed_tests}")
        logger.info(f"跳过测试: {report.skipped_tests}")
        
        success_rate = (report.passed_tests / report.total_tests * 100) if report.total_tests > 0 else 0
        logger.info(f"成功率: {success_rate:.1f}%")
        
        overall_result = "✅ 通过" if report.overall_result else "❌ 失败"
        logger.info(f"整体结果: {overall_result}")
        
        # 显示阶段结果
        if not args.quiet:
            logger.info("\n=== 各阶段测试结果 ===")
            phases = {}
            for result in report.phase_results:
                if result.phase not in phases:
                    phases[result.phase] = {'passed': 0, 'failed': 0, 'total': 0}
                phases[result.phase]['total'] += 1
                if result.status.value == 'passed':
                    phases[result.phase]['passed'] += 1
                else:
                    phases[result.phase]['failed'] += 1
            
            for phase_name, stats in phases.items():
                status_icon = "✅" if stats['failed'] == 0 else "❌"
                logger.info(f"{status_icon} {phase_name}: {stats['passed']}/{stats['total']} 通过")
        
        # 显示性能指标
        if report.performance_metrics and not args.quiet:
            logger.info("\n=== 性能指标 ===")
            for metric_name, metric_data in report.performance_metrics.items():
                if isinstance(metric_data, dict) and 'average' in metric_data:
                    logger.info(f"{metric_name}: 平均 {metric_data['average']:.3f}s")
                else:
                    logger.info(f"{metric_name}: {metric_data}")
        
        # 显示改进建议
        if report.recommendations:
            logger.info("\n=== 改进建议 ===")
            for i, recommendation in enumerate(report.recommendations, 1):
                logger.info(f"{i}. {recommendation}")
        
        # 显示报告文件位置
        logger.info(f"\n=== 详细报告 ===")
        logger.info(f"HTML报告: reports/acceptance_report_{report.test_session_id}.html")
        logger.info(f"JSON报告: reports/acceptance_report_{report.test_session_id}.json")
        logger.info(f"Markdown报告: reports/acceptance_report_{report.test_session_id}.md")
        
        # 清理资源
        orchestrator.cleanup()
        
        # 设置退出码
        if report.overall_result:
            logger.info("🎉 验收测试全部通过！")
            sys.exit(0)
        else:
            logger.error("💥 验收测试存在失败项目，请查看详细报告")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("用户中断了验收测试")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"验收测试执行失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()