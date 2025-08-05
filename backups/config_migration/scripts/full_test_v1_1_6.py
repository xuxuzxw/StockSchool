#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool v1.1.6 全流程实测脚本

该脚本执行完整的数据同步、因子计算、AI模型训练和预测流程
作者: StockSchool Team
版本: v1.1.6
创建时间: 2024-01-16
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import ConfigLoader
from src.compute.tasks import (
    sync_daily_data,
    calculate_daily_factors,
    train_ai_model,
    run_prediction,
    batch_prediction,
    weekly_quality_check
)
from celery import group, chain, chord
from src.compute.tasks import app as celery_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/full_test_v1_1_6.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullTestOrchestrator:
    """
    v1.1.6全流程测试编排器
    """
    
    def __init__(self):
        """初始化编排器"""
        self.config = ConfigLoader()
        self.test_config = self.config.get('full_test_config', {})
        self.start_date = self.test_config.get('start_date', '2024-01-01')
        self.end_date = self.test_config.get('end_date', '2024-01-15')
        self.stock_pool = self.test_config.get('stock_pool', [
            '000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'
        ])
        
        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        
        logger.info(f"初始化全流程测试: {self.start_date} 到 {self.end_date}")
        logger.info(f"测试股票池: {self.stock_pool}")
    
    def check_celery_status(self) -> bool:
        """
        检查Celery服务状态
        
        Returns:
            bool: Celery是否正常运行
        """
        try:
            # 检查Celery worker状态
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            
            if not stats:
                logger.error("没有发现活跃的Celery worker")
                return False
            
            logger.info(f"发现 {len(stats)} 个活跃的Celery worker")
            return True
            
        except Exception as e:
            logger.error(f"检查Celery状态失败: {e}")
            return False
    
    def execute_data_sync_phase(self) -> Dict[str, Any]:
        """
        执行数据同步阶段
        
        Returns:
            Dict[str, Any]: 数据同步结果
        """
        logger.info("=== 开始数据同步阶段 ===")
        
        try:
            # 创建数据同步任务组
            sync_tasks = []
            
            # 为每只股票创建同步任务
            for stock_code in self.stock_pool:
                task = sync_daily_data.s(
                    ts_code=stock_code,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                sync_tasks.append(task)
            
            # 并行执行同步任务
            job = group(sync_tasks)
            result = job.apply_async()
            
            # 等待所有任务完成
            results = result.get(timeout=1800)  # 30分钟超时
            
            # 统计结果
            success_count = sum(1 for r in results if r.get('status') == 'success')
            total_count = len(results)
            
            logger.info(f"数据同步完成: {success_count}/{total_count} 成功")
            
            return {
                'phase': 'data_sync',
                'status': 'success' if success_count == total_count else 'partial',
                'success_count': success_count,
                'total_count': total_count,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"数据同步阶段失败: {e}")
            return {
                'phase': 'data_sync',
                'status': 'failed',
                'error': str(e)
            }
    
    def execute_factor_calculation_phase(self) -> Dict[str, Any]:
        """
        执行因子计算阶段
        
        Returns:
            Dict[str, Any]: 因子计算结果
        """
        logger.info("=== 开始因子计算阶段 ===")
        
        try:
            # 创建因子计算任务组
            factor_tasks = []
            
            # 为每只股票创建因子计算任务
            for stock_code in self.stock_pool:
                task = calculate_daily_factors.s(
                    ts_code=stock_code,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                factor_tasks.append(task)
            
            # 并行执行因子计算任务
            job = group(factor_tasks)
            result = job.apply_async()
            
            # 等待所有任务完成
            results = result.get(timeout=2400)  # 40分钟超时
            
            # 统计结果
            success_count = sum(1 for r in results if r.get('status') == 'success')
            total_count = len(results)
            
            logger.info(f"因子计算完成: {success_count}/{total_count} 成功")
            
            return {
                'phase': 'factor_calculation',
                'status': 'success' if success_count == total_count else 'partial',
                'success_count': success_count,
                'total_count': total_count,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"因子计算阶段失败: {e}")
            return {
                'phase': 'factor_calculation',
                'status': 'failed',
                'error': str(e)
            }
    
    def execute_ai_training_phase(self) -> Dict[str, Any]:
        """
        执行AI模型训练阶段
        
        Returns:
            Dict[str, Any]: AI训练结果
        """
        logger.info("=== 开始AI模型训练阶段 ===")
        
        try:
            # 执行AI模型训练任务
            task = train_ai_model.s(
                start_date=self.start_date,
                end_date=self.end_date,
                stock_pool=self.stock_pool,
                model_type='lightgbm'
            )
            
            result = task.apply_async()
            training_result = result.get(timeout=3600)  # 60分钟超时
            
            logger.info(f"AI模型训练完成: {training_result.get('status')}")
            
            return {
                'phase': 'ai_training',
                'status': training_result.get('status', 'failed'),
                'result': training_result
            }
            
        except Exception as e:
            logger.error(f"AI模型训练阶段失败: {e}")
            return {
                'phase': 'ai_training',
                'status': 'failed',
                'error': str(e)
            }
    
    def execute_prediction_phase(self, model_path: str = None) -> Dict[str, Any]:
        """
        执行AI预测阶段
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        logger.info("=== 开始AI预测阶段 ===")
        
        try:
            # 执行批量预测任务
            task = batch_prediction.s(
                stock_codes=self.stock_pool,
                start_date=self.start_date,
                end_date=self.end_date,
                model_path=model_path
            )
            
            result = task.apply_async()
            prediction_result = result.get(timeout=1800)  # 30分钟超时
            
            logger.info(f"AI预测完成: {prediction_result.get('status')}")
            
            return {
                'phase': 'ai_prediction',
                'status': prediction_result.get('status', 'failed'),
                'result': prediction_result
            }
            
        except Exception as e:
            logger.error(f"AI预测阶段失败: {e}")
            return {
                'phase': 'ai_prediction',
                'status': 'failed',
                'error': str(e)
            }
    
    def execute_quality_check_phase(self) -> Dict[str, Any]:
        """
        执行数据质量检查阶段
        
        Returns:
            Dict[str, Any]: 质量检查结果
        """
        logger.info("=== 开始数据质量检查阶段 ===")
        
        try:
            # 执行质量检查任务
            task = weekly_quality_check.s()
            result = task.apply_async()
            check_result = result.get(timeout=600)  # 10分钟超时
            
            logger.info(f"数据质量检查完成: {check_result.get('status')}")
            
            return {
                'phase': 'quality_check',
                'status': check_result.get('status', 'failed'),
                'result': check_result
            }
            
        except Exception as e:
            logger.error(f"数据质量检查阶段失败: {e}")
            return {
                'phase': 'quality_check',
                'status': 'failed',
                'error': str(e)
            }
    
    def run_full_test(self) -> Dict[str, Any]:
        """
        运行完整测试流程
        
        Returns:
            Dict[str, Any]: 完整测试结果
        """
        logger.info("开始执行StockSchool v1.1.6全流程实测")
        start_time = datetime.now()
        
        # 检查Celery状态
        if not self.check_celery_status():
            return {
                'status': 'failed',
                'error': 'Celery服务未运行',
                'duration': 0
            }
        
        results = {
            'start_time': start_time.isoformat(),
            'test_config': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'stock_pool': self.stock_pool
            },
            'phases': []
        }
        
        try:
            # 阶段1: 数据同步
            sync_result = self.execute_data_sync_phase()
            results['phases'].append(sync_result)
            
            if sync_result['status'] == 'failed':
                logger.error("数据同步失败，终止测试")
                return self._finalize_results(results, 'failed')
            
            # 阶段2: 因子计算
            factor_result = self.execute_factor_calculation_phase()
            results['phases'].append(factor_result)
            
            if factor_result['status'] == 'failed':
                logger.error("因子计算失败，终止测试")
                return self._finalize_results(results, 'failed')
            
            # 阶段3: AI模型训练
            training_result = self.execute_ai_training_phase()
            results['phases'].append(training_result)
            
            model_path = None
            if training_result['status'] == 'success':
                model_path = training_result['result'].get('model_path')
            
            # 阶段4: AI预测（即使训练失败也尝试预测）
            prediction_result = self.execute_prediction_phase(model_path)
            results['phases'].append(prediction_result)
            
            # 阶段5: 数据质量检查
            quality_result = self.execute_quality_check_phase()
            results['phases'].append(quality_result)
            
            # 判断整体状态
            failed_phases = [p for p in results['phases'] if p['status'] == 'failed']
            if failed_phases:
                overall_status = 'partial'
            else:
                overall_status = 'success'
            
            return self._finalize_results(results, overall_status)
            
        except Exception as e:
            logger.error(f"全流程测试异常: {e}")
            return self._finalize_results(results, 'failed', str(e))
    
    def _finalize_results(self, results: Dict[str, Any], status: str, 
                         error: str = None) -> Dict[str, Any]:
        """
        完成结果统计
        
        Args:
            results: 结果字典
            status: 整体状态
            error: 错误信息
            
        Returns:
            Dict[str, Any]: 最终结果
        """
        end_time = datetime.now()
        start_time = datetime.fromisoformat(results['start_time'])
        duration = (end_time - start_time).total_seconds()
        
        results.update({
            'end_time': end_time.isoformat(),
            'duration': duration,
            'status': status
        })
        
        if error:
            results['error'] = error
        
        # 统计各阶段状态
        phase_summary = {}
        for phase in results.get('phases', []):
            phase_name = phase.get('phase', 'unknown')
            phase_status = phase.get('status', 'unknown')
            phase_summary[phase_name] = phase_status
        
        results['phase_summary'] = phase_summary
        
        logger.info(f"全流程测试完成: {status}, 耗时: {duration:.2f}秒")
        logger.info(f"各阶段状态: {phase_summary}")
        
        return results

def main():
    """
    主函数
    """
    print("StockSchool v1.1.6 全流程实测")
    print("=" * 50)
    
    # 创建编排器
    orchestrator = FullTestOrchestrator()
    
    # 运行全流程测试
    results = orchestrator.run_full_test()
    
    # 输出结果
    print("\n测试结果:")
    print(f"状态: {results['status']}")
    print(f"耗时: {results['duration']:.2f}秒")
    
    if 'phase_summary' in results:
        print("\n各阶段状态:")
        for phase, status in results['phase_summary'].items():
            print(f"  {phase}: {status}")
    
    if 'error' in results:
        print(f"\n错误: {results['error']}")
    
    # 保存详细结果到文件
    import json
    with open('logs/full_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n详细结果已保存到: logs/full_test_results.json")
    
    return results['status'] == 'success'

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)