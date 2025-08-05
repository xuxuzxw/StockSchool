#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool 全流程测试脚本
版本: v1.1.6
创建时间: 2024-01-16
作者: StockSchool Team
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config.unified_config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_full_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullPipelineTester:
    """全流程测试器"""

    def __init__(self):
        """初始化测试器"""
        self.config = config
        self.test_config = self.config.get('full_test_config', {})
        self.start_date = self.test_config.get('start_date', '2024-01-01')
        self.end_date = self.test_config.get('end_date', '2024-12-15')
        self.stock_pool = self.test_config.get('stock_pool', [
            '000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'
        ])

        # 创建日志目录
        os.makedirs('logs', exist_ok=True)

        logger.info(f"初始化全流程测试: {self.start_date} 到 {self.end_date}")
        logger.info(f"测试股票池: {self.stock_pool}")

    def test_database_connection(self) -> Dict[str, Any]:
        """测试数据库连接"""
        try:
            from src.database.connection import test_database_connection
            result = test_database_connection()
            return {
                "success": result["success"],
                "message": result["message"],
                "error": result.get("error", "")
            }
        except ImportError as e:
            return {
                "success": False,
                "message": f"数据库连接模块导入失败: {e}",
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"数据库连接测试失败: {e}",
                "error": str(e)
            }

    def test_data_sync(self) -> Dict[str, Any]:
        """测试数据同步功能"""
        try:
            from src.data.sync_manager import DataSyncManager
            
            sync_manager = DataSyncManager()
            result = sync_manager.sync_latest_data(
                symbols=['000001.SZ'],
                start_date='2024-01-01',
                end_date='2024-01-31'
            )
            return {
                "success": result.get("success", False),
                "message": result.get("message", "数据同步测试完成"),
                "error": result.get("error", "")
            }
        except ImportError as e:
            return {
                "success": False,
                "message": f"数据同步模块导入失败: {e}",
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"数据同步测试失败: {e}",
                "error": str(e)
            }

    def test_factor_calculation(self) -> Dict[str, Any]:
        """测试因子计算功能"""
        logger.info("=== 测试因子计算阶段 ===")
        
        try:
            from src.compute.engine_factory import EngineFactory
            
            # 创建技术因子引擎
            engine = EngineFactory.create_technical_engine()
            results = []
            
            for stock_code in self.stock_pool:
                try:
                    # 简化测试，只计算一个因子
                    result = engine.calculate_factors(
                        ts_code=stock_code,
                        factor_names=['MA20'],
                        start_date=self.start_date,
                        end_date=self.end_date
                    )
                    results.append({
                        'ts_code': stock_code,
                        'status': 'success',
                        'factor_count': len(result) if result else 0
                    })
                    logger.info(f"因子计算 {stock_code}: 成功 ({len(result) if result else 0}条数据)")
                except Exception as e:
                    results.append({
                        'ts_code': stock_code,
                        'status': 'failed',
                        'error': str(e)
                    })
                    logger.error(f"因子计算 {stock_code}: 失败 - {e}")
            
            success_count = sum(1 for r in results if r.get('status') == 'success')
            logger.info(f"因子计算完成: {success_count}/{len(results)} 成功")
            
            return {
                'phase': 'factor_calculation',
                'status': 'success' if success_count == len(results) else 'partial',
                'success_count': success_count,
                'total_count': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"因子计算测试失败: {e}")
            return {
                'phase': 'factor_calculation',
                'status': 'failed',
                'error': str(e)
            }

    def run_full_test(self) -> Dict[str, Any]:
        """运行全流程测试"""
        logger.info("开始StockSchool全流程测试...")
        
        start_time = datetime.now()
        
        # 依次执行各阶段测试
        results = {}
        
        # 1. 数据库连接测试
        results['database_connection'] = self.test_database_connection()
        
        # 2. 数据同步测试
        results['data_sync'] = self.test_data_sync()
        
        # 3. 因子计算测试
        results['factor_calculation'] = self.test_factor_calculation()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 汇总结果
        all_success = all(r.get('status') == 'success' for r in results.values())
        
        summary = {
            'overall_status': 'success' if all_success else 'partial',
            'duration_seconds': duration,
            'phases': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"全流程测试完成，用时 {duration:.2f} 秒")
        logger.info(f"总体状态: {'成功' if all_success else '部分成功'}")
        
        return summary

if __name__ == "__main__":
    tester = FullPipelineTester()
    results = tester.run_full_test()
    
    # 保存测试结果
    import json
    with open('logs/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("测试完成，结果已保存到 logs/test_results.json")