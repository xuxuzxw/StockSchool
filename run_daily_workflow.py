#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool v1.1.6 全流程日度工作流调度脚本

该脚本定义并启动一个完整的、从数据同步到预测的Celery任务链。
作者: StockSchool Team
版本: v1.1.6
创建时间: 2024-01-16
"""

import os
import sys
from celery import chain, group
from datetime import datetime
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.compute.tasks import (
    sync_daily_data,
    sync_stock_data,
    calculate_daily_factors,
    calculate_stock_factors,
    train_ai_model,
    batch_prediction,
    weekly_quality_check
)
from src.utils.config_loader import Config
from loguru import logger

# 配置日志
logger.add(
    "logs/daily_workflow.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)

def setup_logger():
    """设置日志配置"""
    return logger

def run_full_workflow():
    """
    定义并启动一个完整的、从数据同步到预测的Celery任务链。
    """
    logger.info("Starting the full daily workflow...")
    
    try:
        # 加载配置
        config = Config()
        test_config = config.get('full_test_config', {})
        
        start_date = test_config.get('start_date', '20230101')
        end_date = test_config.get('end_date', '20230131')
        stock_pool = test_config.get('stock_pool', [
            '000001.SZ', '600519.SH', '300750.SZ'
        ])
        
        logger.info(f"测试配置: {start_date} 到 {end_date}, 股票池: {stock_pool}")
        
        # 阶段1: 数据同步任务组
        logger.info("创建数据同步任务组...")
        sync_tasks = []
        for stock_code in stock_pool:
            task = sync_stock_data.s(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            sync_tasks.append(task)
        
        # 并行执行数据同步
        sync_group = group(sync_tasks)
        
        # 阶段2: 因子计算任务组
        logger.info("创建因子计算任务组...")
        factor_tasks = []
        for stock_code in stock_pool:
            task = calculate_stock_factors.s(
                ts_code=stock_code,
                days=60  # 使用60天数据计算因子
            )
            factor_tasks.append(task)
        
        # 并行执行因子计算
        factor_group = group(factor_tasks)
        
        # 阶段3: AI模型训练任务
        logger.info("创建AI模型训练任务...")
        training_task = train_ai_model.s(
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool,
            model_type='lightgbm'
        )
        
        # 阶段4: 批量预测任务
        logger.info("创建批量预测任务...")
        prediction_task = batch_prediction.s(
            stock_codes=stock_pool,
            start_date=start_date,
            end_date=end_date
        )
        
        # 阶段5: 数据质量检查任务
        logger.info("创建数据质量检查任务...")
        quality_task = weekly_quality_check.s()
        
        # 定义任务链：上一个任务的成功是下一个任务开始的前提
        # 这确保了数据同步完成后才会开始计算因子，以此类推。
        workflow = chain(
            sync_group,           # 并行数据同步
            factor_group,         # 并行因子计算
            training_task,        # AI模型训练
            prediction_task,      # 批量预测
            quality_task          # 质量检查
        )
        
        # 异步执行任务链
        result = workflow.apply_async()
        logger.info(f"Workflow started with chain ID: {result.id}")
        print(f"Workflow started. Check Celery logs for progress. Chain ID: {result.id}")
        
        # 在实际生产中，你会让它在后台运行。
        # 为了测试，我们可以等待结果：
        print("等待工作流完成...（这可能需要一些时间）")
        try:
            final_result = result.get(timeout=7200)  # 2小时超时
            logger.info("Workflow completed successfully.")
            print("工作流执行完成！")
            print(f"最终结果: {final_result}")
            return True
        except Exception as wait_error:
            logger.error(f"等待工作流完成时出错: {wait_error}")
            print(f"工作流可能仍在后台运行，Chain ID: {result.id}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}", exc_info=True)
        print(f"启动工作流失败: {e}")
        return False

def run_simple_workflow():
    """
    运行简化版工作流（用于快速测试）
    """
    logger.info("Starting the simple workflow...")
    
    try:
        # 加载配置
        config = Config()
        test_config = config.get('full_test_config', {})
        
        start_date = test_config.get('start_date', '20230101')
        end_date = test_config.get('end_date', '20230131')
        stock_pool = test_config.get('stock_pool', ['000001.SZ'])  # 只用一只股票测试
        
        logger.info(f"简化测试配置: {start_date} 到 {end_date}, 股票: {stock_pool[0]}")
        
        # 简化的任务链
        workflow = chain(
            sync_stock_data.s(
                ts_code=stock_pool[0],
                start_date=start_date,
                end_date=end_date
            ),
            calculate_stock_factors.s(
                ts_code=stock_pool[0],
                days=30
            ),
            weekly_quality_check.s()
        )
        
        # 执行简化工作流
        result = workflow.apply_async()
        logger.info(f"Simple workflow started with chain ID: {result.id}")
        print(f"简化工作流已启动，Chain ID: {result.id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start simple workflow: {e}", exc_info=True)
        print(f"启动简化工作流失败: {e}")
        return False

def check_celery_status():
    """
    检查Celery服务状态
    """
    try:
        from src.compute.tasks import app as celery_app
        
        # 检查Celery worker状态
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        if not stats:
            print("❌ 没有发现活跃的Celery worker")
            print("请先启动Celery worker:")
            print("celery -A src.compute.tasks worker -l info -P eventlet")
            return False
        
        print(f"✅ 发现 {len(stats)} 个活跃的Celery worker")
        for worker_name in stats.keys():
            print(f"  - {worker_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查Celery状态失败: {e}")
        return False

if __name__ == '__main__':
    print("StockSchool v1.1.6 全流程日度工作流")
    print("=" * 50)
    
    # 检查Celery状态
    if not check_celery_status():
        sys.exit(1)
    
    # 询问用户选择
    print("\n请选择执行模式:")
    print("1. 完整工作流（包含所有股票和AI训练）")
    print("2. 简化工作流（单只股票，快速测试）")
    print("3. 退出")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        print("\n启动完整工作流...")
        success = run_full_workflow()
    elif choice == '2':
        print("\n启动简化工作流...")
        success = run_simple_workflow()
    elif choice == '3':
        print("退出")
        sys.exit(0)
    else:
        print("无效选择")
        sys.exit(1)
    
    if success:
        print("\n✅ 工作流启动成功")
    else:
        print("\n❌ 工作流启动失败")
        sys.exit(1)