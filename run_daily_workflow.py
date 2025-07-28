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
import time # 导入 time 模块
from celery.result import AsyncResult # 导入 AsyncResult
from datetime import datetime
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.compute.tasks import (
    app, # 导入 Celery app 实例
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
        
        print("\n=== 工作流进度 ===")
        start_time = datetime.now()
        timeout_seconds = 7200 # 2小时超时

        # 阶段1: 数据同步
        logger.info("开始执行数据同步任务组...")
        sync_group_result = sync_group.apply_async()
        print(f"数据同步任务组已启动. Group ID: {sync_group_result.id}")
        
        while not sync_group_result.ready():
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time > timeout_seconds:
                logger.error(f"数据同步任务组超时，Group ID: {sync_group_result.id}")
                print("\n❌ 数据同步任务组执行超时！")
                return False

            all_children_ready = True
            for i, child_result in enumerate(sync_group_result.children):
                status = child_result.state
                info = child_result.info
                
                if status == 'PROGRESS' and info:
                    current_step = info.get('current_step', '未知步骤')
                    progress = info.get('progress', 'N/A')
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 步骤: {current_step} | 进度: {progress}")
                elif status == 'PENDING':
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 等待开始...")
                    all_children_ready = False
                elif status == 'STARTED':
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 已启动...")
                    all_children_ready = False
                elif status == 'SUCCESS':
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 完成.")
                elif status == 'FAILURE':
                    logger.error(f"数据同步任务 {i+1} 失败: {child_result.info}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 失败: {child_result.info}")
                    return False # 任何一个子任务失败，整个工作流失败
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1}")
                
                if not child_result.ready():
                    all_children_ready = False

            if all_children_ready and sync_group_result.ready():
                break # 所有子任务和组都已完成
            
            time.sleep(5) # 每5秒检查一次

        if not sync_group_result.successful():
            logger.error("数据同步任务组执行失败.")
            print("\n❌ 数据同步任务组执行失败！")
            return False
        logger.info("数据同步任务组完成.")
        print("\n✅ 数据同步任务组完成！")

        # 阶段2: 因子计算
        logger.info("开始执行因子计算任务组...")
        factor_group_result = factor_group.apply_async()
        print(f"因子计算任务组已启动. Group ID: {factor_group_result.id}")

        while not factor_group_result.ready():
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time > timeout_seconds:
                logger.error(f"因子计算任务组超时，Group ID: {factor_group_result.id}")
                print("\n❌ 因子计算任务组执行超时！")
                return False

            all_children_ready = True
            for i, child_result in enumerate(factor_group_result.children):
                status = child_result.state
                info = child_result.info

                if status == 'PROGRESS' and info:
                    processed_count = info.get('processed_count', 'N/A')
                    total_count = info.get('total_count', 'N/A')
                    progress = info.get('progress', 'N/A')
                    current_stock = info.get('current_stock', 'N/A')
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 已处理: {processed_count}/{total_count} | 进度: {progress} | 当前股票: {current_stock}")
                elif status == 'PENDING':
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 等待开始...")
                    all_children_ready = False
                elif status == 'STARTED':
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 已启动...")
                    all_children_ready = False
                elif status == 'SUCCESS':
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 完成.")
                elif status == 'FAILURE':
                    logger.error(f"因子计算任务 {i+1} 失败: {child_result.info}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1} | 失败: {child_result.info}")
                    return False # 任何一个子任务失败，整个工作流失败
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} | 任务 {i+1}")

                if not child_result.ready():
                    all_children_ready = False
            
            if all_children_ready and factor_group_result.ready():
                break # 所有子任务和组都已完成

            time.sleep(5) # 每5秒检查一次

        if not factor_group_result.successful():
            logger.error("因子计算任务组执行失败.")
            print("\n❌ 因子计算任务组执行失败！")
            return False
        logger.info("因子计算任务组完成.")
        print("\n✅ 因子计算任务组完成！")

        # 阶段3: AI模型训练
        logger.info("开始执行AI模型训练任务...")
        training_result = training_task.apply_async()
        print(f"AI模型训练任务已启动. Task ID: {training_result.id}")
        training_result.get(timeout=timeout_seconds) # 等待任务完成
        if not training_result.successful():
            logger.error("AI模型训练任务执行失败.")
            print("\n❌ AI模型训练任务执行失败！")
            return False
        logger.info("AI模型训练任务完成.")
        print("\n✅ AI模型训练任务完成！")

        # 阶段4: 批量预测
        logger.info("开始执行批量预测任务...")
        prediction_result = prediction_task.apply_async()
        print(f"批量预测任务已启动. Task ID: {prediction_result.id}")
        prediction_result.get(timeout=timeout_seconds) # 等待任务完成
        if not prediction_result.successful():
            logger.error("批量预测任务执行失败.")
            print("\n❌ 批量预测任务执行失败！")
            return False
        logger.info("批量预测任务完成.")
        print("\n✅ 批量预测任务完成！")

        # 阶段5: 数据质量检查
        logger.info("开始执行数据质量检查任务...")
        quality_result = quality_task.apply_async()
        print(f"数据质量检查任务已启动. Task ID: {quality_result.id}")
        quality_result.get(timeout=timeout_seconds) # 等待任务完成
        if not quality_result.successful():
            logger.error("数据质量检查任务执行失败.")
            print("\n❌ 数据质量检查任务执行失败！")
            return False
        logger.info("数据质量检查任务完成.")
        print("\n✅ 数据质量检查任务完成！")

        logger.info("完整工作流执行完成！")
        print("\n✅ 完整工作流执行完成！")
        return True
            
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