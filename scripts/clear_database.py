#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool 数据库清理脚本

该脚本用于清空核心数据表，确保全流程测试从干净状态开始
作者: StockSchool Team
版本: v1.1.6
创建时间: 2024-01-16
"""

import os
import sys
from typing import List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.db import get_db_engine
from src.utils.config_loader import Config
from sqlalchemy import text
from loguru import logger
import pandas as pd

# 配置日志
logger.add(
    "logs/database_clear.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)

class DatabaseCleaner:
    """
    数据库清理器
    """
    
    def __init__(self):
        """初始化清理器"""
        self.engine = get_db_engine()
        self.config = Config()
        
        # 定义需要清理的核心数据表
        self.core_tables = [
            'stock_daily',           # 股票日线数据
            'stock_daily_basic',     # 股票每日基本面数据
            'financial_reports',     # 财务报告数据
            'factor_library',        # 因子库
            'prediction_results',    # 预测结果
            'ai_predictions',        # AI预测结果
            'data_quality_reports',  # 数据质量报告
            'sync_logs',            # 同步日志
            'calculation_logs'       # 计算日志
        ]
        
        # 定义可选清理的表（用户可选择是否清理）
        self.optional_tables = [
            'stock_basic',          # 股票基本信息
            'trade_calendar',       # 交易日历
            'adj_factor',          # 复权因子
            'index_daily',         # 指数日线数据
            'concept_detail',      # 概念股详情
            'limit_list'           # 涨跌停列表
        ]
    
    def check_table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在
        
        Args:
            table_name: 表名
            
        Returns:
            bool: 表是否存在
        """
        try:
            query = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
            """
            result = pd.read_sql(query, self.engine)
            return result['count'].iloc[0] > 0
        except Exception as e:
            logger.warning(f"检查表 {table_name} 是否存在时出错: {e}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """
        获取表的行数
        
        Args:
            table_name: 表名
            
        Returns:
            int: 表的行数
        """
        try:
            if not self.check_table_exists(table_name):
                return 0
            
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = pd.read_sql(query, self.engine)
            return result['count'].iloc[0]
        except Exception as e:
            logger.warning(f"获取表 {table_name} 行数时出错: {e}")
            return 0
    
    def clear_table(self, table_name: str) -> bool:
        """
        清空指定表
        
        Args:
            table_name: 表名
            
        Returns:
            bool: 是否成功清空
        """
        try:
            if not self.check_table_exists(table_name):
                logger.info(f"表 {table_name} 不存在，跳过清理")
                return True
            
            # 获取清理前的行数
            before_count = self.get_table_row_count(table_name)
            
            if before_count == 0:
                logger.info(f"表 {table_name} 已经为空，跳过清理")
                return True
            
            # 清空表
            with self.engine.connect() as conn:
                conn.execute(text(f"DELETE FROM {table_name}"))
                conn.commit()
            
            # 验证清理结果
            after_count = self.get_table_row_count(table_name)
            
            if after_count == 0:
                logger.info(f"✅ 成功清空表 {table_name}，删除了 {before_count} 行数据")
                return True
            else:
                logger.error(f"❌ 清空表 {table_name} 失败，仍有 {after_count} 行数据")
                return False
                
        except Exception as e:
            logger.error(f"❌ 清空表 {table_name} 时出错: {e}")
            return False
    
    def clear_core_tables(self) -> dict:
        """
        清空核心数据表
        
        Returns:
            dict: 清理结果统计
        """
        logger.info("开始清空核心数据表...")
        
        results = {
            'success_tables': [],
            'failed_tables': [],
            'skipped_tables': []
        }
        
        for table_name in self.core_tables:
            logger.info(f"正在清理表: {table_name}")
            
            if self.clear_table(table_name):
                results['success_tables'].append(table_name)
            else:
                results['failed_tables'].append(table_name)
        
        return results
    
    def clear_optional_tables(self, tables_to_clear: List[str]) -> dict:
        """
        清空可选数据表
        
        Args:
            tables_to_clear: 要清理的表列表
            
        Returns:
            dict: 清理结果统计
        """
        logger.info(f"开始清空可选数据表: {tables_to_clear}")
        
        results = {
            'success_tables': [],
            'failed_tables': [],
            'skipped_tables': []
        }
        
        for table_name in tables_to_clear:
            if table_name not in self.optional_tables:
                logger.warning(f"表 {table_name} 不在可选清理列表中，跳过")
                results['skipped_tables'].append(table_name)
                continue
            
            logger.info(f"正在清理表: {table_name}")
            
            if self.clear_table(table_name):
                results['success_tables'].append(table_name)
            else:
                results['failed_tables'].append(table_name)
        
        return results
    
    def show_database_status(self):
        """
        显示数据库状态
        """
        logger.info("数据库状态概览:")
        print("\n=== 数据库状态概览 ===")
        
        print("\n核心数据表:")
        for table_name in self.core_tables:
            count = self.get_table_row_count(table_name)
            status = "存在" if self.check_table_exists(table_name) else "不存在"
            print(f"  {table_name:<25} | {status:<6} | {count:>10,} 行")
        
        print("\n可选数据表:")
        for table_name in self.optional_tables:
            count = self.get_table_row_count(table_name)
            status = "存在" if self.check_table_exists(table_name) else "不存在"
            print(f"  {table_name:<25} | {status:<6} | {count:>10,} 行")
    
    def backup_important_data(self) -> bool:
        """
        备份重要数据（如股票基本信息、交易日历等）
        
        Returns:
            bool: 备份是否成功
        """
        try:
            logger.info("开始备份重要数据...")
            
            backup_tables = ['stock_basic', 'trade_calendar']
            backup_dir = 'backups'
            os.makedirs(backup_dir, exist_ok=True)
            
            for table_name in backup_tables:
                if self.check_table_exists(table_name):
                    df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
                    backup_file = f"{backup_dir}/{table_name}_backup.csv"
                    df.to_csv(backup_file, index=False)
                    logger.info(f"已备份 {table_name} 到 {backup_file}")
            
            logger.info("重要数据备份完成")
            return True
            
        except Exception as e:
            logger.error(f"备份重要数据失败: {e}")
            return False

def main():
    """
    主函数
    """
    print("StockSchool 数据库清理工具")
    print("=" * 50)
    
    # 创建清理器
    cleaner = DatabaseCleaner()
    
    # 显示当前数据库状态
    cleaner.show_database_status()
    
    print("\n请选择清理模式:")
    print("1. 仅清理核心数据表（推荐用于全流程测试）")
    print("2. 清理核心数据表 + 选择可选表")
    print("3. 仅显示状态，不清理")
    print("4. 备份重要数据")
    print("5. 退出")
    
    choice = input("\n请输入选择 (1/2/3/4/5): ").strip()
    
    if choice == '1':
        # 确认清理
        confirm = input("\n⚠️  确认清理核心数据表？这将删除所有交易数据、因子和预测结果！(y/N): ")
        if confirm.lower() == 'y':
            print("\n开始清理核心数据表...")
            results = cleaner.clear_core_tables()
            
            print("\n清理结果:")
            print(f"✅ 成功清理: {len(results['success_tables'])} 个表")
            if results['success_tables']:
                for table in results['success_tables']:
                    print(f"  - {table}")
            
            if results['failed_tables']:
                print(f"❌ 清理失败: {len(results['failed_tables'])} 个表")
                for table in results['failed_tables']:
                    print(f"  - {table}")
        else:
            print("取消清理")
    
    elif choice == '2':
        print("\n可选清理的表:")
        for i, table in enumerate(cleaner.optional_tables, 1):
            count = cleaner.get_table_row_count(table)
            print(f"  {i}. {table} ({count:,} 行)")
        
        selected = input("\n请输入要清理的表序号（用逗号分隔，如: 1,3,5）: ").strip()
        
        if selected:
            try:
                indices = [int(x.strip()) - 1 for x in selected.split(',')]
                tables_to_clear = [cleaner.optional_tables[i] for i in indices 
                                 if 0 <= i < len(cleaner.optional_tables)]
                
                if tables_to_clear:
                    confirm = input(f"\n⚠️  确认清理这些表？{tables_to_clear} (y/N): ")
                    if confirm.lower() == 'y':
                        # 先清理核心表
                        print("\n清理核心数据表...")
                        core_results = cleaner.clear_core_tables()
                        
                        # 再清理可选表
                        print("\n清理可选数据表...")
                        optional_results = cleaner.clear_optional_tables(tables_to_clear)
                        
                        print("\n清理完成！")
                    else:
                        print("取消清理")
                else:
                    print("没有选择有效的表")
            except ValueError:
                print("输入格式错误")
    
    elif choice == '3':
        print("\n当前数据库状态已显示")
    
    elif choice == '4':
        print("\n开始备份重要数据...")
        if cleaner.backup_important_data():
            print("✅ 备份完成")
        else:
            print("❌ 备份失败")
    
    elif choice == '5':
        print("退出")
    
    else:
        print("无效选择")

if __name__ == '__main__':
    main()