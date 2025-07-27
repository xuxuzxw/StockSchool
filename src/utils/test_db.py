#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据库连接工具
提供独立的测试数据库连接和管理功能

主要功能：
1. 测试数据库连接管理
2. 测试数据准备和清理
3. 测试环境隔离
4. 数据库状态检查

作者: StockSchool Team
创建时间: 2024
"""

import os
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from datetime import datetime, timedelta
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDatabaseManager:
    """测试数据库管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化测试数据库管理器
        
        Args:
            config: 数据库配置，如果为None则使用环境变量
        """
        if config is None:
            config = self._load_test_db_config()
        
        self.config = config
        self.engine = None
        self._connect()
        
        logger.info("测试数据库管理器初始化完成")
    
    def _load_test_db_config(self) -> Dict[str, Any]:
        """从环境变量加载测试数据库配置
        
        Returns:
            数据库配置字典
        """
        return {
            'host': os.getenv('TEST_DB_HOST', 'localhost'),
            'port': int(os.getenv('TEST_DB_PORT', '15433')),
            'database': os.getenv('TEST_POSTGRES_DB', 'stockschool_test'),
            'username': os.getenv('TEST_POSTGRES_USER', 'stockschool_test'),
            'password': os.getenv('TEST_POSTGRES_PASSWORD', 'test123')
        }
    
    def _connect(self) -> None:
        """连接到测试数据库"""
        try:
            connection_string = (
                f"postgresql://{self.config['username']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False
            )
            
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"测试数据库连接成功: {self.config['host']}:{self.config['port']}/{self.config['database']}")
            
        except Exception as e:
            logger.error(f"测试数据库连接失败: {str(e)}")
            raise
    
    def get_engine(self) -> Engine:
        """获取数据库引擎
        
        Returns:
            SQLAlchemy引擎对象
        """
        if self.engine is None:
            self._connect()
        return self.engine
    
    def check_connection(self) -> bool:
        """检查数据库连接状态
        
        Returns:
            连接是否正常
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                return result == 1
        except Exception as e:
            logger.error(f"数据库连接检查失败: {str(e)}")
            return False
    
    def wait_for_database(self, max_attempts: int = 30, delay: int = 2) -> bool:
        """等待数据库就绪
        
        Args:
            max_attempts: 最大尝试次数
            delay: 每次尝试间隔（秒）
            
        Returns:
            数据库是否就绪
        """
        logger.info("等待测试数据库就绪...")
        
        for attempt in range(max_attempts):
            try:
                if self.check_connection():
                    logger.info(f"测试数据库就绪 (尝试 {attempt + 1}/{max_attempts})")
                    return True
            except Exception:
                pass
            
            if attempt < max_attempts - 1:
                logger.info(f"数据库未就绪，等待 {delay} 秒... (尝试 {attempt + 1}/{max_attempts})")
                time.sleep(delay)
        
        logger.error(f"等待数据库超时 ({max_attempts * delay} 秒)")
        return False
    
    def get_table_list(self) -> List[str]:
        """获取数据库中的表列表
        
        Returns:
            表名列表
        """
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            logger.info(f"找到 {len(tables)} 个表: {tables}")
            return tables
        except Exception as e:
            logger.error(f"获取表列表失败: {str(e)}")
            return []
    
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在
        
        Args:
            table_name: 表名
            
        Returns:
            表是否存在
        """
        try:
            inspector = inspect(self.engine)
            return table_name in inspector.get_table_names()
        except Exception as e:
            logger.error(f"检查表存在性失败: {str(e)}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """获取表的行数
        
        Args:
            table_name: 表名
            
        Returns:
            行数
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                return result or 0
        except Exception as e:
            logger.error(f"获取表行数失败: {str(e)}")
            return 0
    
    def clear_table(self, table_name: str) -> bool:
        """清空表数据
        
        Args:
            table_name: 表名
            
        Returns:
            是否成功
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
                conn.commit()
            logger.info(f"表 {table_name} 已清空")
            return True
        except Exception as e:
            logger.error(f"清空表失败: {str(e)}")
            return False
    
    def clear_all_tables(self) -> bool:
        """清空所有表数据
        
        Returns:
            是否成功
        """
        try:
            tables = self.get_table_list()
            for table in tables:
                self.clear_table(table)
            logger.info(f"已清空所有 {len(tables)} 个表")
            return True
        except Exception as e:
            logger.error(f"清空所有表失败: {str(e)}")
            return False
    
    def insert_test_data(self, table_name: str, data: pd.DataFrame) -> bool:
        """插入测试数据
        
        Args:
            table_name: 表名
            data: 测试数据
            
        Returns:
            是否成功
        """
        try:
            data.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            logger.info(f"向表 {table_name} 插入了 {len(data)} 行测试数据")
            return True
        except Exception as e:
            logger.error(f"插入测试数据失败: {str(e)}")
            return False
    
    def create_sample_stock_data(self, 
                               stock_codes: List[str] = None,
                               start_date: str = '20240101',
                               end_date: str = '20240131',
                               base_price: float = 10.0) -> pd.DataFrame:
        """创建样本股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            base_price: 基础价格
            
        Returns:
            样本股票数据
        """
        if stock_codes is None:
            stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        
        # 生成日期范围
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        sample_data = []
        
        for stock_code in stock_codes:
            current_price = base_price
            
            for date in date_range:
                # 模拟价格变动（随机游走）
                import random
                change_pct = random.uniform(-0.05, 0.05)  # ±5%的变动
                current_price *= (1 + change_pct)
                
                # 计算其他价格（确保数据质量：low <= open,close <= high）
                high = current_price * random.uniform(1.0, 1.03)
                low = current_price * random.uniform(0.97, 1.0)
                open_price = max(low, min(high, current_price * random.uniform(0.98, 1.02)))
                current_price = max(low, min(high, current_price))
                
                # 生成成交量
                volume = random.randint(1000000, 10000000)
                amount = volume * current_price
                
                sample_data.append({
                    'ts_code': stock_code,
                    'trade_date': date.strftime('%Y-%m-%d'),
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(current_price, 2),
                    'pre_close': round(current_price / (1 + change_pct), 2),
                    'change': round(current_price - current_price / (1 + change_pct), 2),
                    'pct_chg': round(change_pct * 100, 2),
                    'vol': volume,
                    'amount': round(amount, 2)
                })
        
        df = pd.DataFrame(sample_data)
        logger.info(f"创建了 {len(df)} 行样本股票数据")
        return df
    
    def create_sample_stock_basic(self, stock_codes: List[str] = None) -> pd.DataFrame:
        """创建样本股票基本信息
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            样本股票基本信息
        """
        if stock_codes is None:
            stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        
        stock_names = {
            '000001.SZ': '平安银行',
            '000002.SZ': '万科A',
            '600000.SH': '浦发银行',
            '600036.SH': '招商银行',
            '000858.SZ': '五粮液'
        }
        
        sample_data = []
        for stock_code in stock_codes:
            sample_data.append({
                'ts_code': stock_code,
                'symbol': stock_code.split('.')[0],
                'name': stock_names.get(stock_code, f'股票{stock_code}'),
                'area': '深圳' if stock_code.endswith('.SZ') else '上海',
                'industry': '银行' if '银行' in stock_names.get(stock_code, '') else '其他',
                'market': 'main',
                'list_date': '20000101'
            })
        
        df = pd.DataFrame(sample_data)
        logger.info(f"创建了 {len(df)} 行样本股票基本信息")
        return df
    
    def setup_test_environment(self, 
                             stock_codes: List[str] = None,
                             start_date: str = '20240101',
                             end_date: str = '20240131') -> bool:
        """设置测试环境
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            是否成功
        """
        try:
            logger.info("开始设置测试环境...")
            
            # 1. 等待数据库就绪
            if not self.wait_for_database():
                return False
            
            # 2. 清空现有数据
            self.clear_all_tables()
            
            # 3. 插入样本股票基本信息
            stock_basic_data = self.create_sample_stock_basic(stock_codes)
            if self.table_exists('stock_basic'):
                self.insert_test_data('stock_basic', stock_basic_data)
            
            # 4. 插入样本股票日线数据
            stock_daily_data = self.create_sample_stock_data(stock_codes, start_date, end_date)
            if self.table_exists('stock_daily'):
                self.insert_test_data('stock_daily', stock_daily_data)
            
            logger.info("测试环境设置完成")
            return True
            
        except Exception as e:
            logger.error(f"设置测试环境失败: {str(e)}")
            return False
    
    def cleanup_test_environment(self) -> bool:
        """清理测试环境
        
        Returns:
            是否成功
        """
        try:
            logger.info("开始清理测试环境...")
            self.clear_all_tables()
            logger.info("测试环境清理完成")
            return True
        except Exception as e:
            logger.error(f"清理测试环境失败: {str(e)}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息
        
        Returns:
            数据库统计信息
        """
        stats = {
            'connection_status': self.check_connection(),
            'tables': {},
            'total_rows': 0
        }
        
        try:
            tables = self.get_table_list()
            for table in tables:
                row_count = self.get_table_row_count(table)
                stats['tables'][table] = row_count
                stats['total_rows'] += row_count
            
            logger.info(f"数据库统计: {len(tables)} 个表, 总计 {stats['total_rows']} 行数据")
            
        except Exception as e:
            logger.error(f"获取数据库统计失败: {str(e)}")
        
        return stats
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("测试数据库连接已关闭")


# 全局测试数据库管理器实例
_test_db_manager = None


def get_test_db_engine() -> Engine:
    """获取测试数据库引擎
    
    Returns:
        SQLAlchemy引擎对象
    """
    global _test_db_manager
    if _test_db_manager is None:
        _test_db_manager = TestDatabaseManager()
    return _test_db_manager.get_engine()


def get_test_db_manager() -> TestDatabaseManager:
    """获取测试数据库管理器
    
    Returns:
        测试数据库管理器实例
    """
    global _test_db_manager
    if _test_db_manager is None:
        _test_db_manager = TestDatabaseManager()
    return _test_db_manager


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试数据库管理工具')
    parser.add_argument('--action', choices=['setup', 'cleanup', 'stats', 'check'], 
                       default='check', help='执行的操作')
    parser.add_argument('--start-date', type=str, default='20240101',
                       help='开始日期')
    parser.add_argument('--end-date', type=str, default='20240131',
                       help='结束日期')
    
    args = parser.parse_args()
    
    try:
        manager = TestDatabaseManager()
        
        if args.action == 'setup':
            success = manager.setup_test_environment(
                start_date=args.start_date,
                end_date=args.end_date
            )
            print(f"测试环境设置: {'成功' if success else '失败'}")
            
        elif args.action == 'cleanup':
            success = manager.cleanup_test_environment()
            print(f"测试环境清理: {'成功' if success else '失败'}")
            
        elif args.action == 'stats':
            stats = manager.get_database_stats()
            print(f"数据库统计信息: {stats}")
            
        elif args.action == 'check':
            connected = manager.check_connection()
            print(f"数据库连接状态: {'正常' if connected else '异常'}")
            
        manager.close()
        
    except Exception as e:
        print(f"操作失败: {str(e)}")
        raise