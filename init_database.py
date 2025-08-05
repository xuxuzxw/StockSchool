#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool 数据库初始化脚本

根据StockSchool_完整程序构建指南.md要求，初始化完整的数据库结构
包括：
1. PostgreSQL + TimescaleDB扩展
2. 核心数据表结构
3. 时序数据超表配置
4. 索引优化
5. 基础数据填充

使用方法:
    python init_database.py [--check-only] [--force-init]

作者: StockSchool Team
创建时间: 2025-01-19
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.db import get_db_engine, DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """数据库初始化器"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'postgresql://stockschool:password@localhost:5432/stockschool'
        )
        self.engine = None
        
    def check_postgresql_connection(self) -> bool:
        """检查PostgreSQL连接"""
        try:
            conn = psycopg2.connect(self.database_url)
            conn.close()
            logger.info("✅ PostgreSQL连接成功")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL连接失败: {e}")
            return False
    
    def check_timescaledb_extension(self) -> bool:
        """检查TimescaleDB扩展"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                logger.info("✅ TimescaleDB扩展已安装")
                return True
            else:
                logger.warning("⚠️ TimescaleDB扩展未安装")
                return False
        except Exception as e:
            logger.error(f"❌ TimescaleDB检查失败: {e}")
            return False
    
    def create_database_and_user(self) -> bool:
        """创建数据库和用户（如果需要）"""
        try:
            # 连接到默认的postgres数据库
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                user='postgres',
                password=os.getenv('POSTGRES_PASSWORD', 'password'),
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # 检查数据库是否存在
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'stockschool'")
            if not cursor.fetchone():
                cursor.execute("CREATE DATABASE stockschool")
                logger.info("✅ 数据库 'stockschool' 已创建")
            
            # 检查用户是否存在
            cursor.execute("SELECT 1 FROM pg_user WHERE usename = 'stockschool'")
            if not cursor.fetchone():
                cursor.execute("CREATE USER stockschool WITH PASSWORD 'password'")
                cursor.execute("GRANT ALL PRIVILEGES ON DATABASE stockschool TO stockschool")
                logger.info("✅ 用户 'stockschool' 已创建")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据库/用户创建失败: {e}")
            return False
    
    def initialize_schema(self) -> bool:
        """初始化数据库Schema"""
        try:
            self.engine = create_engine(self.database_url)
            
            # 读取数据库Schema文件
            schema_file = project_root / "database_schema.sql"
            if not schema_file.exists():
                logger.error(f"❌ Schema文件不存在: {schema_file}")
                return False
            
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # 执行Schema创建
            with self.engine.connect() as conn:
                logger.info("开始执行数据库Schema初始化...")
                
                # 分割SQL语句并逐条执行
                statements = schema_sql.split(';')
                for statement in statements:
                    statement = statement.strip()
                    if statement and not statement.startswith('--'):
                        try:
                            conn.execute(text(statement))
                        except Exception as e:
                            logger.warning(f"⚠️ SQL执行警告: {e}")
                
                conn.commit()
                logger.info("✅ 数据库Schema初始化完成")
                return True
                
        except Exception as e:
            logger.error(f"❌ Schema初始化失败: {e}")
            return False
    
    def verify_tables(self) -> List[str]:
        """验证表是否创建成功"""
        try:
            if not self.engine:
                self.engine = create_engine(self.database_url)
            
            inspector = self.engine
            with inspector.connect() as conn:
                # 获取所有表
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name
                """))
                tables = [row[0] for row in result]
                
                logger.info(f"✅ 已创建 {len(tables)} 个表")
                
                # 检查核心表
                core_tables = [
                    'stock_basic', 'stock_daily', 'daily_basic', 'trade_calendar',
                    'income', 'balance_sheet', 'cash_flow', 'financial_indicator',
                    'technical_factors', 'fundamental_factors', 'sentiment_factors'
                ]
                
                missing_tables = []
                for table in core_tables:
                    if table not in tables:
                        missing_tables.append(table)
                        logger.warning(f"⚠️ 缺少表: {table}")
                    else:
                        logger.info(f"✅ 表 {table} 已存在")
                
                return missing_tables
                
        except Exception as e:
            logger.error(f"❌ 表验证失败: {e}")
            return []
    
    def initialize_timescaledb(self) -> bool:
        """初始化TimescaleDB超表"""
        try:
            if not self.engine:
                self.engine = create_engine(self.database_url)
            
            with self.engine.connect() as conn:
                # 检查并创建超表
                hypertables = [
                    'stock_daily', 'daily_basic', 'money_flow', 'index_daily',
                    'index_daily_basic', 'technical_factors', 'fundamental_factors',
                    'sentiment_factors'
                ]
                
                for table in hypertables:
                    try:
                        conn.execute(text(f"""
                            SELECT create_hypertable('{table}', 'trade_date', 
                                                   if_not_exists => TRUE)
                        """))
                        logger.info(f"✅ 超表 {table} 已创建")
                    except Exception as e:
                        logger.warning(f"⚠️ 超表 {table} 创建失败: {e}")
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ TimescaleDB初始化失败: {e}")
            return False
    
    def run_full_initialization(self) -> bool:
        """运行完整的数据库初始化流程"""
        logger.info("开始StockSchool数据库初始化...")
        
        steps = [
            ("检查PostgreSQL连接", self.check_postgresql_connection),
            ("创建数据库和用户", self.create_database_and_user),
            ("检查TimescaleDB扩展", self.check_timescaledb_extension),
            ("初始化数据库Schema", self.initialize_schema),
            ("初始化TimescaleDB超表", self.initialize_timescaledb),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"执行步骤: {step_name}")
            if not step_func():
                logger.error(f"❌ 步骤失败: {step_name}")
                return False
        
        # 验证结果
        missing_tables = self.verify_tables()
        if missing_tables:
            logger.error(f"❌ 初始化不完整，缺少表: {missing_tables}")
            return False
        
        logger.info("🎉 StockSchool数据库初始化完成！")
        return True
    
    def check_status(self) -> dict:
        """检查数据库状态"""
        status = {
            'postgresql_connected': self.check_postgresql_connection(),
            'timescaledb_installed': self.check_timescaledb_extension(),
            'missing_tables': self.verify_tables()
        }
        
        if all([
            status['postgresql_connected'],
            status['timescaledb_installed'],
            len(status['missing_tables']) == 0
        ]):
            status['overall_status'] = 'healthy'
        else:
            status['overall_status'] = 'needs_init'
        
        return status


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='StockSchool数据库初始化工具')
    parser.add_argument('--check-only', action='store_true', help='仅检查状态')
    parser.add_argument('--force-init', action='store_true', help='强制重新初始化')
    parser.add_argument('--database-url', help='数据库连接字符串')
    
    args = parser.parse_args()
    
    initializer = DatabaseInitializer(args.database_url)
    
    if args.check_only:
        status = initializer.check_status()
        print("\n=== 数据库状态检查 ===")
        print(f"PostgreSQL连接: {'✅' if status['postgresql_connected'] else '❌'}")
        print(f"TimescaleDB扩展: {'✅' if status['timescaledb_installed'] else '❌'}")
        print(f"缺失表: {len(status['missing_tables'])}")
        print(f"整体状态: {status['overall_status']}")
        return
    
    # 运行完整初始化
    success = initializer.run_full_initialization()
    if success:
        print("\n🎉 数据库初始化成功完成！")
        print("现在可以运行: python scripts/test_full_pipeline.py")
    else:
        print("\n❌ 数据库初始化失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())