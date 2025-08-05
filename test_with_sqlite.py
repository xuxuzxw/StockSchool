#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool SQLite测试环境

为测试目的创建SQLite数据库环境，验证系统功能
"""

import os
import sys
from pathlib import Path
import sqlite3
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.db import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLiteTestEnvironment:
    """SQLite测试环境"""
    
    def __init__(self):
        self.db_path = project_root / "data" / "test_stock_data.db"
        self.db_path.parent.mkdir(exist_ok=True)
        
    def create_test_tables(self):
        """创建测试表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建股票基础信息表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            area TEXT,
            industry TEXT,
            market TEXT,
            exchange TEXT,
            list_status TEXT,
            list_date TEXT,
            delist_date TEXT
        )
        """)
        
        # 创建股票日线数据表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            UNIQUE(ts_code, trade_date)
        )
        """)
        
        # 创建交易日历表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_calendar (
            cal_date TEXT PRIMARY KEY,
            is_open INTEGER,
            pretrade_date TEXT
        )
        """)
        
        # 创建技术因子表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            factor_value REAL,
            factor_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ts_code, trade_date, factor_name)
        )
        """)
        
        # 插入测试数据
        self._insert_test_data(cursor)
        
        conn.commit()
        conn.close()
        logger.info("✅ SQLite测试表创建完成")
    
    def _insert_test_data(self, cursor):
        """插入测试数据"""
        # 插入测试股票
        test_stocks = [
            ('000001.SZ', '000001', '平安银行', '深圳', '银行', '主板', 'SZSE', 'L', '1991-04-03', None),
            ('000002.SZ', '000002', '万科A', '深圳', '房地产', '主板', 'SZSE', 'L', '1991-01-29', None),
            ('600000.SH', '600000', '浦发银行', '上海', '银行', '主板', 'SSE', 'L', '1999-11-10', None),
            ('600519.SH', '600519', '贵州茅台', '贵州', '白酒', '主板', 'SSE', 'L', '2001-08-27', None),
            ('601398.SH', '601398', '工商银行', '北京', '银行', '主板', 'SSE', 'L', '2006-10-27', None),
        ]
        
        cursor.executemany("""
        INSERT OR IGNORE INTO stock_basic VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, test_stocks)
        
        # 插入测试交易日历
        import datetime
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            is_open = current_date.weekday() < 5  # 周一到周五为交易日
            cursor.execute("""
            INSERT OR IGNORE INTO trade_calendar VALUES (?, ?, ?)
            """, (current_date.strftime('%Y-%m-%d'), 1 if is_open else 0, None))
            current_date += datetime.timedelta(days=1)
        
        # 插入测试行情数据
        import random
        for ts_code in ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH', '601398.SH']:
            base_price = random.uniform(10, 100)
            for i in range(30):  # 最近30天数据
                date = (datetime.date.today() - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
                open_price = base_price * (1 + random.uniform(-0.05, 0.05))
                high = open_price * (1 + random.uniform(0, 0.1))
                low = open_price * (1 + random.uniform(-0.1, 0))
                close = random.uniform(low, high)
                pre_close = open_price
                change = close - pre_close
                pct_chg = (change / pre_close) * 100
                vol = random.uniform(10000, 1000000)
                amount = vol * close
                
                cursor.execute("""
                INSERT OR IGNORE INTO stock_daily 
                (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ts_code, date, open_price, high, low, close, pre_close, change, pct_chg, vol, amount))
    
    def get_test_config(self):
        """获取测试配置"""
        return {
            'database_url': f'sqlite:///{self.db_path}',
            'db_path': str(self.db_path)
        }
    
    def run_factor_calculation_test(self):
        """运行因子计算测试"""
        import os
        os.environ['DATABASE_URL'] = f'sqlite:///{self.db_path}'
        
        try:
            from src.compute.technical_factor_engine import TechnicalFactorEngine
            from src.compute.factor_engine import EngineFactory
            
            engine = EngineFactory.create_technical_engine()
            
            # 测试因子计算
            result = engine.calculate_factors(
                stock_codes=['000001.SZ', '000002.SZ'],
                start_date='2024-01-01',
                end_date='2024-12-31',
                factor_types=['MA20', 'RSI14', 'MACD']
            )
            
            logger.info(f"✅ 因子计算测试成功，计算了 {len(result)} 个因子")
            return True
            
        except Exception as e:
            logger.error(f"❌ 因子计算测试失败: {e}")
            return False
    
    def cleanup(self):
        """清理测试环境"""
        if self.db_path.exists():
            self.db_path.unlink()
            logger.info("✅ 测试环境清理完成")


def main():
    """主函数"""
    print("=== StockSchool SQLite测试环境 ===")
    
    test_env = SQLiteTestEnvironment()
    
    try:
        # 创建测试环境
        test_env.create_test_tables()
        
        # 获取配置
        config = test_env.get_test_config()
        print(f"测试数据库: {config['db_path']}")
        
        # 运行因子计算测试
        success = test_env.run_factor_calculation_test()
        
        if success:
            print("🎉 所有测试通过！")
        else:
            print("❌ 测试失败")
            
    finally:
        # 清理测试环境
        # test_env.cleanup()  # 保留测试数据用于调试
        pass


if __name__ == "__main__":
    main()