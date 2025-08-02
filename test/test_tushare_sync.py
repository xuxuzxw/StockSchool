#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import sys
import os
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.tushare_sync import TushareSynchronizer
from src.utils.db import get_db_engine
from sqlalchemy import text

class TestTushareSynchronizer:
    """Tushare数据同步器测试"""
    
    def test_synchronizer_init(self):
        """测试同步器初始化"""
        # 确保环境变量存在
        if not os.getenv("TUSHARE_TOKEN"):
            pytest.skip("TUSHARE_TOKEN环境变量未设置")
        
        sync = TushareSynchronizer()
        assert sync.pro is not None
        assert sync.engine is not None
    
    def test_get_last_trade_date(self):
        """测试获取最后交易日期"""
        if not os.getenv("TUSHARE_TOKEN"):
            pytest.skip("TUSHARE_TOKEN环境变量未设置")
        
        sync = TushareSynchronizer()
        last_date = sync.get_last_trade_date()
        assert isinstance(last_date, str)
        assert len(last_date) == 8  # YYYYMMDD格式
    
    def test_database_tables_exist(self):
        """测试数据库表是否存在"""
        engine = get_db_engine()
        
        with engine.connect() as conn:
            # 检查必要的表是否存在
            tables = ['stock_basic', 'trade_calendar', 'stock_daily']
            
            for table in tables:
                result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    )
                """))
                assert result.scalar() is True, f"表 {table} 不存在"
    
    def test_data_integrity(self):
        """测试数据完整性"""
        engine = get_db_engine()
        
        with engine.connect() as conn:
            # 检查股票基本信息数据
            result = conn.execute(text("""
                SELECT COUNT(*) FROM stock_basic 
                WHERE ts_code IS NOT NULL AND name IS NOT NULL
            """))
            stock_count = result.scalar()
            assert stock_count > 0, "股票基本信息数据为空"
            
            # 检查交易日历数据
            result = conn.execute(text("""
                SELECT COUNT(*) FROM trade_calendar 
                WHERE cal_date IS NOT NULL AND is_open IS NOT NULL
            """))
            calendar_count = result.scalar()
            assert calendar_count > 0, "交易日历数据为空"
            
            # 检查日线数据
            result = conn.execute(text("""
                SELECT COUNT(*) FROM stock_daily 
                WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL
            """))
            daily_count = result.scalar()
            assert daily_count > 0, "日线数据为空"
    
    def test_data_consistency(self):
        """测试数据一致性"""
        engine = get_db_engine()
        
        with engine.connect() as conn:
            # 检查股票代码一致性
            result = conn.execute(text("""
                SELECT COUNT(DISTINCT sd.ts_code) as daily_stocks,
                       COUNT(DISTINCT sb.ts_code) as basic_stocks
                FROM stock_daily sd
                FULL OUTER JOIN stock_basic sb ON sd.ts_code = sb.ts_code
            """))
            row = result.fetchone()
            
            # 日线数据中的股票代码应该都在基本信息表中（允许少量退市股票不在基本信息表中）
            result = conn.execute(text("""
                SELECT COUNT(*) FROM stock_daily sd
                LEFT JOIN stock_basic sb ON sd.ts_code = sb.ts_code
                WHERE sb.ts_code IS NULL
            """))
            orphan_count = result.scalar()
            
            # 获取总的日线数据股票数量
            result = conn.execute(text("""
                SELECT COUNT(DISTINCT ts_code) FROM stock_daily
            """))
            total_daily_stocks = result.scalar()
            
            # 允许最多5%的股票不在基本信息表中（主要是退市股票）
            max_allowed_orphans = int(total_daily_stocks * 0.05)
            assert orphan_count <= max_allowed_orphans, f"发现 {orphan_count} 条日线数据没有对应的股票基本信息，超过允许的 {max_allowed_orphans} 条"
    
    def test_date_format_consistency(self):
        """测试日期格式一致性"""
        engine = get_db_engine()
        
        with engine.connect() as conn:
            # 检查交易日历日期格式
            result = conn.execute(text("""
                SELECT COUNT(*) FROM trade_calendar 
                WHERE cal_date IS NULL OR cal_date::text !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
            """))
            invalid_calendar_dates = result.scalar()
            assert invalid_calendar_dates == 0, f"发现 {invalid_calendar_dates} 条无效的交易日历日期"
            
            # 检查日线数据日期格式
            result = conn.execute(text("""
                SELECT COUNT(*) FROM stock_daily 
                WHERE trade_date IS NULL OR trade_date::text !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
            """))
            invalid_daily_dates = result.scalar()
            assert invalid_daily_dates == 0, f"发现 {invalid_daily_dates} 条无效的日线数据日期"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])