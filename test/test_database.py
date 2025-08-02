#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.db import get_db_engine
from src.utils.retry import idempotent_retry
from sqlalchemy import text

class TestDatabase:
    """数据库连接测试"""
    
    def test_db_connection(self):
        """测试数据库连接"""
        engine = get_db_engine()
        assert engine is not None
        
        with engine.connect() as conn:
            result = conn.execute(text('SELECT 1'))
            assert result.scalar() == 1
    
    def test_stock_basic_table(self):
        """测试股票基本信息表"""
        engine = get_db_engine()
        
        with engine.connect() as conn:
            # 检查表是否存在
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'stock_basic'
                )
            """))
            assert result.scalar() is True
            
            # 检查是否有数据
            result = conn.execute(text('SELECT COUNT(*) FROM stock_basic'))
            count = result.scalar()
            assert count > 0
    
    def test_trade_calendar_table(self):
        """测试交易日历表"""
        engine = get_db_engine()
        
        with engine.connect() as conn:
            # 检查表是否存在
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'trade_calendar'
                )
            """))
            assert result.scalar() is True
            
            # 检查是否有数据
            result = conn.execute(text('SELECT COUNT(*) FROM trade_calendar'))
            count = result.scalar()
            assert count > 0
    
    def test_stock_daily_table(self):
        """测试日线数据表"""
        engine = get_db_engine()
        
        with engine.connect() as conn:
            # 检查表是否存在
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'stock_daily'
                )
            """))
            assert result.scalar() is True
            
            # 检查是否有数据
            result = conn.execute(text('SELECT COUNT(*) FROM stock_daily'))
            count = result.scalar()
            assert count > 0

class TestRetryDecorator:
    """重试装饰器测试"""
    
    def test_retry_success(self):
        """测试重试装饰器成功情况"""
        @idempotent_retry(max_retries=3)
        def success_function():
            return "success"
        
        result = success_function()
        assert result == "success"
    
    def test_retry_failure(self):
        """测试重试装饰器失败情况"""
        call_count = 0
        
        @idempotent_retry(max_retries=3)
        def failure_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Test exception")
        
        with pytest.raises(Exception):
            failure_function()
        
        assert call_count == 3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])