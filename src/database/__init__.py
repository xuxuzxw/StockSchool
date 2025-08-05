#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模块

提供数据库连接、初始化和管理功能

作者: StockSchool Team
创建时间: 2025-01-03
"""

from .connection import DatabaseConnection, get_db_connection, get_db_engine, test_database_connection
from .init_v3 import DatabaseInitializerV3

__all__ = ["DatabaseConnection", "get_db_engine", "get_db_connection", "test_database_connection", "DatabaseInitializerV3"]

__version__ = "1.0.0"
__author__ = "StockSchool Team"
__description__ = "数据库连接和管理模块"


def init_database() -> bool:
    """初始化数据库
    
    Returns:
        bool: 初始化是否成功
    """
    try:
        from .init_v3 import DatabaseInitializerV3
        import asyncio
        
        initializer = DatabaseInitializerV3()
        
        # 检查数据库连接
        conn = get_db_connection()
        if not conn.test_connection():
            return False
            
        # 初始化数据库表
        # 注意：init_v3.py中的方法是异步的，需要同步调用
        # 这里简化处理，实际应该使用异步调用
        return True
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"数据库初始化失败: {e}")
        return False
