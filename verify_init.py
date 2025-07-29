#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证数据库初始化结果"""

from src.utils.db import DatabaseManager

def verify_database_init():
    """验证数据库初始化结果"""
    print("\n=== 验证数据库初始化 ===")
    
    try:
        # 创建数据库管理器实例
        db_manager = DatabaseManager()
        
        # 检查已安装的扩展
        extensions = db_manager.execute_query('SELECT extname FROM pg_extension;')
        print("已安装的扩展:")
        for _, row in extensions.iterrows():
            print(f"  - {row['extname']}")
        
        # 检查已创建的函数
        functions = db_manager.execute_query("SELECT proname FROM pg_proc WHERE proname = 'create_hypertable_if_not_exists'")
        print("\n已创建的函数:")
        for _, row in functions.iterrows():
            print(f"  - {row['proname']}")
            
        print("\n✅ 数据库初始化验证完成")
        
    except Exception as e:
        print(f"❌ 数据库初始化验证失败: {e}")

if __name__ == "__main__":
    verify_database_init()