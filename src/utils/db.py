#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库工具类
抽象重复的数据库操作代码

主要功能：
1. 数据库连接管理
2. 批量数据插入/更新
3. 通用查询操作
4. 事务管理
5. 错误处理

作者: StockSchool Team
创建时间: 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from dotenv import load_dotenv

# 加载环境变量，指定编码格式
load_dotenv(encoding='utf-8')

def get_db_engine():
    """获取数据库引擎"""
    db_user = "stockschool"
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_name = "stockschool"
    db_host = "localhost"  # 使用localhost连接本地数据库
    db_port = os.getenv('POSTGRES_PORT', '15432')
    
    # 确保密码是字符串并且正确编码
    if db_password:
        db_password = str(db_password).strip()
    
    db_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    print(f"数据库连接字符串: {db_uri}")
    
    # 添加连接参数以处理编码问题
    engine = create_engine(db_uri, connect_args={"client_encoding": "utf8"})
    return engine


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, engine: Optional[Engine] = None):
        """初始化数据库管理器
        
        Args:
            engine: 数据库引擎，如果为None则使用默认引擎
        """
        self.engine = engine or get_db_engine()
        print("✅ 数据库管理器初始化完成")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def get_transaction(self):
        """获取事务的上下文管理器"""
        conn = self.engine.connect()
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
        except Exception:
            trans.rollback()
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """执行查询并返回DataFrame
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果DataFrame
        """
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(text(query), conn, params=params or {})
            return df
        except SQLAlchemyError as e:
            print(f"❌ 查询执行失败: {e}")
            raise
    
    def execute_sql(self, sql: str, params: Optional[Dict] = None) -> Any:
        """执行SQL语句
        
        Args:
            sql: SQL语句
            params: 参数
            
        Returns:
            执行结果
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(sql), params or {})
                conn.commit()
                return result
        except SQLAlchemyError as e:
            print(f"❌ SQL执行失败: {e}")
            raise
    
    def batch_insert_or_update(self, 
                              table_name: str, 
                              data: Union[pd.DataFrame, List[Dict]], 
                              conflict_columns: Optional[List[str]] = None,
                              update_columns: Optional[List[str]] = None,
                              batch_size: int = 1000) -> bool:
        """批量插入或更新数据
        
        Args:
            table_name: 表名
            data: 数据（DataFrame或字典列表）
            conflict_columns: 冲突检测列
            update_columns: 更新列
            batch_size: 批次大小
            
        Returns:
            是否成功
        """
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            if df.empty:
                print(f"⚠️ 表 {table_name} 的数据为空，跳过插入")
                return True
            
            # 分批处理
            total_rows = len(df)
            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                if conflict_columns and update_columns:
                    # 使用UPSERT逻辑
                    self._upsert_batch(table_name, batch_df, conflict_columns, update_columns)
                else:
                    # 简单插入
                    batch_df.to_sql(
                        table_name,
                        self.engine,
                        if_exists='append',
                        index=False,
                        method='multi'
                    )
                
                print(f"✅ 表 {table_name} 批次 {i//batch_size + 1} 插入完成，共 {len(batch_df)} 条记录")
            
            print(f"✅ 表 {table_name} 数据插入完成，共 {total_rows} 条记录")
            return True
            
        except Exception as e:
            print(f"❌ 批量插入表 {table_name} 失败: {e}")
            raise
    
    def _upsert_batch(self, 
                     table_name: str, 
                     df: pd.DataFrame, 
                     conflict_columns: List[str], 
                     update_columns: List[str]):
        """执行UPSERT操作
        
        Args:
            table_name: 表名
            df: 数据DataFrame
            conflict_columns: 冲突检测列
            update_columns: 更新列
        """
        # 构建UPSERT SQL（适用于PostgreSQL）
        columns = list(df.columns)
        placeholders = ', '.join([f':{col}' for col in columns])
        
        conflict_clause = ', '.join(conflict_columns)
        update_clause = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_columns])
        
        upsert_sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_clause})
        DO UPDATE SET {update_clause}, updated_at = CURRENT_TIMESTAMP
        """
        
        # 对于SQLite，使用INSERT OR REPLACE
        if 'sqlite' in str(self.engine.url).lower():
            upsert_sql = f"""
            INSERT OR REPLACE INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
            """
        
        with self.get_transaction() as conn:
            for _, row in df.iterrows():
                conn.execute(text(upsert_sql), row.to_dict())
    
    def get_stock_list(self, list_status: str = 'L') -> List[str]:
        """获取股票列表
        
        Args:
            list_status: 上市状态
            
        Returns:
            股票代码列表
        """
        query = "SELECT DISTINCT ts_code FROM stock_basic WHERE list_status = :list_status ORDER BY ts_code"
        
        with self.get_connection() as conn:
            result = conn.execute(text(query), {'list_status': list_status})
            stocks = [row[0] for row in result.fetchall()]
        
        print(f"✅ 获取到 {len(stocks)} 只股票（状态: {list_status}）")
        return stocks
    
    def get_stock_data(self, 
                      ts_code: str, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      limit: Optional[int] = None,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取股票数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            limit: 限制条数
            columns: 指定列
            
        Returns:
            股票数据DataFrame
        """
        if columns:
            column_str = ', '.join(columns)
        else:
            column_str = 'trade_date, ts_code, open, high, low, close, vol, amount'
        
        query = f"""
            SELECT {column_str}
            FROM stock_daily 
            WHERE ts_code = :ts_code
        """
        
        params = {'ts_code': ts_code}
        
        if start_date:
            query += " AND trade_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND trade_date <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY trade_date"
        
        if limit:
            query += " LIMIT :limit"
            params['limit'] = limit
        
        df = self.execute_query(query, params)
        
        if df.empty:
            print(f"⚠️ 未找到股票 {ts_code} 的数据")
            return df
        
        # 转换日期格式
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        print(f"✅ 获取股票 {ts_code} 数据 {len(df)} 条")
        return df
    
    def get_trading_dates(self, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> List[str]:
        """获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日期列表
        """
        query = "SELECT cal_date FROM trade_cal WHERE is_open = 1"
        params = {}
        
        if start_date:
            query += " AND cal_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND cal_date <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY cal_date"
        
        with self.get_connection() as conn:
            result = conn.execute(text(query), params)
            dates = [row[0] for row in result.fetchall()]
        
        print(f"✅ 获取到 {len(dates)} 个交易日")
        return dates
    
    def batch_process_stocks(self, 
                           stocks: List[str], 
                           process_func, 
                           batch_size: int = 50,
                           **kwargs) -> Dict[str, Any]:
        """批量处理股票
        
        Args:
            stocks: 股票代码列表
            process_func: 处理函数
            batch_size: 批次大小
            **kwargs: 传递给处理函数的参数
            
        Returns:
            处理结果统计
        """
        total_stocks = len(stocks)
        success_count = 0
        error_count = 0
        errors = []
        
        print(f"开始批量处理 {total_stocks} 只股票，批次大小: {batch_size}")
        
        for i in range(0, total_stocks, batch_size):
            batch_stocks = stocks[i:i+batch_size]
            print(f"处理第 {i//batch_size + 1} 批，共 {len(batch_stocks)} 只股票")
            
            for ts_code in batch_stocks:
                try:
                    result = process_func(ts_code, **kwargs)
                    if result:
                        success_count += 1
                    else:
                        error_count += 1
                        errors.append(f"{ts_code}: 处理返回False")
                except Exception as e:
                    error_count += 1
                    error_msg = f"{ts_code}: {str(e)}"
                    errors.append(error_msg)
                    print(f"❌ 处理股票 {ts_code} 失败: {e}")
            
            print(f"第 {i//batch_size + 1} 批处理完成")
        
        result_stats = {
            'total': total_stocks,
            'success': success_count,
            'error': error_count,
            'errors': errors
        }
        
        print(f"批量处理完成: 总计 {total_stocks}, 成功 {success_count}, 失败 {error_count}")
        return result_stats


# 全局数据库管理器实例
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# 便捷函数
def execute_query(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """执行查询的便捷函数"""
    return get_db_manager().execute_query(query, params)

def execute_sql(sql: str, params: Optional[Dict] = None) -> Any:
    """执行SQL的便捷函数"""
    return get_db_manager().execute_sql(sql, params)

def batch_insert_or_update(table_name: str, 
                          data: Union[pd.DataFrame, List[Dict]], 
                          conflict_columns: Optional[List[str]] = None,
                          update_columns: Optional[List[str]] = None,
                          batch_size: int = 1000) -> bool:
    """批量插入或更新的便捷函数"""
    return get_db_manager().batch_insert_or_update(
        table_name, data, conflict_columns, update_columns, batch_size
    )

def get_stock_list(list_status: str = 'L') -> List[str]:
    """获取股票列表的便捷函数"""
    return get_db_manager().get_stock_list(list_status)

def get_stock_data(ts_code: str, 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None,
                  limit: Optional[int] = None,
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
    """获取股票数据的便捷函数"""
    return get_db_manager().get_stock_data(ts_code, start_date, end_date, limit, columns)

def get_trading_dates(start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> List[str]:
    """获取交易日历的便捷函数"""
    return get_db_manager().get_trading_dates(start_date, end_date)