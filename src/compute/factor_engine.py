#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算引擎
负责从数据库获取数据，计算因子，并存储结果
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import text, create_engine
from loguru import logger

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import get_db_engine
from utils.retry import idempotent_retry
from utils.config_loader import config
from compute.indicators import FactorCalculator, FundamentalFactorCalculator


class FactorEngine:
    """因子计算引擎"""
    
    def __init__(self):
        """初始化因子引擎"""
        self.engine = get_db_engine()
        self.calculator = FactorCalculator()
        self.fundamental_calculator = FundamentalFactorCalculator()
        logger.info("因子计算引擎初始化完成")
    
    def get_stock_data(self, 
                      ts_code: str, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """获取股票数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            limit: 限制条数
            
        Returns:
            股票数据DataFrame
        """
        query = """
            SELECT trade_date, ts_code, open, high, low, close, vol, amount
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
            
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
            
        if df.empty:
            logger.warning(f"未找到股票 {ts_code} 的数据")
            return df
            
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        logger.info(f"获取股票 {ts_code} 数据 {len(df)} 条")
        return df
    
    def get_all_stocks(self) -> List[str]:
        """获取所有股票代码
        
        Returns:
            股票代码列表
        """
        query = "SELECT DISTINCT ts_code FROM stock_basic WHERE list_status = 'L' ORDER BY ts_code"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            stocks = [row[0] for row in result.fetchall()]
            
        logger.info(f"获取到 {len(stocks)} 只股票")
        return stocks
    
    def create_factor_tables(self):
        """创建因子存储表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_factors (
            id SERIAL PRIMARY KEY,
            ts_code VARCHAR(20) NOT NULL,
            trade_date DATE NOT NULL,
            factor_name VARCHAR(50) NOT NULL,
            factor_value DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ts_code, trade_date, factor_name)
        );
        
        CREATE INDEX IF NOT EXISTS idx_stock_factors_code_date 
        ON stock_factors(ts_code, trade_date);
        
        CREATE INDEX IF NOT EXISTS idx_stock_factors_name_date 
        ON stock_factors(factor_name, trade_date);
        
        -- 创建技术因子汇总表
        CREATE TABLE IF NOT EXISTS stock_technical_factors (
            ts_code VARCHAR(20) NOT NULL,
            trade_date DATE NOT NULL,
            
            -- 动量因子
            momentum_5 DOUBLE PRECISION,
            momentum_10 DOUBLE PRECISION,
            momentum_20 DOUBLE PRECISION,
            rsi_14 DOUBLE PRECISION,
            williams_r_14 DOUBLE PRECISION,
            
            -- 趋势因子
            sma_5 DOUBLE PRECISION,
            sma_10 DOUBLE PRECISION,
            sma_20 DOUBLE PRECISION,
            sma_60 DOUBLE PRECISION,
            ema_12 DOUBLE PRECISION,
            ema_26 DOUBLE PRECISION,
            macd DOUBLE PRECISION,
            macd_signal DOUBLE PRECISION,
            macd_histogram DOUBLE PRECISION,
            price_to_sma20 DOUBLE PRECISION,
            price_to_sma60 DOUBLE PRECISION,
            
            -- 波动率因子
            volatility_5 DOUBLE PRECISION,
            volatility_20 DOUBLE PRECISION,
            volatility_60 DOUBLE PRECISION,
            atr_14 DOUBLE PRECISION,
            bb_upper DOUBLE PRECISION,
            bb_middle DOUBLE PRECISION,
            bb_lower DOUBLE PRECISION,
            bb_width DOUBLE PRECISION,
            bb_position DOUBLE PRECISION,
            
            -- 成交量因子
            volume_sma_5 DOUBLE PRECISION,
            volume_sma_20 DOUBLE PRECISION,
            volume_ratio_5 DOUBLE PRECISION,
            volume_ratio_20 DOUBLE PRECISION,
            vpt DOUBLE PRECISION,
            mfi DOUBLE PRECISION,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            PRIMARY KEY (ts_code, trade_date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_technical_factors_date 
        ON stock_technical_factors(trade_date);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
            
        logger.info("因子存储表创建完成")
    
    @idempotent_retry(max_retries=3)
    def calculate_stock_factors(self, ts_code: str, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> bool:
        """计算单只股票的因子
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            是否成功
        """
        try:
            # 获取股票数据
            df = self.get_stock_data(ts_code, start_date, end_date)
            
            if df.empty:
                logger.warning(f"股票 {ts_code} 无数据，跳过因子计算")
                return True
                
            # 从配置获取最小数据天数要求
            min_data_days = config.get('factor_params.min_data_days', 60)
            if len(df) < min_data_days:
                logger.warning(f"股票 {ts_code} 数据不足{min_data_days}天，跳过因子计算")
                return True
            
            # 计算技术因子
            factors_df = self.calculator.calculate_all_factors(df)
            
            # 准备存储数据
            factor_columns = [
                'momentum_5', 'momentum_10', 'momentum_20', 'rsi_14', 'williams_r_14',
                'sma_5', 'sma_10', 'sma_20', 'sma_60', 'ema_12', 'ema_26',
                'macd', 'macd_signal', 'macd_histogram', 'price_to_sma20', 'price_to_sma60',
                'volatility_5', 'volatility_20', 'volatility_60', 'atr_14',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                'volume_sma_5', 'volume_sma_20', 'volume_ratio_5', 'volume_ratio_20',
                'vpt', 'mfi'
            ]
            
            # 选择存在的因子列
            available_factors = [col for col in factor_columns if col in factors_df.columns]
            
            # 准备插入数据
            insert_data = factors_df[['ts_code', 'trade_date'] + available_factors].copy()
            insert_data['trade_date'] = insert_data['trade_date'].dt.strftime('%Y-%m-%d')
            
            # 删除包含NaN的行
            insert_data = insert_data.dropna()
            
            if insert_data.empty:
                logger.warning(f"股票 {ts_code} 计算后无有效因子数据")
                return True
            
            # 存储到数据库
            with self.engine.connect() as conn:
                # 先删除已存在的数据
                delete_sql = """
                    DELETE FROM stock_technical_factors 
                    WHERE ts_code = :ts_code 
                    AND trade_date BETWEEN :start_date AND :end_date
                """
                
                conn.execute(text(delete_sql), {
                    'ts_code': ts_code,
                    'start_date': insert_data['trade_date'].min(),
                    'end_date': insert_data['trade_date'].max()
                })
                
                # 插入新数据
                insert_data.to_sql(
                    'stock_technical_factors', 
                    conn, 
                    if_exists='append', 
                    index=False,
                    method='multi'
                )
                
                conn.commit()
                
            logger.info(f"股票 {ts_code} 因子计算完成，存储 {len(insert_data)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"股票 {ts_code} 因子计算失败: {str(e)}")
            raise
    
    def calculate_all_factors(self, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             stock_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """计算所有股票的因子
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 指定股票列表，如果为None则计算所有股票
            
        Returns:
            计算结果统计
        """
        # 确保因子表存在
        self.create_factor_tables()
        
        # 获取股票列表
        if stock_list is None:
            stocks = self.get_all_stocks()
        else:
            stocks = stock_list
            
        logger.info(f"开始计算 {len(stocks)} 只股票的因子")
        
        success_count = 0
        failed_count = 0
        failed_stocks = []
        
        for i, ts_code in enumerate(stocks, 1):
            try:
                logger.info(f"[{i}/{len(stocks)}] 计算股票 {ts_code} 的因子")
                
                success = self.calculate_stock_factors(ts_code, start_date, end_date)
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_stocks.append(ts_code)
                    
            except Exception as e:
                logger.error(f"股票 {ts_code} 因子计算异常: {str(e)}")
                failed_count += 1
                failed_stocks.append(ts_code)
                
            # 从配置获取进度报告间隔
            progress_interval = config.get('data_sync_params.progress_interval', 100)
            if i % progress_interval == 0:
                logger.info(f"进度: {i}/{len(stocks)}, 成功: {success_count}, 失败: {failed_count}")
        
        result = {
            'total_stocks': len(stocks),
            'success_count': success_count,
            'failed_count': failed_count,
            'failed_stocks': failed_stocks,
            'success_rate': success_count / len(stocks) if stocks else 0
        }
        
        logger.info(f"因子计算完成: 总计 {len(stocks)} 只股票, 成功 {success_count} 只, 失败 {failed_count} 只")
        
        return result
    
    def _calculate_fundamental_factors(self, 
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     stock_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """计算基本面因子
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 指定股票列表
            
        Returns:
            计算结果统计
        """
        logger.info("开始计算基本面因子")
        
        # 构建查询条件
        query = """
            SELECT ts_code, ann_date, end_date, report_type,
                   revenue, net_profit, total_assets, total_hldr_eqy_exc_min_int,
                   oper_cost, total_liab, total_cur_assets, total_cur_liab, inventories
            FROM financial_reports
            WHERE 1=1
        """
        
        params = {}
        
        if start_date:
            query += " AND end_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND end_date <= :end_date"
            params['end_date'] = end_date
            
        if stock_list:
            placeholders = ', '.join([f":stock_{i}" for i in range(len(stock_list))])
            query += f" AND ts_code IN ({placeholders})"
            for i, stock in enumerate(stock_list):
                params[f'stock_{i}'] = stock
                
        query += " ORDER BY ts_code, end_date"
        
        try:
            with self.engine.connect() as conn:
                financial_data = pd.read_sql_query(text(query), conn, params=params)
                
            if financial_data.empty:
                logger.warning("未找到财务数据")
                return {
                    'total_records': 0,
                    'success_count': 0,
                    'failed_count': 0,
                    'success_rate': 0
                }
            
            logger.info(f"获取到 {len(financial_data)} 条财务数据")
            
            # 计算基本面因子
            factors_df = self.fundamental_calculator.calculate_all_fundamental_factors(financial_data)
            
            # 准备存储数据
            factor_columns = [
                'roe', 'roa', 'gross_margin', 'net_margin',
                'revenue_yoy', 'net_profit_yoy',
                'debt_to_equity', 'current_ratio', 'quick_ratio'
            ]
            
            # 选择存在的因子列
            available_factors = [col for col in factor_columns if col in factors_df.columns]
            
            if not available_factors:
                logger.warning("未计算出有效的基本面因子")
                return {
                    'total_records': len(financial_data),
                    'success_count': 0,
                    'failed_count': len(financial_data),
                    'success_rate': 0
                }
            
            # 准备插入数据
            insert_data = factors_df[['ts_code', 'end_date'] + available_factors].copy()
            
            # 删除包含NaN的行
            insert_data = insert_data.dropna(subset=available_factors, how='all')
            
            if insert_data.empty:
                logger.warning("计算后无有效因子数据")
                return {
                    'total_records': len(financial_data),
                    'success_count': 0,
                    'failed_count': len(financial_data),
                    'success_rate': 0
                }
            
            # 添加因子分类和计算日期
            factor_records = []
            for _, row in insert_data.iterrows():
                for factor_name in available_factors:
                    if not pd.isna(row[factor_name]):
                        factor_records.append({
                            'ts_code': row['ts_code'],
                            'trade_date': row['end_date'],  # 使用财报期末日期
                            'factor_name': factor_name,
                            'factor_value': row[factor_name],
                            'factor_category': self._get_factor_category(factor_name),
                            'created_at': pd.Timestamp.now()
                        })
            
            if not factor_records:
                logger.warning("未生成有效的因子记录")
                return {
                    'total_records': len(financial_data),
                    'success_count': 0,
                    'failed_count': len(financial_data),
                    'success_rate': 0
                }
            
            factor_df = pd.DataFrame(factor_records)
            
            # 存储到数据库
            with self.engine.connect() as conn:
                # 先删除已存在的数据
                if start_date and end_date:
                    delete_sql = """
                        DELETE FROM factor_library 
                        WHERE factor_category IN ('profitability', 'growth', 'leverage', 'liquidity')
                        AND trade_date BETWEEN :start_date AND :end_date
                    """
                    conn.execute(text(delete_sql), {
                        'start_date': start_date,
                        'end_date': end_date
                    })
                
                # 插入新数据
                factor_df.to_sql(
                    'factor_library', 
                    conn, 
                    if_exists='append', 
                    index=False,
                    method='multi'
                )
                
                conn.commit()
                
            logger.info(f"基本面因子计算完成，存储 {len(factor_df)} 条记录")
            
            return {
                'total_records': len(financial_data),
                'success_count': len(factor_df),
                'failed_count': len(financial_data) - len(insert_data),
                'success_rate': len(insert_data) / len(financial_data) if financial_data else 0
            }
            
        except Exception as e:
            logger.error(f"基本面因子计算失败: {str(e)}")
            raise
    
    def _get_factor_category(self, factor_name: str) -> str:
        """获取因子分类
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子分类
        """
        profitability_factors = ['roe', 'roa', 'gross_margin', 'net_margin']
        growth_factors = ['revenue_yoy', 'net_profit_yoy']
        leverage_factors = ['debt_to_equity']
        liquidity_factors = ['current_ratio', 'quick_ratio']
        
        if factor_name in profitability_factors:
            return 'profitability'
        elif factor_name in growth_factors:
            return 'growth'
        elif factor_name in leverage_factors:
            return 'leverage'
        elif factor_name in liquidity_factors:
            return 'liquidity'
        else:
            return 'other'
    
    def calculate_all_factors_with_fundamental(self, 
                                             start_date: Optional[str] = None,
                                             end_date: Optional[str] = None,
                                             stock_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """计算所有因子（包括技术因子和基本面因子）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 指定股票列表
            
        Returns:
            计算结果统计
        """
        logger.info("开始计算所有因子（技术因子 + 基本面因子）")
        
        # 计算技术因子
        technical_result = self.calculate_all_factors(start_date, end_date, stock_list)
        
        # 计算基本面因子
        fundamental_result = self._calculate_fundamental_factors(start_date, end_date, stock_list)
        
        # 合并结果
        total_result = {
            'technical_factors': technical_result,
            'fundamental_factors': fundamental_result,
            'total_success_count': technical_result['success_count'] + fundamental_result['success_count'],
            'total_failed_count': technical_result['failed_count'] + fundamental_result['failed_count']
        }
        
        logger.info(f"所有因子计算完成: 技术因子成功 {technical_result['success_count']} 只, 基本面因子成功 {fundamental_result['success_count']} 条")
        
        return total_result
    
    def get_factor_data(self, 
                       factor_names: List[str],
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       stock_list: Optional[List[str]] = None) -> pd.DataFrame:
        """获取因子数据
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 股票列表
            
        Returns:
            因子数据DataFrame
        """
        # 构建查询SQL
        factor_cols = ', '.join(factor_names)
        query = f"""
            SELECT ts_code, trade_date, {factor_cols}
            FROM stock_technical_factors
            WHERE 1=1
        """
        
        params = {}
        
        if start_date:
            query += " AND trade_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND trade_date <= :end_date"
            params['end_date'] = end_date
            
        if stock_list:
            placeholders = ', '.join([f":stock_{i}" for i in range(len(stock_list))])
            query += f" AND ts_code IN ({placeholders})"
            for i, stock in enumerate(stock_list):
                params[f'stock_{i}'] = stock
                
        query += " ORDER BY ts_code, trade_date"
        
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
            
        logger.info(f"获取因子数据 {len(df)} 条")
        return df
    
    def get_factor_statistics(self, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """获取因子统计信息
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            因子统计DataFrame
        """
        query = """
            SELECT 
                COUNT(DISTINCT ts_code) as stock_count,
                COUNT(DISTINCT trade_date) as date_count,
                COUNT(*) as total_records,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date
            FROM stock_technical_factors
            WHERE 1=1
        """
        
        params = {}
        
        if start_date:
            query += " AND trade_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND trade_date <= :end_date"
            params['end_date'] = end_date
            
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
            
        return df


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='因子计算引擎')
    parser.add_argument('action', choices=['create_tables', 'calculate', 'stats', 'test'],
                       help='执行的操作')
    parser.add_argument('--stock', type=str, help='指定股票代码')
    parser.add_argument('--start_date', type=str, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end_date', type=str, help='结束日期 (YYYYMMDD)')
    parser.add_argument('--limit', type=int, default=10, help='限制股票数量（测试用）')
    
    args = parser.parse_args()
    
    engine = FactorEngine()
    
    if args.action == 'create_tables':
        engine.create_factor_tables()
        print("因子表创建完成")
        
    elif args.action == 'calculate':
        if args.stock:
            # 计算单只股票
            success = engine.calculate_stock_factors(args.stock, args.start_date, args.end_date)
            print(f"股票 {args.stock} 因子计算{'成功' if success else '失败'}")
        else:
            # 计算所有股票（测试模式下限制数量）
            stocks = engine.get_all_stocks()
            if args.limit and args.limit < len(stocks):
                stocks = stocks[:args.limit]
                print(f"测试模式：只计算前 {args.limit} 只股票")
                
            # 计算技术因子和基本面因子
            result = engine.calculate_all_factors_with_fundamental(args.start_date, args.end_date, stocks)
            print(f"计算完成: {result}")
            
    elif args.action == 'stats':
        stats = engine.get_factor_statistics(args.start_date, args.end_date)
        print("因子统计信息:")
        print(stats.to_string(index=False))
        
    elif args.action == 'test':
        # 测试单只股票的因子计算
        test_stock = '000001.SZ'
        print(f"测试股票 {test_stock} 的因子计算")
        
        success = engine.calculate_stock_factors(test_stock)
        if success:
            print("测试成功")
            
            # 获取计算结果
            factors = engine.get_factor_data(
                ['rsi_14', 'macd', 'bb_width', 'volume_ratio_20'],
                stock_list=[test_stock]
            )
            
            if not factors.empty:
                print("\n因子数据示例:")
                print(factors.tail(10).to_string(index=False))
            else:
                print("未找到因子数据")
        else:
            print("测试失败")