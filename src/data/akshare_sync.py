#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AkShare数据同步模块

主要功能：
1. 同步市场情绪数据
2. 同步新闻舆情数据
3. 同步资金流向数据
4. 同步北向资金数据

作者: StockSchool Team
创建时间: 2024
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import time
from sqlalchemy import text
from ..database.connection import get_db_engine
from ..utils.retry import idempotent_retry
from ..utils.config_loader import config_loader


class AkShareSynchronizer:
    """AkShare数据同步器"""
    
    def __init__(self):
        """初始化同步器"""
        self.engine = get_db_engine()
        self.config = config_loader
        print("✅ AkShare同步器初始化成功")
    
    @idempotent_retry()
    def sync_market_sentiment(self, start_date=None, end_date=None):
        """同步市场情绪数据"""
        print("开始同步市场情绪数据...")
        try:
            if start_date is None:
                start_date = self.config.get('data_sync_params.sentiment_data.start_date', '20200101')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            sleep_interval = self.config.get('data_sync_params.sentiment_data.sleep_interval', 1.0)
            
            # 获取A股市场总貌数据
            print("正在获取A股市场总貌数据...")
            market_overview_df = ak.stock_zh_a_gdhs()
            if not market_overview_df.empty:
                # 添加日期字段
                market_overview_df['trade_date'] = datetime.now().date()
                market_overview_df['data_type'] = 'market_overview'
                
                # 存储到情绪数据表
                self._save_sentiment_data(market_overview_df, 'market_overview')
                print(f"✅ A股市场总貌数据同步完成，共 {len(market_overview_df)} 条记录")
            
            time.sleep(sleep_interval)
            
            # 获取沪深港通资金流向
            print("正在获取沪深港通资金流向数据...")
            try:
                hk_flow_df = ak.stock_hsgt_fund_flow_summary_em()
                if not hk_flow_df.empty:
                    hk_flow_df['trade_date'] = datetime.now().date()
                    hk_flow_df['data_type'] = 'hk_fund_flow'
                    
                    self._save_sentiment_data(hk_flow_df, 'hk_fund_flow')
                    print(f"✅ 沪深港通资金流向数据同步完成，共 {len(hk_flow_df)} 条记录")
            except Exception as e:
                print(f"⚠️ 沪深港通资金流向数据获取失败: {e}")
            
            time.sleep(sleep_interval)
            
            # 获取龙虎榜数据
            print("正在获取龙虎榜数据...")
            try:
                lhb_df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
                if not lhb_df.empty:
                    lhb_df['data_type'] = 'dragon_tiger_list'
                    
                    self._save_sentiment_data(lhb_df, 'dragon_tiger_list')
                    print(f"✅ 龙虎榜数据同步完成，共 {len(lhb_df)} 条记录")
            except Exception as e:
                print(f"⚠️ 龙虎榜数据获取失败: {e}")
            
            print("✅ 市场情绪数据同步完成")
            
        except Exception as e:
            print(f"❌ 市场情绪数据同步失败: {e}")
            raise
    
    @idempotent_retry()
    def sync_fund_flow_data(self, start_date=None):
        """同步资金流向数据"""
        print("开始同步资金流向数据...")
        try:
            if start_date is None:
                start_date = self.config.get('data_sync_params.fund_flow_data.start_date', '20200101')
            
            batch_size = self.config.get('data_sync_params.fund_flow_data.batch_size', 100)
            sleep_interval = self.config.get('data_sync_params.fund_flow_data.sleep_interval', 1.0)
            
            # 获取所有股票代码
            stocks_df = pd.read_sql(
                "SELECT ts_code, symbol FROM stock_basic WHERE list_status = 'L' LIMIT 100",  # 限制数量避免API限制
                self.engine
            )
            
            total_stocks = len(stocks_df)
            print(f"需要同步 {total_stocks} 只股票的资金流向数据")
            
            # 分批处理股票
            for i in range(0, total_stocks, batch_size):
                batch_stocks = stocks_df.iloc[i:i+batch_size]
                print(f"正在处理第 {i//batch_size + 1} 批，共 {len(batch_stocks)} 只股票")
                
                for _, stock in batch_stocks.iterrows():
                    symbol = stock['symbol']
                    ts_code = stock['ts_code']
                    
                    try:
                        # 获取个股资金流向数据
                        fund_flow_df = ak.stock_individual_fund_flow_rank(symbol=symbol)
                        
                        if not fund_flow_df.empty:
                            fund_flow_df['ts_code'] = ts_code
                            fund_flow_df['trade_date'] = datetime.now().date()
                            fund_flow_df['data_type'] = 'individual_fund_flow'
                            
                            self._save_sentiment_data(fund_flow_df, 'individual_fund_flow')
                            print(f"✅ {ts_code} 资金流向数据同步完成")
                        
                        time.sleep(sleep_interval)
                        
                    except Exception as e:
                        print(f"❌ 同步 {ts_code} 资金流向数据失败: {e}")
                        continue
                
                print(f"第 {i//batch_size + 1} 批处理完成")
            
            print("✅ 资金流向数据同步完成")
            
        except Exception as e:
            print(f"❌ 资金流向数据同步失败: {e}")
            raise
    
    @idempotent_retry()
    def sync_north_money_flow(self, start_date=None):
        """同步北向资金数据"""
        print("开始同步北向资金数据...")
        try:
            if start_date is None:
                start_date = self.config.get('data_sync_params.north_money_data.start_date', '20200101')
            
            sleep_interval = self.config.get('data_sync_params.north_money_data.sleep_interval', 2.0)
            
            # 获取北向资金流向数据
            print("正在获取北向资金流向数据...")
            north_money_df = ak.stock_hsgt_north_net_flow_in_em(start_date=start_date)
            
            if not north_money_df.empty:
                north_money_df['data_type'] = 'north_money_flow'
                
                self._save_sentiment_data(north_money_df, 'north_money_flow')
                print(f"✅ 北向资金流向数据同步完成，共 {len(north_money_df)} 条记录")
            
            time.sleep(sleep_interval)
            
            # 获取北向资金持股数据
            print("正在获取北向资金持股数据...")
            try:
                north_holding_df = ak.stock_hsgt_hold_stock_em()
                if not north_holding_df.empty:
                    north_holding_df['trade_date'] = datetime.now().date()
                    north_holding_df['data_type'] = 'north_money_holding'
                    
                    self._save_sentiment_data(north_holding_df, 'north_money_holding')
                    print(f"✅ 北向资金持股数据同步完成，共 {len(north_holding_df)} 条记录")
            except Exception as e:
                print(f"⚠️ 北向资金持股数据获取失败: {e}")
            
            print("✅ 北向资金数据同步完成")
            
        except Exception as e:
            print(f"❌ 北向资金数据同步失败: {e}")
            raise
    
    def _save_sentiment_data(self, df, data_category):
        """保存情绪数据到数据库"""
        try:
            # 将DataFrame转换为适合存储的格式
            records = []
            
            for _, row in df.iterrows():
                record = {
                    'trade_date': row.get('trade_date', datetime.now().date()),
                    'ts_code': row.get('ts_code', 'MARKET'),  # 如果没有股票代码，使用MARKET表示市场数据
                    'data_category': data_category,
                    'data_type': row.get('data_type', data_category),
                    'data_value': str(row.to_dict()),  # 将整行数据序列化为JSON字符串
                    'created_at': datetime.now()
                }
                records.append(record)
            
            # 批量插入数据
            if records:
                records_df = pd.DataFrame(records)
                records_df.to_sql(
                    'sentiment_data',
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
        except Exception as e:
            print(f"❌ 保存情绪数据失败: {e}")
            raise
    
    def full_sync(self):
        """完整同步所有情绪数据"""
        print("开始完整情绪数据同步...")
        
        # 1. 同步市场情绪数据
        self.sync_market_sentiment()
        
        # 2. 同步资金流向数据
        self.sync_fund_flow_data()
        
        # 3. 同步北向资金数据
        self.sync_north_money_flow()
        
        print("✅ 完整情绪数据同步完成")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AkShare情绪数据同步工具')
    parser.add_argument('--mode', choices=['sentiment', 'fund_flow', 'north_money', 'full'], 
                       default='sentiment', help='同步模式')
    parser.add_argument('--start-date', type=str,
                       help='开始日期（格式：YYYYMMDD）')
    parser.add_argument('--end-date', type=str,
                       help='结束日期（格式：YYYYMMDD）')
    
    args = parser.parse_args()
    
    try:
        # 创建同步器实例
        syncer = AkShareSynchronizer()
        
        # 根据模式执行不同的同步任务
        if args.mode == 'sentiment':
            syncer.sync_market_sentiment(start_date=args.start_date, end_date=args.end_date)
        elif args.mode == 'fund_flow':
            syncer.sync_fund_flow_data(start_date=args.start_date)
        elif args.mode == 'north_money':
            syncer.sync_north_money_flow(start_date=args.start_date)
        elif args.mode == 'full':
            syncer.full_sync()
            
    except Exception as e:
        print(f"❌ 同步过程中发生错误: {e}")
        raise