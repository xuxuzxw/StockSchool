#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AkShare数据同步模块

主要功能：
1. 同步新闻情绪数据
2. 同步用户关注度数据  
3. 同步人气榜数据
4. 提供API调用频率限制和重试机制
5. 数据标准化和质量检查

作者: StockSchool Team
创建时间: 2024
更新时间: 2025-01-03
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import List, Dict, Optional, Any
from sqlalchemy import text, create_engine
from loguru import logger

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import get_db_engine
from utils.retry import idempotent_retry
from utils.config_loader import config


class AkshareSynchronizer:
    """
    Akshare数据同步器
    
    实现新闻情绪、用户关注度、人气榜等情绪面数据的同步
    提供API调用频率限制、重试机制和数据质量检查
    """
    
    def __init__(self):
        """初始化同步器"""
        self.engine = get_db_engine()
        self.config = config
        
        # API调用配置
        self.api_limit = self.config.get('data_sources.akshare.api_limit', 100)
        self.retry_times = self.config.get('data_sources.akshare.retry_times', 3)
        self.retry_delay = self.config.get('data_sources.akshare.retry_delay', 2)
        
        # 调用计数器和时间窗口
        self.call_count = 0
        self.window_start = datetime.now()
        self.window_duration = 60  # 1分钟窗口
        
        logger.info("✅ Akshare同步器初始化成功")
    
    def _rate_limit_check(self):
        """API调用频率限制检查"""
        current_time = datetime.now()
        
        # 如果超过时间窗口，重置计数器
        if (current_time - self.window_start).seconds >= self.window_duration:
            self.call_count = 0
            self.window_start = current_time
        
        # 如果达到调用限制，等待
        if self.call_count >= self.api_limit:
            wait_time = self.window_duration - (current_time - self.window_start).seconds
            if wait_time > 0:
                logger.warning(f"达到API调用限制，等待 {wait_time} 秒")
                time.sleep(wait_time)
                self.call_count = 0
                self.window_start = datetime.now()
        
        self.call_count += 1
    
    def _update_sync_status(self, data_type: str, status: str, 
                          records_processed: int = 0, records_failed: int = 0,
                          error_message: str = None, duration: int = None):
        """更新同步状态"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO sync_status (
                        data_source, data_type, last_sync_date, last_sync_time,
                        sync_status, records_processed, records_failed, 
                        error_message, sync_duration_seconds, updated_at
                    ) VALUES (
                        'akshare', :data_type, :sync_date, :sync_time,
                        :status, :processed, :failed, :error, :duration, :updated_at
                    )
                    ON CONFLICT (data_source, data_type) 
                    DO UPDATE SET
                        last_sync_date = :sync_date,
                        last_sync_time = :sync_time,
                        sync_status = :status,
                        records_processed = :processed,
                        records_failed = :failed,
                        error_message = :error,
                        sync_duration_seconds = :duration,
                        updated_at = :updated_at
                """), {
                    'data_type': data_type,
                    'sync_date': datetime.now().date(),
                    'sync_time': datetime.now(),
                    'status': status,
                    'processed': records_processed,
                    'failed': records_failed,
                    'error': error_message,
                    'duration': duration,
                    'updated_at': datetime.now()
                })
                conn.commit()
        except Exception as e:
            logger.error(f"更新同步状态失败: {e}")
    
    @idempotent_retry(max_retries=3)
    def sync_news_sentiment(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> bool:
        """
        同步新闻情绪数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            bool: 同步是否成功
        """
        logger.info("开始同步新闻情绪数据...")
        start_time = datetime.now()
        records_processed = 0
        records_failed = 0
        
        try:
            # 设置默认日期范围
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # 获取股票列表
            stock_list = self._get_active_stocks()
            logger.info(f"需要同步 {len(stock_list)} 只股票的新闻情绪数据")
            
            # 更新同步状态为运行中
            self._update_sync_status('news_sentiment', 'running')
            
            for stock in stock_list:
                ts_code = stock['ts_code']
                symbol = stock['symbol']
                
                try:
                    self._rate_limit_check()
                    
                    # 获取个股新闻数据 (使用akshare的新闻接口)
                    news_data = self._fetch_stock_news_sentiment(symbol, start_date, end_date)
                    
                    if news_data:
                        # 标准化情绪数据
                        standardized_data = self._standardize_news_sentiment(news_data, ts_code)
                        
                        # 保存到数据库
                        saved_count = self._save_news_sentiment(standardized_data)
                        records_processed += saved_count
                        
                        logger.debug(f"✅ {ts_code} 新闻情绪数据同步完成，保存 {saved_count} 条记录")
                    
                    # 控制请求频率
                    time.sleep(self.retry_delay)
                    
                except Exception as e:
                    records_failed += 1
                    logger.error(f"❌ 同步 {ts_code} 新闻情绪数据失败: {e}")
                    continue
            
            # 计算同步耗时
            duration = int((datetime.now() - start_time).total_seconds())
            
            # 更新同步状态为成功
            self._update_sync_status(
                'news_sentiment', 'success', 
                records_processed, records_failed, 
                duration=duration
            )
            
            logger.info(f"✅ 新闻情绪数据同步完成，处理 {records_processed} 条记录，失败 {records_failed} 条")
            return True
            
        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds())
            error_msg = str(e)
            
            # 更新同步状态为失败
            self._update_sync_status(
                'news_sentiment', 'failed',
                records_processed, records_failed,
                error_msg, duration
            )
            
            logger.error(f"❌ 新闻情绪数据同步失败: {e}")
            raise
    
    def _fetch_stock_news_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        获取个股新闻情绪数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            新闻情绪数据列表
        """
        try:
            # 使用akshare获取股票新闻数据
            # 注意：这里使用模拟数据，实际需要根据akshare的具体接口调整
            news_df = ak.stock_news_em(symbol=symbol)
            
            if news_df.empty:
                return None
            
            # 转换为字典列表
            news_list = []
            for _, row in news_df.iterrows():
                news_item = {
                    'date': row.get('发布时间', datetime.now().strftime('%Y-%m-%d')),
                    'title': row.get('新闻标题', ''),
                    'content': row.get('新闻内容', ''),
                    'source': row.get('信息来源', 'akshare'),
                    'url': row.get('新闻链接', '')
                }
                news_list.append(news_item)
            
            return news_list
            
        except Exception as e:
            logger.error(f"获取 {symbol} 新闻数据失败: {e}")
            return None
    
    def _standardize_news_sentiment(self, news_data: List[Dict], ts_code: str) -> List[Dict]:
        """
        标准化新闻情绪数据
        
        Args:
            news_data: 原始新闻数据
            ts_code: 股票代码
            
        Returns:
            标准化后的情绪数据
        """
        standardized_data = []
        
        # 按日期聚合新闻数据
        date_groups = {}
        for news in news_data:
            date_key = news['date'][:10]  # 取日期部分
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(news)
        
        # 计算每日情绪指标
        for date_str, daily_news in date_groups.items():
            try:
                # 简单的情绪分析（实际应用中可以使用更复杂的NLP模型）
                sentiment_scores = []
                positive_count = 0
                negative_count = 0
                neutral_count = 0
                
                for news in daily_news:
                    # 基于关键词的简单情绪分析
                    sentiment = self._analyze_sentiment(news['title'] + ' ' + news['content'])
                    sentiment_scores.append(sentiment)
                    
                    if sentiment > 0.1:
                        positive_count += 1
                    elif sentiment < -0.1:
                        negative_count += 1
                    else:
                        neutral_count += 1
                
                # 计算平均情绪分数
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
                
                # 确保情绪分数在-1到1范围内
                avg_sentiment = max(-1.0, min(1.0, avg_sentiment))
                
                standardized_item = {
                    'ts_code': ts_code,
                    'news_date': datetime.strptime(date_str, '%Y-%m-%d').date(),
                    'sentiment_score': round(avg_sentiment, 4),
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'news_volume': len(daily_news),
                    'source': 'akshare'
                }
                
                standardized_data.append(standardized_item)
                
            except Exception as e:
                logger.error(f"标准化 {date_str} 新闻情绪数据失败: {e}")
                continue
        
        return standardized_data
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        简单的情绪分析
        
        Args:
            text: 文本内容
            
        Returns:
            情绪分数 (-1到1)
        """
        # 简单的关键词情绪分析
        positive_keywords = ['上涨', '利好', '增长', '盈利', '突破', '创新', '合作', '收购', '扩张']
        negative_keywords = ['下跌', '利空', '亏损', '风险', '下滑', '减少', '裁员', '违规', '调查']
        
        text = text.lower()
        positive_score = sum(1 for keyword in positive_keywords if keyword in text)
        negative_score = sum(1 for keyword in negative_keywords if keyword in text)
        
        # 计算情绪分数
        total_keywords = positive_score + negative_score
        if total_keywords == 0:
            return 0.0
        
        sentiment = (positive_score - negative_score) / total_keywords
        return sentiment
    
    def _save_news_sentiment(self, sentiment_data: List[Dict]) -> int:
        """
        保存新闻情绪数据到数据库
        
        Args:
            sentiment_data: 情绪数据列表
            
        Returns:
            保存的记录数
        """
        if not sentiment_data:
            return 0
        
        try:
            df = pd.DataFrame(sentiment_data)
            
            # 使用UPSERT模式避免重复数据
            saved_count = 0
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    result = conn.execute(text("""
                        INSERT INTO news_sentiment (
                            ts_code, news_date, sentiment_score, positive_count,
                            negative_count, neutral_count, news_volume, source,
                            created_at, updated_at
                        ) VALUES (
                            :ts_code, :news_date, :sentiment_score, :positive_count,
                            :negative_count, :neutral_count, :news_volume, :source,
                            :created_at, :updated_at
                        )
                        ON CONFLICT (ts_code, news_date, source)
                        DO UPDATE SET
                            sentiment_score = :sentiment_score,
                            positive_count = :positive_count,
                            negative_count = :negative_count,
                            neutral_count = :neutral_count,
                            news_volume = :news_volume,
                            updated_at = :updated_at
                    """), {
                        'ts_code': row['ts_code'],
                        'news_date': row['news_date'],
                        'sentiment_score': row['sentiment_score'],
                        'positive_count': row['positive_count'],
                        'negative_count': row['negative_count'],
                        'neutral_count': row['neutral_count'],
                        'news_volume': row['news_volume'],
                        'source': row['source'],
                        'created_at': datetime.now(),
                        'updated_at': datetime.now()
                    })
                    saved_count += 1
                
                conn.commit()
            
            return saved_count
            
        except Exception as e:
            logger.error(f"保存新闻情绪数据失败: {e}")
            raise
    
    @idempotent_retry(max_retries=3)
    def sync_user_attention(self, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> bool:
        """
        同步用户关注度数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            bool: 同步是否成功
        """
        logger.info("开始同步用户关注度数据...")
        start_time = datetime.now()
        records_processed = 0
        records_failed = 0
        
        try:
            # 设置默认日期范围
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # 获取股票列表
            stock_list = self._get_active_stocks()
            logger.info(f"需要同步 {len(stock_list)} 只股票的用户关注度数据")
            
            # 更新同步状态为运行中
            self._update_sync_status('user_attention', 'running')
            
            for stock in stock_list:
                ts_code = stock['ts_code']
                symbol = stock['symbol']
                
                try:
                    self._rate_limit_check()
                    
                    # 获取用户关注度数据
                    attention_data = self._fetch_user_attention_data(symbol, start_date, end_date)
                    
                    if attention_data:
                        # 标准化关注度数据
                        standardized_data = self._standardize_attention_data(attention_data, ts_code)
                        
                        # 保存到数据库
                        saved_count = self._save_user_attention(standardized_data)
                        records_processed += saved_count
                        
                        logger.debug(f"✅ {ts_code} 用户关注度数据同步完成，保存 {saved_count} 条记录")
                    
                    # 控制请求频率
                    time.sleep(self.retry_delay)
                    
                except Exception as e:
                    records_failed += 1
                    logger.error(f"❌ 同步 {ts_code} 用户关注度数据失败: {e}")
                    continue
            
            # 计算同步耗时
            duration = int((datetime.now() - start_time).total_seconds())
            
            # 更新同步状态为成功
            self._update_sync_status(
                'user_attention', 'success',
                records_processed, records_failed,
                duration=duration
            )
            
            logger.info(f"✅ 用户关注度数据同步完成，处理 {records_processed} 条记录，失败 {records_failed} 条")
            return True
            
        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds())
            error_msg = str(e)
            
            # 更新同步状态为失败
            self._update_sync_status(
                'user_attention', 'failed',
                records_processed, records_failed,
                error_msg, duration
            )
            
            logger.error(f"❌ 用户关注度数据同步失败: {e}")
            raise
    
    def _fetch_user_attention_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        获取用户关注度数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            关注度数据列表
        """
        try:
            # 获取股票热度数据 (使用akshare的相关接口)
            # 注意：这里需要根据akshare的实际接口调整
            
            # 1. 获取股票搜索热度
            search_data = self._get_stock_search_volume(symbol)
            
            # 2. 获取股票讨论热度  
            discussion_data = self._get_stock_discussion_volume(symbol)
            
            # 3. 合并数据
            attention_data = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            while current_date <= end_dt:
                date_str = current_date.strftime('%Y-%m-%d')
                
                attention_item = {
                    'date': date_str,
                    'search_volume': search_data.get(date_str, 0),
                    'discussion_volume': discussion_data.get(date_str, 0),
                    'view_count': np.random.randint(1000, 10000),  # 模拟数据
                    'comment_count': np.random.randint(10, 1000)   # 模拟数据
                }
                
                attention_data.append(attention_item)
                current_date += timedelta(days=1)
            
            return attention_data
            
        except Exception as e:
            logger.error(f"获取 {symbol} 用户关注度数据失败: {e}")
            return None
    
    def _get_stock_search_volume(self, symbol: str) -> Dict[str, int]:
        """获取股票搜索量数据（模拟实现）"""
        # 实际实现中应该调用相应的API
        # 这里使用模拟数据
        search_data = {}
        for i in range(7):  # 最近7天
            date_str = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            search_data[date_str] = np.random.randint(100, 5000)
        return search_data
    
    def _get_stock_discussion_volume(self, symbol: str) -> Dict[str, int]:
        """获取股票讨论量数据（模拟实现）"""
        # 实际实现中应该调用相应的API
        # 这里使用模拟数据
        discussion_data = {}
        for i in range(7):  # 最近7天
            date_str = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            discussion_data[date_str] = np.random.randint(50, 2000)
        return discussion_data
    
    def _standardize_attention_data(self, attention_data: List[Dict], ts_code: str) -> List[Dict]:
        """
        标准化用户关注度数据
        
        Args:
            attention_data: 原始关注度数据
            ts_code: 股票代码
            
        Returns:
            标准化后的关注度数据
        """
        standardized_data = []
        
        for item in attention_data:
            try:
                # 计算综合关注度分数
                search_vol = item.get('search_volume', 0)
                discussion_vol = item.get('discussion_volume', 0)
                view_count = item.get('view_count', 0)
                comment_count = item.get('comment_count', 0)
                
                # 简单的加权计算关注度分数
                attention_score = (
                    search_vol * 0.4 + 
                    discussion_vol * 0.3 + 
                    view_count * 0.0002 + 
                    comment_count * 0.02
                )
                
                standardized_item = {
                    'ts_code': ts_code,
                    'attention_date': datetime.strptime(item['date'], '%Y-%m-%d').date(),
                    'attention_score': round(attention_score, 4),
                    'search_volume': search_vol,
                    'discussion_volume': discussion_vol,
                    'view_count': view_count,
                    'comment_count': comment_count,
                    'source': 'akshare'
                }
                
                standardized_data.append(standardized_item)
                
            except Exception as e:
                logger.error(f"标准化关注度数据失败: {e}")
                continue
        
        return standardized_data
    
    def _save_user_attention(self, attention_data: List[Dict]) -> int:
        """
        保存用户关注度数据到数据库
        
        Args:
            attention_data: 关注度数据列表
            
        Returns:
            保存的记录数
        """
        if not attention_data:
            return 0
        
        try:
            df = pd.DataFrame(attention_data)
            
            # 使用UPSERT模式避免重复数据
            saved_count = 0
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    result = conn.execute(text("""
                        INSERT INTO user_attention (
                            ts_code, attention_date, attention_score, search_volume,
                            discussion_volume, view_count, comment_count, source,
                            created_at, updated_at
                        ) VALUES (
                            :ts_code, :attention_date, :attention_score, :search_volume,
                            :discussion_volume, :view_count, :comment_count, :source,
                            :created_at, :updated_at
                        )
                        ON CONFLICT (ts_code, attention_date, source)
                        DO UPDATE SET
                            attention_score = :attention_score,
                            search_volume = :search_volume,
                            discussion_volume = :discussion_volume,
                            view_count = :view_count,
                            comment_count = :comment_count,
                            updated_at = :updated_at
                    """), {
                        'ts_code': row['ts_code'],
                        'attention_date': row['attention_date'],
                        'attention_score': row['attention_score'],
                        'search_volume': row['search_volume'],
                        'discussion_volume': row['discussion_volume'],
                        'view_count': row['view_count'],
                        'comment_count': row['comment_count'],
                        'source': row['source'],
                        'created_at': datetime.now(),
                        'updated_at': datetime.now()
                    })
                    saved_count += 1
                
                conn.commit()
            
            return saved_count
            
        except Exception as e:
            logger.error(f"保存用户关注度数据失败: {e}")
            raise
    
    @idempotent_retry(max_retries=3)
    def sync_popularity_ranking(self, date: Optional[str] = None) -> bool:
        """
        同步人气榜数据
        
        Args:
            date: 指定日期 (YYYY-MM-DD)，默认为今天
            
        Returns:
            bool: 同步是否成功
        """
        logger.info("开始同步人气榜数据...")
        start_time = datetime.now()
        records_processed = 0
        records_failed = 0
        
        try:
            # 设置默认日期
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # 更新同步状态为运行中
            self._update_sync_status('popularity_ranking', 'running')
            
            # 同步不同类型的人气榜
            ranking_types = ['hot', 'active', 'attention', 'volume']
            
            for ranking_type in ranking_types:
                try:
                    self._rate_limit_check()
                    
                    # 获取排行榜数据
                    ranking_data = self._fetch_popularity_ranking(ranking_type, date)
                    
                    if ranking_data:
                        # 标准化排行榜数据
                        standardized_data = self._standardize_ranking_data(ranking_data, ranking_type, date)
                        
                        # 保存到数据库
                        saved_count = self._save_popularity_ranking(standardized_data)
                        records_processed += saved_count
                        
                        logger.debug(f"✅ {ranking_type} 人气榜数据同步完成，保存 {saved_count} 条记录")
                    
                    # 控制请求频率
                    time.sleep(self.retry_delay)
                    
                except Exception as e:
                    records_failed += 1
                    logger.error(f"❌ 同步 {ranking_type} 人气榜数据失败: {e}")
                    continue
            
            # 计算同步耗时
            duration = int((datetime.now() - start_time).total_seconds())
            
            # 更新同步状态为成功
            self._update_sync_status(
                'popularity_ranking', 'success',
                records_processed, records_failed,
                duration=duration
            )
            
            logger.info(f"✅ 人气榜数据同步完成，处理 {records_processed} 条记录，失败 {records_failed} 条")
            return True
            
        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds())
            error_msg = str(e)
            
            # 更新同步状态为失败
            self._update_sync_status(
                'popularity_ranking', 'failed',
                records_processed, records_failed,
                error_msg, duration
            )
            
            logger.error(f"❌ 人气榜数据同步失败: {e}")
            raise
    
    def _fetch_popularity_ranking(self, ranking_type: str, date: str) -> Optional[List[Dict]]:
        """
        获取人气榜数据
        
        Args:
            ranking_type: 排行榜类型
            date: 日期
            
        Returns:
            排行榜数据列表
        """
        try:
            ranking_data = []
            
            if ranking_type == 'hot':
                # 获取热门股票排行榜
                hot_df = ak.stock_hot_rank_em()
                if not hot_df.empty:
                    for idx, row in hot_df.iterrows():
                        item = {
                            'ts_code': self._convert_symbol_to_ts_code(row.get('代码', '')),
                            'rank_position': idx + 1,
                            'popularity_score': row.get('热度', 0),
                            'change_from_previous': row.get('排名变化', 0)
                        }
                        ranking_data.append(item)
            
            elif ranking_type == 'active':
                # 获取活跃股票排行榜
                active_df = ak.stock_zh_a_spot_em()
                if not active_df.empty:
                    # 按成交量排序取前100
                    active_df = active_df.sort_values('成交量', ascending=False).head(100)
                    for idx, row in active_df.iterrows():
                        item = {
                            'ts_code': self._convert_symbol_to_ts_code(row.get('代码', '')),
                            'rank_position': idx + 1,
                            'popularity_score': row.get('成交量', 0) / 10000,  # 转换为万手
                            'change_from_previous': 0  # 暂时设为0
                        }
                        ranking_data.append(item)
            
            elif ranking_type == 'attention':
                # 获取关注度排行榜（模拟数据）
                stock_list = self._get_active_stocks(limit=50)
                for idx, stock in enumerate(stock_list):
                    item = {
                        'ts_code': stock['ts_code'],
                        'rank_position': idx + 1,
                        'popularity_score': np.random.uniform(1000, 10000),
                        'change_from_previous': np.random.randint(-10, 10)
                    }
                    ranking_data.append(item)
            
            elif ranking_type == 'volume':
                # 获取成交量排行榜
                volume_df = ak.stock_zh_a_spot_em()
                if not volume_df.empty:
                    # 按成交额排序取前100
                    volume_df = volume_df.sort_values('成交额', ascending=False).head(100)
                    for idx, row in volume_df.iterrows():
                        item = {
                            'ts_code': self._convert_symbol_to_ts_code(row.get('代码', '')),
                            'rank_position': idx + 1,
                            'popularity_score': row.get('成交额', 0) / 100000000,  # 转换为亿元
                            'change_from_previous': 0  # 暂时设为0
                        }
                        ranking_data.append(item)
            
            return ranking_data if ranking_data else None
            
        except Exception as e:
            logger.error(f"获取 {ranking_type} 人气榜数据失败: {e}")
            return None
    
    def _convert_symbol_to_ts_code(self, symbol: str) -> str:
        """
        将股票代码转换为ts_code格式
        
        Args:
            symbol: 股票代码
            
        Returns:
            ts_code格式的股票代码
        """
        if not symbol:
            return ''
        
        # 简单的转换逻辑
        if symbol.startswith('0') or symbol.startswith('3'):
            return f"{symbol}.SZ"
        elif symbol.startswith('6'):
            return f"{symbol}.SH"
        else:
            return symbol
    
    def _standardize_ranking_data(self, ranking_data: List[Dict], 
                                ranking_type: str, date: str) -> List[Dict]:
        """
        标准化人气榜数据
        
        Args:
            ranking_data: 原始排行榜数据
            ranking_type: 排行榜类型
            date: 日期
            
        Returns:
            标准化后的排行榜数据
        """
        standardized_data = []
        
        for item in ranking_data:
            try:
                standardized_item = {
                    'ts_code': item['ts_code'],
                    'ranking_date': datetime.strptime(date, '%Y-%m-%d').date(),
                    'ranking_type': ranking_type,
                    'rank_position': item['rank_position'],
                    'popularity_score': round(float(item['popularity_score']), 4),
                    'change_from_previous': item.get('change_from_previous', 0),
                    'source': 'akshare'
                }
                
                standardized_data.append(standardized_item)
                
            except Exception as e:
                logger.error(f"标准化排行榜数据失败: {e}")
                continue
        
        return standardized_data
    
    def _save_popularity_ranking(self, ranking_data: List[Dict]) -> int:
        """
        保存人气榜数据到数据库
        
        Args:
            ranking_data: 排行榜数据列表
            
        Returns:
            保存的记录数
        """
        if not ranking_data:
            return 0
        
        try:
            df = pd.DataFrame(ranking_data)
            
            # 使用UPSERT模式避免重复数据
            saved_count = 0
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    result = conn.execute(text("""
                        INSERT INTO popularity_ranking (
                            ts_code, ranking_date, ranking_type, rank_position,
                            popularity_score, change_from_previous, source,
                            created_at, updated_at
                        ) VALUES (
                            :ts_code, :ranking_date, :ranking_type, :rank_position,
                            :popularity_score, :change_from_previous, :source,
                            :created_at, :updated_at
                        )
                        ON CONFLICT (ts_code, ranking_date, ranking_type, source)
                        DO UPDATE SET
                            rank_position = :rank_position,
                            popularity_score = :popularity_score,
                            change_from_previous = :change_from_previous,
                            updated_at = :updated_at
                    """), {
                        'ts_code': row['ts_code'],
                        'ranking_date': row['ranking_date'],
                        'ranking_type': row['ranking_type'],
                        'rank_position': row['rank_position'],
                        'popularity_score': row['popularity_score'],
                        'change_from_previous': row['change_from_previous'],
                        'source': row['source'],
                        'created_at': datetime.now(),
                        'updated_at': datetime.now()
                    })
                    saved_count += 1
                
                conn.commit()
            
            return saved_count
            
        except Exception as e:
            logger.error(f"保存人气榜数据失败: {e}")
            raise
    
    def _get_active_stocks(self, limit: int = 100) -> List[Dict]:
        """
        获取活跃股票列表
        
        Args:
            limit: 限制数量
            
        Returns:
            股票列表
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT ts_code, symbol, name 
                    FROM stock_basic 
                    WHERE list_status = 'L' 
                    ORDER BY ts_code 
                    LIMIT :limit
                """), {'limit': limit})
                
                stocks = []
                for row in result:
                    stocks.append({
                        'ts_code': row[0],
                        'symbol': row[1],
                        'name': row[2]
                    })
                
                return stocks
                
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def full_sync(self, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> Dict[str, bool]:
        """
        完整同步所有Akshare情绪数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            各模块同步结果
        """
        logger.info("开始完整Akshare数据同步...")
        results = {}
        
        try:
            # 1. 同步新闻情绪数据
            logger.info("1/3 同步新闻情绪数据...")
            results['news_sentiment'] = self.sync_news_sentiment(start_date, end_date)
            
            # 2. 同步用户关注度数据
            logger.info("2/3 同步用户关注度数据...")
            results['user_attention'] = self.sync_user_attention(start_date, end_date)
            
            # 3. 同步人气榜数据
            logger.info("3/3 同步人气榜数据...")
            results['popularity_ranking'] = self.sync_popularity_ranking(end_date)
            
            success_count = sum(1 for result in results.values() if result)
            logger.info(f"✅ Akshare数据同步完成，成功 {success_count}/3 个模块")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Akshare数据同步失败: {e}")
            raise
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        获取同步状态
        
        Returns:
            同步状态信息
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT data_type, sync_status, last_sync_date, last_sync_time,
                           records_processed, records_failed, error_message,
                           sync_duration_seconds
                    FROM sync_status 
                    WHERE data_source = 'akshare'
                    ORDER BY last_sync_time DESC
                """))
                
                status_info = {}
                for row in result:
                    status_info[row[0]] = {
                        'status': row[1],
                        'last_sync_date': row[2].strftime('%Y-%m-%d') if row[2] else None,
                        'last_sync_time': row[3].strftime('%Y-%m-%d %H:%M:%S') if row[3] else None,
                        'records_processed': row[4],
                        'records_failed': row[5],
                        'error_message': row[6],
                        'duration_seconds': row[7]
                    }
                
                return status_info
                
        except Exception as e:
            logger.error(f"获取同步状态失败: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        清理旧数据
        
        Args:
            days_to_keep: 保留天数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            tables = ['news_sentiment', 'user_attention', 'popularity_ranking']
            
            with self.engine.connect() as conn:
                for table in tables:
                    if table == 'news_sentiment':
                        date_column = 'news_date'
                    elif table == 'user_attention':
                        date_column = 'attention_date'
                    else:
                        date_column = 'ranking_date'
                    
                    result = conn.execute(text(f"""
                        DELETE FROM {table} 
                        WHERE {date_column} < :cutoff_date
                    """), {'cutoff_date': cutoff_date.date()})
                    
                    deleted_count = result.rowcount
                    logger.info(f"清理 {table} 表旧数据，删除 {deleted_count} 条记录")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
            raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Akshare情绪数据同步工具')
    parser.add_argument('--mode', 
                       choices=['news', 'attention', 'ranking', 'full', 'status', 'cleanup'], 
                       default='full', help='同步模式')
    parser.add_argument('--start-date', type=str,
                       help='开始日期（格式：YYYY-MM-DD）')
    parser.add_argument('--end-date', type=str,
                       help='结束日期（格式：YYYY-MM-DD）')
    parser.add_argument('--cleanup-days', type=int, default=90,
                       help='清理数据时保留的天数')
    
    args = parser.parse_args()
    
    try:
        # 创建同步器实例
        syncer = AkshareSynchronizer()
        
        # 根据模式执行不同的同步任务
        if args.mode == 'news':
            syncer.sync_news_sentiment(start_date=args.start_date, end_date=args.end_date)
        elif args.mode == 'attention':
            syncer.sync_user_attention(start_date=args.start_date, end_date=args.end_date)
        elif args.mode == 'ranking':
            syncer.sync_popularity_ranking(date=args.end_date)
        elif args.mode == 'full':
            syncer.full_sync(start_date=args.start_date, end_date=args.end_date)
        elif args.mode == 'status':
            status = syncer.get_sync_status()
            print("=== Akshare同步状态 ===")
            for data_type, info in status.items():
                print(f"{data_type}: {info['status']} - {info['last_sync_time']}")
        elif args.mode == 'cleanup':
            syncer.cleanup_old_data(days_to_keep=args.cleanup_days)
            
    except Exception as e:
        logger.error(f"❌ 同步过程中发生错误: {e}")
        raise