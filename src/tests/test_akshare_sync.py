#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Akshare同步器测试

测试新闻情绪、用户关注度、人气榜数据同步功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.akshare_sync import AkshareSynchronizer


class TestAkshareSynchronizer:
    """Akshare同步器测试类"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        engine = Mock()
        conn = Mock()
        engine.connect.return_value.__enter__.return_value = conn
        return engine, conn
    
    @pytest.fixture
    def syncer(self, mock_engine):
        """创建同步器实例"""
        engine, conn = mock_engine
        
        with patch('src.data.akshare_sync.get_db_engine', return_value=engine):
            syncer = AkshareSynchronizer()
            syncer.engine = engine
            return syncer
    
    def test_init_synchronizer(self, syncer):
        """测试同步器初始化"""
        assert syncer is not None
        assert syncer.api_limit > 0
        assert syncer.retry_times > 0
        assert syncer.retry_delay > 0
    
    def test_rate_limit_check(self, syncer):
        """测试API调用频率限制"""
        # 重置计数器
        syncer.call_count = 0
        syncer.window_start = datetime.now()
        
        # 测试正常调用
        syncer._rate_limit_check()
        assert syncer.call_count == 1
        
        # 测试达到限制
        syncer.call_count = syncer.api_limit
        start_time = datetime.now()
        syncer._rate_limit_check()
        end_time = datetime.now()
        
        # 应该重置计数器
        assert syncer.call_count == 1
    
    def test_get_active_stocks(self, syncer, mock_engine):
        """测试获取活跃股票列表"""
        engine, conn = mock_engine
        
        # 模拟数据库查询结果
        mock_result = [
            ('000001.SZ', '000001', '平安银行'),
            ('000002.SZ', '000002', '万科A'),
            ('600000.SH', '600000', '浦发银行')
        ]
        conn.execute.return_value = mock_result
        
        stocks = syncer._get_active_stocks(limit=3)
        
        assert len(stocks) == 3
        assert stocks[0]['ts_code'] == '000001.SZ'
        assert stocks[0]['symbol'] == '000001'
        assert stocks[0]['name'] == '平安银行'
    
    def test_analyze_sentiment(self, syncer):
        """测试情绪分析功能"""
        # 测试正面情绪
        positive_text = "股价上涨 利好消息 业绩增长"
        sentiment = syncer._analyze_sentiment(positive_text)
        assert sentiment > 0
        
        # 测试负面情绪
        negative_text = "股价下跌 利空消息 业绩亏损"
        sentiment = syncer._analyze_sentiment(negative_text)
        assert sentiment < 0
        
        # 测试中性情绪
        neutral_text = "公司发布公告"
        sentiment = syncer._analyze_sentiment(neutral_text)
        assert sentiment == 0
    
    def test_standardize_news_sentiment(self, syncer):
        """测试新闻情绪数据标准化"""
        # 准备测试数据
        news_data = [
            {
                'date': '2024-01-01',
                'title': '股价上涨利好消息',
                'content': '公司业绩增长',
                'source': 'test'
            },
            {
                'date': '2024-01-01',
                'title': '股价下跌利空消息',
                'content': '市场风险增加',
                'source': 'test'
            }
        ]
        
        result = syncer._standardize_news_sentiment(news_data, '000001.SZ')
        
        assert len(result) == 1  # 按日期聚合
        assert result[0]['ts_code'] == '000001.SZ'
        assert result[0]['news_volume'] == 2
        assert result[0]['positive_count'] + result[0]['negative_count'] + result[0]['neutral_count'] == 2
        assert -1 <= result[0]['sentiment_score'] <= 1
    
    def test_standardize_attention_data(self, syncer):
        """测试用户关注度数据标准化"""
        # 准备测试数据
        attention_data = [
            {
                'date': '2024-01-01',
                'search_volume': 1000,
                'discussion_volume': 500,
                'view_count': 5000,
                'comment_count': 100
            }
        ]
        
        result = syncer._standardize_attention_data(attention_data, '000001.SZ')
        
        assert len(result) == 1
        assert result[0]['ts_code'] == '000001.SZ'
        assert result[0]['attention_score'] > 0
        assert result[0]['search_volume'] == 1000
        assert result[0]['discussion_volume'] == 500
    
    def test_convert_symbol_to_ts_code(self, syncer):
        """测试股票代码转换"""
        # 测试深圳股票
        assert syncer._convert_symbol_to_ts_code('000001') == '000001.SZ'
        assert syncer._convert_symbol_to_ts_code('300001') == '300001.SZ'
        
        # 测试上海股票
        assert syncer._convert_symbol_to_ts_code('600000') == '600000.SH'
        
        # 测试空值
        assert syncer._convert_symbol_to_ts_code('') == ''
    
    @patch('src.data.akshare_sync.ak.stock_news_em')
    def test_fetch_stock_news_sentiment(self, mock_news_em, syncer):
        """测试获取股票新闻情绪数据"""
        # 模拟akshare返回数据
        mock_df = pd.DataFrame({
            '发布时间': ['2024-01-01 10:00:00', '2024-01-01 15:00:00'],
            '新闻标题': ['利好消息', '市场分析'],
            '新闻内容': ['公司业绩增长', '行业前景看好'],
            '信息来源': ['财经网', '证券报'],
            '新闻链接': ['http://test1.com', 'http://test2.com']
        })
        mock_news_em.return_value = mock_df
        
        result = syncer._fetch_stock_news_sentiment('000001', '2024-01-01', '2024-01-01')
        
        assert result is not None
        assert len(result) == 2
        assert result[0]['title'] == '利好消息'
        assert result[0]['source'] == 'akshare'
    
    def test_update_sync_status(self, syncer, mock_engine):
        """测试更新同步状态"""
        engine, conn = mock_engine
        
        syncer._update_sync_status(
            'news_sentiment', 'success',
            records_processed=100,
            records_failed=5,
            duration=60
        )
        
        # 验证SQL执行
        conn.execute.assert_called()
        conn.commit.assert_called()
    
    @patch('src.data.akshare_sync.ak.stock_hot_rank_em')
    def test_fetch_popularity_ranking_hot(self, mock_hot_rank, syncer):
        """测试获取热门股票排行榜"""
        # 模拟akshare返回数据
        mock_df = pd.DataFrame({
            '代码': ['000001', '000002', '600000'],
            '热度': [9500, 8800, 8200],
            '排名变化': [1, -1, 0]
        })
        mock_hot_rank.return_value = mock_df
        
        result = syncer._fetch_popularity_ranking('hot', '2024-01-01')
        
        assert result is not None
        assert len(result) == 3
        assert result[0]['ts_code'] == '000001.SZ'
        assert result[0]['rank_position'] == 1
        assert result[0]['popularity_score'] == 9500
    
    def test_full_sync_success(self, syncer):
        """测试完整同步成功"""
        with patch.object(syncer, 'sync_news_sentiment', return_value=True), \
             patch.object(syncer, 'sync_user_attention', return_value=True), \
             patch.object(syncer, 'sync_popularity_ranking', return_value=True):
            
            result = syncer.full_sync()
            
            assert result['news_sentiment'] is True
            assert result['user_attention'] is True
            assert result['popularity_ranking'] is True
    
    def test_full_sync_partial_failure(self, syncer):
        """测试完整同步部分失败"""
        with patch.object(syncer, 'sync_news_sentiment', return_value=True), \
             patch.object(syncer, 'sync_user_attention', return_value=False), \
             patch.object(syncer, 'sync_popularity_ranking', return_value=True):
            
            result = syncer.full_sync()
            
            assert result['news_sentiment'] is True
            assert result['user_attention'] is False
            assert result['popularity_ranking'] is True


# 性能测试
class TestAkshareSynchronizerPerformance:
    """Akshare同步器性能测试"""
    
    @pytest.mark.performance
    def test_sentiment_analysis_performance(self):
        """测试情绪分析性能"""
        syncer = AkshareSynchronizer()
        
        # 测试大量文本的情绪分析性能
        texts = ["股价上涨利好消息业绩增长" * 100] * 1000
        
        import time
        start_time = time.time()
        
        for text in texts:
            syncer._analyze_sentiment(text)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 1000条文本应在5秒内完成
        assert duration < 5.0, f"情绪分析性能过慢: {duration:.2f}秒"
    
    @pytest.mark.performance
    def test_data_standardization_performance(self):
        """测试数据标准化性能"""
        syncer = AkshareSynchronizer()
        
        # 准备大量测试数据
        news_data = []
        for i in range(1000):
            news_data.append({
                'date': f'2024-01-{(i % 30) + 1:02d}',
                'title': f'新闻标题{i}',
                'content': f'新闻内容{i}',
                'source': 'test'
            })
        
        import time
        start_time = time.time()
        
        result = syncer._standardize_news_sentiment(news_data, '000001.SZ')
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 1000条新闻应在3秒内完成标准化
        assert duration < 3.0, f"数据标准化性能过慢: {duration:.2f}秒"
        assert len(result) > 0


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])