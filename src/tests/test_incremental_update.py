#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能增量更新引擎测试

测试缺失数据检测、任务调度、同步执行等功能
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

from src.data.incremental_update import IncrementalUpdateManager, SyncTask, SyncPriority


class TestIncrementalUpdateManager:
    """智能增量更新管理器测试类"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        engine = Mock()
        conn = Mock()
        engine.connect.return_value.__enter__.return_value = conn
        return engine, conn
    
    @pytest.fixture
    def manager(self, mock_engine):
        """创建管理器实例"""
        engine, conn = mock_engine
        
        with patch('src.data.incremental_update.get_db_engine', return_value=engine):
            manager = IncrementalUpdateManager()
            manager.engine = engine
            return manager
    
    def test_init_manager(self, manager):
        """测试管理器初始化"""
        assert manager is not None
        assert manager.max_workers > 0
        assert manager.check_interval > 0
        assert manager.batch_size > 0
        assert isinstance(manager.task_queue, list)
        assert isinstance(manager.running_tasks, dict)
    
    def test_get_trading_dates(self, manager, mock_engine):
        """测试获取交易日期"""
        engine, conn = mock_engine
        
        # 模拟数据库查询结果
        mock_dates = [
            (datetime(2024, 1, 2).date(),),
            (datetime(2024, 1, 3).date(),),
            (datetime(2024, 1, 4).date(),),
            (datetime(2024, 1, 5).date(),)
        ]
        conn.execute.return_value = mock_dates
        
        dates = manager._get_trading_dates('2024-01-02', '2024-01-05')
        
        assert len(dates) == 4
        assert dates[0] == '2024-01-02'
        assert dates[-1] == '2024-01-05'
    
    def test_generate_business_days(self, manager):
        """测试生成工作日列表"""
        # 测试一周的工作日（排除周末）
        business_days = manager._generate_business_days('2024-01-01', '2024-01-07')
        
        # 2024-01-01是周一，2024-01-07是周日
        # 应该包含周一到周五，共5天
        expected_days = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        assert business_days == expected_days
    
    def test_get_existing_dates(self, manager, mock_engine):
        """测试获取已存在数据日期"""
        engine, conn = mock_engine
        
        # 模拟数据库查询结果
        mock_dates = [
            (datetime(2024, 1, 2).date(),),
            (datetime(2024, 1, 4).date(),)
        ]
        conn.execute.return_value = mock_dates
        
        dates = manager._get_existing_dates('daily', 'tushare', '000001.SZ', '2024-01-01', '2024-01-05')
        
        assert len(dates) == 2
        assert '2024-01-02' in dates
        assert '2024-01-04' in dates
    
    def test_get_table_config(self, manager):
        """测试获取表配置"""
        # 测试已知数据类型
        config = manager._get_table_config('daily')
        assert config is not None
        assert config['table'] == 'stock_daily'
        assert config['date_column'] == 'trade_date'
        
        # 测试未知数据类型
        config = manager._get_table_config('unknown')
        assert config is None
    
    def test_get_missing_dates(self, manager):
        """测试获取缺失日期"""
        with patch.object(manager, '_get_trading_dates', return_value=['2024-01-01', '2024-01-02', '2024-01-03']), \
             patch.object(manager, '_get_existing_dates', return_value=['2024-01-01', '2024-01-03']):
            
            missing_dates = manager.get_missing_dates('daily', 'tushare')
            
            assert len(missing_dates) == 1
            assert '2024-01-02' in missing_dates
    
    def test_create_sync_task(self, manager):
        """测试创建同步任务"""
        task_id = manager.create_sync_task(
            'tushare', 'daily', '2024-01-01', ['000001.SZ', '000002.SZ'], SyncPriority.HIGH
        )
        
        assert task_id is not None
        assert len(manager.task_queue) == 1
        
        task = manager.task_queue[0]
        assert task.data_source == 'tushare'
        assert task.data_type == 'daily'
        assert task.target_date == '2024-01-01'
        assert task.priority == SyncPriority.HIGH
        assert len(task.stock_codes) == 2
    
    def test_determine_priority(self, manager):
        """测试确定同步优先级"""
        # 测试最近日期的高优先级数据
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        priority = manager._determine_priority('daily', yesterday)
        assert priority == SyncPriority.URGENT
        
        # 测试普通日期的普通优先级数据
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        priority = manager._determine_priority('news_sentiment', week_ago)
        assert priority == SyncPriority.NORMAL
        
        # 测试较旧日期的低优先级数据
        month_ago = (datetime.now() - timedelta(days=35)).strftime('%Y-%m-%d')
        priority = manager._determine_priority('popularity_ranking', month_ago)
        assert priority == SyncPriority.LOW
    
    def test_get_active_stocks(self, manager, mock_engine):
        """测试获取活跃股票列表"""
        engine, conn = mock_engine
        
        # 模拟数据库查询结果
        mock_stocks = [
            ('000001.SZ', '000001', '平安银行'),
            ('000002.SZ', '000002', '万科A'),
            ('600000.SH', '600000', '浦发银行')
        ]
        conn.execute.return_value = mock_stocks
        
        stocks = manager._get_active_stocks(limit=3)
        
        assert len(stocks) == 3
        assert stocks[0]['ts_code'] == '000001.SZ'
        assert stocks[0]['symbol'] == '000001'
        assert stocks[0]['name'] == '平安银行'
    
    def test_schedule_incremental_sync(self, manager):
        """测试调度增量同步"""
        with patch.object(manager, 'get_missing_dates', return_value=['2024-01-01', '2024-01-02']), \
             patch.object(manager, '_get_active_stocks', return_value=[
                 {'ts_code': '000001.SZ', 'symbol': '000001', 'name': '平安银行'},
                 {'ts_code': '000002.SZ', 'symbol': '000002', 'name': '万科A'}
             ]):
            
            created_tasks = manager.schedule_incremental_sync(
                data_sources=['tushare'],
                data_types=['daily'],
                days_back=7
            )
            
            assert 'tushare' in created_tasks
            assert len(created_tasks['tushare']) > 0
            assert len(manager.task_queue) > 0
    
    def test_execute_single_task_success(self, manager):
        """测试执行单个任务成功"""
        task = SyncTask(
            task_id='test_task',
            data_source='tushare',
            data_type='daily',
            target_date='2024-01-01',
            stock_codes=['000001.SZ'],
            priority=SyncPriority.NORMAL
        )
        
        with patch.object(manager, '_call_sync_method', return_value={'records_processed': 1}), \
             patch.object(manager, '_update_task_sync_status'):
            
            result = manager._execute_single_task(task)
            
            assert result['success'] is True
            assert result['task_id'] == 'test_task'
            assert result['records_processed'] == 1
    
    def test_execute_single_task_failure(self, manager):
        """测试执行单个任务失败"""
        task = SyncTask(
            task_id='test_task',
            data_source='tushare',
            data_type='daily',
            target_date='2024-01-01',
            stock_codes=['000001.SZ'],
            priority=SyncPriority.NORMAL
        )
        
        with patch.object(manager, '_call_sync_method', side_effect=Exception('同步失败')), \
             patch.object(manager, '_update_task_sync_status'):
            
            result = manager._execute_single_task(task)
            
            assert result['success'] is False
            assert result['task_id'] == 'test_task'
            assert '同步失败' in result['error']
    
    def test_call_sync_method(self, manager):
        """测试调用同步方法"""
        task = SyncTask(
            task_id='test_task',
            data_source='tushare',
            data_type='daily',
            target_date='2024-01-01',
            stock_codes=['000001.SZ', '000002.SZ'],
            priority=SyncPriority.NORMAL
        )
        
        # 测试模拟同步方法
        result = manager._call_sync_method(task)
        
        # 由于是模拟方法，应该返回成功结果
        assert result['success'] is True
        assert result['records_processed'] == 2
    
    def test_get_sync_status_summary(self, manager, mock_engine):
        """测试获取同步状态摘要"""
        engine, conn = mock_engine
        
        # 模拟数据库查询结果
        mock_sync_status = [
            ('tushare', 'daily', 'success', datetime.now(), 1000, 0),
            ('akshare', 'news_sentiment', 'running', datetime.now(), 500, 5)
        ]
        
        # 模拟多次查询调用
        conn.execute.side_effect = [
            mock_sync_status,  # 同步状态查询
            [],  # daily数据查询
            [],  # daily_basic数据查询  
            []   # news_sentiment数据查询
        ]
        
        with patch.object(manager, '_get_trading_dates', return_value=['2024-01-01', '2024-01-02']), \
             patch.object(manager, '_get_existing_dates', return_value=['2024-01-01']):
            
            summary = manager.get_sync_status_summary()
            
            assert 'queue_status' in summary
            assert 'recent_sync_status' in summary
            assert 'data_completeness' in summary
            
            # 检查队列状态
            assert summary['queue_status']['pending_tasks'] == 0
            assert summary['queue_status']['running_tasks'] == 0
            
            # 检查数据完整性
            assert 'daily' in summary['data_completeness']
    
    def test_task_priority_sorting(self, manager):
        """测试任务优先级排序"""
        # 创建不同优先级的任务
        manager.create_sync_task('tushare', 'daily', '2024-01-01', ['000001.SZ'], SyncPriority.LOW)
        manager.create_sync_task('tushare', 'daily', '2024-01-02', ['000001.SZ'], SyncPriority.URGENT)
        manager.create_sync_task('tushare', 'daily', '2024-01-03', ['000001.SZ'], SyncPriority.NORMAL)
        
        # 验证任务按优先级排序
        assert len(manager.task_queue) == 3
        assert manager.task_queue[0].priority == SyncPriority.URGENT
        assert manager.task_queue[1].priority == SyncPriority.NORMAL
        assert manager.task_queue[2].priority == SyncPriority.LOW


class TestSyncTask:
    """同步任务测试类"""
    
    def test_sync_task_creation(self):
        """测试同步任务创建"""
        task = SyncTask(
            task_id='test_task',
            data_source='tushare',
            data_type='daily',
            target_date='2024-01-01',
            stock_codes=['000001.SZ', '000002.SZ'],
            priority=SyncPriority.HIGH
        )
        
        assert task.task_id == 'test_task'
        assert task.data_source == 'tushare'
        assert task.data_type == 'daily'
        assert task.target_date == '2024-01-01'
        assert len(task.stock_codes) == 2
        assert task.priority == SyncPriority.HIGH
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.created_at is not None


# 性能测试
class TestIncrementalUpdateManagerPerformance:
    """智能增量更新管理器性能测试"""
    
    @pytest.mark.performance
    def test_missing_dates_detection_performance(self):
        """测试缺失日期检测性能"""
        manager = IncrementalUpdateManager()
        
        # 模拟大量交易日期和已存在日期
        trading_dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d') 
                        for i in range(365)]
        existing_dates = [date for i, date in enumerate(trading_dates) if i % 3 != 0]  # 缺失1/3的数据
        
        with patch.object(manager, '_get_trading_dates', return_value=trading_dates), \
             patch.object(manager, '_get_existing_dates', return_value=existing_dates):
            
            import time
            start_time = time.time()
            
            missing_dates = manager.get_missing_dates('daily', 'tushare')
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 365天的数据检测应在0.1秒内完成
            assert duration < 0.1, f"缺失日期检测性能过慢: {duration:.3f}秒"
            assert len(missing_dates) > 0
    
    @pytest.mark.performance
    def test_task_creation_performance(self):
        """测试任务创建性能"""
        manager = IncrementalUpdateManager()
        
        # 准备大量股票代码
        stock_codes = [f'{i:06d}.SZ' for i in range(1000)]
        
        import time
        start_time = time.time()
        
        # 创建100个任务
        for i in range(100):
            manager.create_sync_task(
                'tushare', 'daily', f'2024-01-{(i % 30) + 1:02d}', 
                stock_codes[:10], SyncPriority.NORMAL
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 100个任务创建应在1秒内完成
        assert duration < 1.0, f"任务创建性能过慢: {duration:.2f}秒"
        assert len(manager.task_queue) == 100


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])