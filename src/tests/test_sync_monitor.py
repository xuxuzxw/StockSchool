#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步监控器测试

测试同步监控仪表板的各项功能
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.monitoring.sync_monitor import (
    SyncMonitor, SyncTaskInfo, SyncEvent, SyncTaskStatus, SyncEventType
)


class TestSyncMonitor:
    """同步监控器测试类"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        engine = Mock()
        conn = Mock()
        engine.connect.return_value.__enter__.return_value = conn
        engine.connect.return_value.__exit__.return_value = None
        return engine, conn
    
    @pytest.fixture
    def sync_monitor(self, mock_engine):
        """创建同步监控器实例"""
        engine, conn = mock_engine
        
        with patch('src.monitoring.sync_monitor.get_db_engine', return_value=engine):
            monitor = SyncMonitor()
            monitor.engine = engine
            return monitor, conn
    
    @pytest.fixture
    def sample_task_info(self):
        """创建示例任务信息"""
        return SyncTaskInfo(
            task_id="test_task_001",
            data_source="tushare",
            data_type="daily",
            target_date="2024-01-01",
            status=SyncTaskStatus.PENDING,
            priority=1
        )
    
    def test_sync_monitor_initialization(self, mock_engine):
        """测试同步监控器初始化"""
        engine, conn = mock_engine
        
        with patch('src.monitoring.sync_monitor.get_db_engine', return_value=engine):
            monitor = SyncMonitor()
            
            assert monitor.engine == engine
            assert monitor.monitoring_enabled is True
            assert len(monitor.active_tasks) == 0
            assert len(monitor.event_queue) == 0
            assert len(monitor.event_subscribers) == 0
    
    def test_start_task_monitoring(self, sync_monitor, sample_task_info):
        """测试开始任务监控"""
        monitor, conn = sync_monitor
        
        # 开始监控
        monitor.start_task_monitoring(sample_task_info)
        
        # 验证任务状态
        assert sample_task_info.task_id in monitor.active_tasks
        assert monitor.active_tasks[sample_task_info.task_id].status == SyncTaskStatus.RUNNING
        assert monitor.active_tasks[sample_task_info.task_id].start_time is not None
        
        # 验证数据库调用
        conn.execute.assert_called()
        conn.commit.assert_called()
        
        # 验证事件发送
        assert len(monitor.event_queue) > 0
        event = monitor.event_queue[-1]
        assert event.event_type == SyncEventType.TASK_STARTED
        assert event.task_id == sample_task_info.task_id
    
    def test_update_task_progress(self, sync_monitor, sample_task_info):
        """测试更新任务进度"""
        monitor, conn = sync_monitor
        
        # 先开始监控
        monitor.start_task_monitoring(sample_task_info)
        
        # 更新进度
        monitor.update_task_progress(
            task_id=sample_task_info.task_id,
            progress=50.0,
            records_processed=500,
            records_failed=10
        )
        
        # 验证任务状态更新
        task = monitor.active_tasks[sample_task_info.task_id]
        assert task.progress == 50.0
        assert task.records_processed == 500
        assert task.records_failed == 10
        
        # 验证进度事件
        progress_events = [e for e in monitor.event_queue if e.event_type == SyncEventType.TASK_PROGRESS]
        assert len(progress_events) > 0
        assert progress_events[-1].data['progress'] == 50.0
    
    def test_complete_task_monitoring_success(self, sync_monitor, sample_task_info):
        """测试成功完成任务监控"""
        monitor, conn = sync_monitor
        
        # 开始监控
        monitor.start_task_monitoring(sample_task_info)
        
        # 完成任务
        final_stats = {
            'records_processed': 1000,
            'records_failed': 5
        }
        monitor.complete_task_monitoring(
            task_id=sample_task_info.task_id,
            success=True,
            final_stats=final_stats
        )
        
        # 验证任务已从活跃列表移除
        assert sample_task_info.task_id not in monitor.active_tasks
        
        # 验证完成事件
        completion_events = [e for e in monitor.event_queue if e.event_type == SyncEventType.TASK_COMPLETED]
        assert len(completion_events) > 0
        assert completion_events[-1].data['success'] is True
        assert completion_events[-1].data['records_processed'] == 1000
    
    def test_complete_task_monitoring_failure(self, sync_monitor, sample_task_info):
        """测试失败完成任务监控"""
        monitor, conn = sync_monitor
        
        # 开始监控
        monitor.start_task_monitoring(sample_task_info)
        
        # 失败完成任务
        error_message = "数据库连接失败"
        monitor.complete_task_monitoring(
            task_id=sample_task_info.task_id,
            success=False,
            error_message=error_message
        )
        
        # 验证任务已从活跃列表移除
        assert sample_task_info.task_id not in monitor.active_tasks
        
        # 验证失败事件
        failure_events = [e for e in monitor.event_queue if e.event_type == SyncEventType.TASK_FAILED]
        assert len(failure_events) > 0
        assert failure_events[-1].data['success'] is False
        assert failure_events[-1].data['error_message'] == error_message
    
    def test_cancel_task_monitoring(self, sync_monitor, sample_task_info):
        """测试取消任务监控"""
        monitor, conn = sync_monitor
        
        # 开始监控
        monitor.start_task_monitoring(sample_task_info)
        
        # 取消任务
        reason = "用户手动取消"
        monitor.cancel_task_monitoring(
            task_id=sample_task_info.task_id,
            reason=reason
        )
        
        # 验证任务已从活跃列表移除
        assert sample_task_info.task_id not in monitor.active_tasks
        
        # 验证取消事件
        cancel_events = [e for e in monitor.event_queue if e.event_type == SyncEventType.TASK_CANCELLED]
        assert len(cancel_events) > 0
        assert cancel_events[-1].data['reason'] == reason
    
    def test_get_active_tasks(self, sync_monitor, sample_task_info):
        """测试获取活跃任务"""
        monitor, conn = sync_monitor
        
        # 开始监控
        monitor.start_task_monitoring(sample_task_info)
        
        # 获取活跃任务
        active_tasks = monitor.get_active_tasks()
        
        assert len(active_tasks) == 1
        assert active_tasks[0]['task_id'] == sample_task_info.task_id
        assert active_tasks[0]['status'] == SyncTaskStatus.RUNNING.value
    
    def test_get_real_time_dashboard(self, sync_monitor, sample_task_info):
        """测试获取实时仪表板"""
        monitor, conn = sync_monitor
        
        # 开始监控多个任务
        monitor.start_task_monitoring(sample_task_info)
        
        task_info_2 = SyncTaskInfo(
            task_id="test_task_002",
            data_source="akshare",
            data_type="sentiment",
            target_date="2024-01-01",
            status=SyncTaskStatus.PENDING,
            priority=2
        )
        monitor.start_task_monitoring(task_info_2)
        
        # 获取仪表板数据
        dashboard = monitor.get_real_time_dashboard()
        
        assert 'timestamp' in dashboard
        assert 'active_tasks' in dashboard
        assert 'summary' in dashboard
        assert 'data_sources' in dashboard
        assert 'data_types' in dashboard
        assert 'recent_events' in dashboard
        
        assert dashboard['summary']['total_active'] == 2
        assert dashboard['summary']['running_tasks'] == 2
        assert 'tushare' in dashboard['data_sources']
        assert 'akshare' in dashboard['data_sources']
    
    def test_event_subscription(self, sync_monitor, sample_task_info):
        """测试事件订阅"""
        monitor, conn = sync_monitor
        
        # 创建事件回调
        received_events = []
        
        def event_callback(event):
            received_events.append(event)
        
        # 订阅事件
        monitor.subscribe_events(event_callback)
        
        # 开始监控（会触发事件）
        monitor.start_task_monitoring(sample_task_info)
        
        # 验证事件接收
        assert len(received_events) > 0
        assert received_events[-1].event_type == SyncEventType.TASK_STARTED
        
        # 取消订阅
        monitor.unsubscribe_events(event_callback)
        
        # 更新进度（不应该再接收到事件）
        initial_count = len(received_events)
        monitor.update_task_progress(sample_task_info.task_id, 50.0)
        
        # 由于取消订阅，事件数量不应该增加
        # 注意：事件仍会添加到队列，但不会调用回调
        assert len(received_events) == initial_count
    
    def test_get_sync_history(self, sync_monitor):
        """测试获取同步历史记录"""
        monitor, conn = sync_monitor
        
        # 模拟数据库查询结果
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 10
        
        mock_history_result = Mock()
        mock_history_result.__iter__.return_value = iter([
            ('task_001', 'tushare', 'daily', datetime(2024, 1, 1).date(), 'completed',
             1, datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 5), 300.0, 100.0,
             1000, 0, None, 0, 3, datetime(2024, 1, 1, 9, 55), datetime(2024, 1, 1, 10, 5))
        ])
        
        conn.execute.side_effect = [mock_count_result, mock_history_result]
        
        # 获取历史记录
        history = monitor.get_sync_history(
            start_date='2024-01-01',
            end_date='2024-01-02',
            limit=10
        )
        
        assert 'records' in history
        assert 'pagination' in history
        assert 'filters' in history
        assert len(history['records']) == 1
        assert history['records'][0]['task_id'] == 'task_001'
        assert history['pagination']['total'] == 10
    
    def test_get_error_log_analysis(self, sync_monitor):
        """测试获取错误日志分析"""
        monitor, conn = sync_monitor
        
        # 模拟错误统计查询结果
        mock_stats_result = Mock()
        mock_stats_result.__iter__.return_value = iter([
            ('tushare', 'daily', 5, 3, 1.5)
        ])
        
        # 模拟错误分类查询结果
        mock_classification_result = Mock()
        mock_classification_result.__iter__.return_value = iter([
            ('Connection timeout', 3, 'tushare', 'daily', datetime(2024, 1, 1, 12, 0))
        ])
        
        # 模拟错误趋势查询结果
        mock_trend_result = Mock()
        mock_trend_result.__iter__.return_value = iter([
            (datetime(2024, 1, 1).date(), 5, 2)
        ])
        
        conn.execute.side_effect = [mock_stats_result, mock_classification_result, mock_trend_result]
        
        # 获取错误分析
        error_analysis = monitor.get_error_log_analysis(
            start_date='2024-01-01',
            end_date='2024-01-02'
        )
        
        assert 'error_statistics' in error_analysis
        assert 'error_classification' in error_analysis
        assert 'error_trend' in error_analysis
        assert len(error_analysis['error_statistics']) == 1
        assert error_analysis['error_statistics'][0]['data_source'] == 'tushare'
        assert len(error_analysis['error_classification']) == 1
        assert error_analysis['error_classification'][0]['error_category'] == 'network'
    
    def test_calculate_performance_metrics(self, sync_monitor):
        """测试计算性能指标"""
        monitor, conn = sync_monitor
        
        # 模拟性能指标查询结果
        mock_result = Mock()
        mock_result.fetchone.return_value = (
            100,  # total_tasks
            90,   # completed_tasks
            8,    # failed_tasks
            2,    # running_tasks
            150.5, # avg_duration
            30.0,  # min_duration
            300.0, # max_duration
            45.2,  # duration_stddev
            50000, # total_records_processed
            500,   # total_records_failed
            500.0, # avg_records_per_task
            15,    # tasks_with_retries
            0.8    # avg_retry_count
        )
        
        conn.execute.return_value = mock_result
        
        # 计算性能指标
        metrics = monitor.calculate_performance_metrics(
            start_date='2024-01-01 00:00:00',
            end_date='2024-01-01 23:59:59'
        )
        
        assert 'total_tasks' in metrics
        assert 'success_rate' in metrics
        assert 'failure_rate' in metrics
        assert 'avg_duration_seconds' in metrics
        assert 'throughput_tasks_per_hour' in metrics
        assert 'data_quality_rate' in metrics
        
        assert metrics['total_tasks'] == 100
        assert metrics['success_rate'] == 90.0
        assert metrics['failure_rate'] == 8.0
        assert metrics['avg_duration_seconds'] == 150.5
    
    def test_analyze_performance_trends(self, sync_monitor):
        """测试分析性能趋势"""
        monitor, conn = sync_monitor
        
        # 模拟趋势数据查询结果
        mock_result = Mock()
        mock_result.__iter__.return_value = iter([
            (datetime(2024, 1, 1, 10, 0), 20, 18, 2, 120.0, 1000, 3),
            (datetime(2024, 1, 1, 11, 0), 25, 22, 3, 130.0, 1200, 4),
            (datetime(2024, 1, 1, 12, 0), 30, 28, 2, 110.0, 1500, 2)
        ])
        
        conn.execute.return_value = mock_result
        
        # 分析趋势
        trend_analysis = monitor.analyze_performance_trends(days=1, granularity='hour')
        
        assert 'analysis_period' in trend_analysis
        assert 'trend_data' in trend_analysis
        assert 'trend_analysis' in trend_analysis
        
        assert len(trend_analysis['trend_data']) == 3
        assert trend_analysis['trend_data'][0]['total_tasks'] == 20
        assert trend_analysis['trend_data'][1]['success_rate'] == 88.0  # 22/25 * 100
    
    def test_generate_performance_forecast(self, sync_monitor):
        """测试生成性能预测"""
        monitor, conn = sync_monitor
        
        # 模拟历史趋势数据
        mock_result = Mock()
        mock_result.__iter__.return_value = iter([
            (datetime(2024, 1, 1, i, 0), 20 + i, 18 + i, 2, 120.0 - i, 1000 + i * 50, 1)
            for i in range(24)
        ])
        
        conn.execute.return_value = mock_result
        
        # 生成预测
        forecast = monitor.generate_performance_forecast(forecast_hours=6)
        
        assert 'forecast_generated_at' in forecast
        assert 'forecast_summary' in forecast
        assert 'forecast_data' in forecast
        assert 'historical_basis' in forecast
        
        assert len(forecast['forecast_data']) == 6
        assert 'predicted_success_rate' in forecast['forecast_data'][0]
        assert 'predicted_avg_duration' in forecast['forecast_data'][0]
        assert 'confidence_level' in forecast['forecast_data'][0]
    
    def test_export_performance_report(self, sync_monitor):
        """测试导出性能报告"""
        monitor, conn = sync_monitor
        
        # 模拟各种查询结果
        mock_results = [
            # performance_metrics查询
            Mock(fetchone=lambda: (100, 90, 8, 2, 150.5, 30.0, 300.0, 45.2, 50000, 500, 500.0, 15, 0.8)),
            # trend_analysis查询
            Mock(__iter__=lambda: iter([(datetime(2024, 1, 1, 10, 0), 20, 18, 2, 120.0, 1000, 3)])),
            # 其他查询...
        ]
        
        conn.execute.side_effect = mock_results * 10  # 确保有足够的mock结果
        
        # 导出JSON格式报告
        json_report = monitor.export_performance_report(
            start_date='2024-01-01',
            end_date='2024-01-02',
            format='json'
        )
        
        assert json_report['success'] is True
        assert json_report['export_format'] == 'json'
        assert 'data' in json_report
        assert 'report_metadata' in json_report['data']
        assert 'performance_metrics' in json_report['data']
    
    def test_get_performance_alerts(self, sync_monitor):
        """测试获取性能告警"""
        monitor, conn = sync_monitor
        
        # 模拟低成功率的性能指标
        mock_result = Mock()
        mock_result.fetchone.return_value = (
            100,  # total_tasks
            80,   # completed_tasks (低成功率)
            18,   # failed_tasks (高失败率)
            2,    # running_tasks
            400.0, # avg_duration (高耗时)
            30.0,  # min_duration
            600.0, # max_duration
            45.2,  # duration_stddev
            50000, # total_records_processed
            500,   # total_records_failed
            500.0, # avg_records_per_task
            25,    # tasks_with_retries (高重试率)
            1.5    # avg_retry_count
        )
        
        conn.execute.return_value = mock_result
        
        # 获取告警
        alerts = monitor.get_performance_alerts()
        
        # 应该有多个告警
        assert len(alerts) > 0
        
        # 检查告警类型
        alert_types = [alert['alert_type'] for alert in alerts]
        assert 'low_success_rate' in alert_types
        assert 'high_failure_rate' in alert_types
        assert 'high_duration' in alert_types
        assert 'high_retry_rate' in alert_types
        
        # 检查告警严重程度
        high_severity_alerts = [alert for alert in alerts if alert['severity'] == 'high']
        assert len(high_severity_alerts) > 0
    
    def test_cleanup_old_records(self, sync_monitor):
        """测试清理旧记录"""
        monitor, conn = sync_monitor
        
        # 模拟删除操作结果
        mock_task_result = Mock()
        mock_task_result.rowcount = 50
        
        mock_event_result = Mock()
        mock_event_result.rowcount = 200
        
        mock_perf_result = Mock()
        mock_perf_result.rowcount = 30
        
        conn.execute.side_effect = [mock_task_result, mock_event_result, mock_perf_result]
        
        # 执行清理
        cleanup_result = monitor.cleanup_old_records(retention_days=30)
        
        assert cleanup_result['success'] is True
        assert cleanup_result['cleaned_tasks'] == 50
        assert cleanup_result['cleaned_events'] == 200
        assert cleanup_result['cleaned_performance'] == 30
        assert cleanup_result['retention_days'] == 30
    
    def test_error_categorization(self, sync_monitor):
        """测试错误分类"""
        monitor, conn = sync_monitor
        
        # 测试各种错误类型的分类
        test_cases = [
            ('Connection timeout error', 'network'),
            ('API rate limit exceeded', 'api'),
            ('Database constraint violation', 'database'),
            ('Invalid JSON format', 'data_format'),
            ('Out of memory error', 'system_resource'),
            ('Unknown error occurred', 'other')
        ]
        
        for error_message, expected_category in test_cases:
            category = monitor._categorize_error(error_message)
            assert category == expected_category, f"错误消息 '{error_message}' 应该分类为 '{expected_category}'，但得到 '{category}'"


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])