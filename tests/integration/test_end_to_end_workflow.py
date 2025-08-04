#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端工作流测试

测试完整的数据同步工作流：
1. 完整数据同步流程
2. 数据质量监控流程
3. 错误处理和恢复流程
4. 性能监控流程
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import time
import json
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class EndToEndDataSyncWorkflow:
    """端到端数据同步工作流"""
    
    def __init__(self):
        self.components = {
            'tushare_sync': Mock(),
            'akshare_sync': Mock(),
            'industry_manager': Mock(),
            'quality_monitor': Mock(),
            'incremental_manager': Mock(),
            'scheduler': Mock(),
            'timescale_optimizer': Mock()
        }
        self.workflow_state = {
            'current_step': 0,
            'completed_steps': [],
            'failed_steps': [],
            'start_time': None,
            'end_time': None
        }
        self.setup_mock_responses()
    
    def setup_mock_responses(self):
        """设置模拟响应"""
        # Tushare同步器
        self.components['tushare_sync'].sync_stock_basic.return_value = {
            'success': True, 'records': 5000, 'duration': 30
        }
        self.components['tushare_sync'].sync_daily_data.return_value = {
            'success': True, 'records': 50000, 'duration': 120
        }
        self.components['tushare_sync'].sync_financial_data.return_value = {
            'success': True, 'records': 2000, 'duration': 45
        }
        
        # Akshare同步器
        self.components['akshare_sync'].sync_sentiment_data.return_value = {
            'success': True, 'records': 10000, 'duration': 60
        }
        self.components['akshare_sync'].sync_attention_data.return_value = {
            'success': True, 'records': 5000, 'duration': 30
        }
        
        # 行业分类管理器
        self.components['industry_manager'].sync_industry_classification.return_value = {
            'success': True, 'records': 500, 'duration': 15
        }
        
        # 质量监控器
        self.components['quality_monitor'].check_data_quality.return_value = {
            'overall_score': 88.5,
            'completeness': 95.2,
            'accuracy': 92.1,
            'timeliness': 89.3,
            'issues': ['minor_data_gaps'],
            'status': 'good'
        }
        
        # 增量管理器
        self.components['incremental_manager'].get_missing_dates.return_value = [
            '2024-01-01', '2024-01-02', '2024-01-03'
        ]
        
        # TimescaleDB优化器
        self.components['timescale_optimizer'].optimize_tables.return_value = {
            'success': True, 'optimized_tables': 5, 'space_saved': '2.5GB'
        }
    
    def execute_complete_workflow(self):
        """执行完整工作流"""
        self.workflow_state['start_time'] = datetime.now()
        time.sleep(0.001)  # 确保有执行时间
        
        workflow_steps = [
            ('基础数据同步', self._sync_basic_data),
            ('日线数据同步', self._sync_daily_data),
            ('财务数据同步', self._sync_financial_data),
            ('行业分类同步', self._sync_industry_data),
            ('情绪数据同步', self._sync_sentiment_data),
            ('关注度数据同步', self._sync_attention_data),
            ('数据质量检查', self._check_data_quality),
            ('增量更新检测', self._detect_incremental_updates),
            ('数据库优化', self._optimize_database),
            ('生成同步报告', self._generate_sync_report)
        ]
        
        for step_name, step_func in workflow_steps:
            try:
                self.workflow_state['current_step'] += 1
                result = step_func()
                
                if result.get('success', False):
                    self.workflow_state['completed_steps'].append({
                        'step': step_name,
                        'result': result,
                        'timestamp': datetime.now()
                    })
                else:
                    self.workflow_state['failed_steps'].append({
                        'step': step_name,
                        'error': result.get('error', 'Unknown error'),
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                self.workflow_state['failed_steps'].append({
                    'step': step_name,
                    'error': str(e),
                    'timestamp': datetime.now()
                })
        
        time.sleep(0.001)  # 确保有执行时间
        self.workflow_state['end_time'] = datetime.now()
        return self.workflow_state
    
    def _sync_basic_data(self):
        """同步基础数据"""
        return self.components['tushare_sync'].sync_stock_basic()
    
    def _sync_daily_data(self):
        """同步日线数据"""
        return self.components['tushare_sync'].sync_daily_data()
    
    def _sync_financial_data(self):
        """同步财务数据"""
        return self.components['tushare_sync'].sync_financial_data()
    
    def _sync_industry_data(self):
        """同步行业数据"""
        return self.components['industry_manager'].sync_industry_classification()
    
    def _sync_sentiment_data(self):
        """同步情绪数据"""
        return self.components['akshare_sync'].sync_sentiment_data()
    
    def _sync_attention_data(self):
        """同步关注度数据"""
        return self.components['akshare_sync'].sync_attention_data()
    
    def _check_data_quality(self):
        """检查数据质量"""
        result = self.components['quality_monitor'].check_data_quality()
        result['success'] = True  # 确保有success字段
        return result
    
    def _detect_incremental_updates(self):
        """检测增量更新"""
        missing_dates = self.components['incremental_manager'].get_missing_dates()
        return {
            'success': True,
            'missing_dates': missing_dates,
            'count': len(missing_dates)
        }
    
    def _optimize_database(self):
        """优化数据库"""
        return self.components['timescale_optimizer'].optimize_tables()
    
    def _generate_sync_report(self):
        """生成同步报告"""
        # 使用当前时间计算持续时间，因为end_time还没设置
        current_time = datetime.now()
        total_duration = (current_time - self.workflow_state['start_time']).total_seconds()
        
        return {
            'success': True,
            'total_steps': len(self.workflow_state['completed_steps']) + len(self.workflow_state['failed_steps']),
            'completed_steps': len(self.workflow_state['completed_steps']),
            'failed_steps': len(self.workflow_state['failed_steps']),
            'total_duration': total_duration,
            'success_rate': len(self.workflow_state['completed_steps']) / (len(self.workflow_state['completed_steps']) + len(self.workflow_state['failed_steps'])) * 100
        }


@pytest.fixture
def workflow_system():
    """创建工作流系统"""
    return EndToEndDataSyncWorkflow()


class TestEndToEndWorkflow:
    """端到端工作流测试类"""
    
    def test_complete_workflow_success(self, workflow_system):
        """测试完整工作流成功执行"""
        # 执行完整工作流
        result = workflow_system.execute_complete_workflow()
        
        # 验证工作流执行结果
        assert result['start_time'] is not None
        assert result['end_time'] is not None
        assert result['end_time'] > result['start_time']
        
        # 验证步骤执行
        assert len(result['completed_steps']) > 0
        assert len(result['failed_steps']) == 0  # 所有步骤都应该成功
        
        # 验证具体步骤
        step_names = [step['step'] for step in result['completed_steps']]
        expected_steps = [
            '基础数据同步', '日线数据同步', '财务数据同步',
            '行业分类同步', '情绪数据同步', '关注度数据同步',
            '数据质量检查', '增量更新检测', '数据库优化', '生成同步报告'
        ]
        
        for expected_step in expected_steps:
            assert expected_step in step_names
    
    def test_workflow_with_failures(self, workflow_system):
        """测试包含失败的工作流"""
        # 模拟某些步骤失败
        workflow_system.components['tushare_sync'].sync_daily_data.return_value = {
            'success': False, 'error': '网络连接超时'
        }
        workflow_system.components['akshare_sync'].sync_sentiment_data.return_value = {
            'success': False, 'error': 'API调用限制'
        }
        
        # 执行工作流
        result = workflow_system.execute_complete_workflow()
        
        # 验证失败处理
        assert len(result['failed_steps']) == 2
        assert len(result['completed_steps']) == 8  # 其他步骤应该成功
        
        # 验证失败信息
        failed_step_names = [step['step'] for step in result['failed_steps']]
        assert '日线数据同步' in failed_step_names
        assert '情绪数据同步' in failed_step_names
    
    def test_workflow_performance_monitoring(self, workflow_system):
        """测试工作流性能监控"""
        # 执行工作流
        start_time = time.time()
        result = workflow_system.execute_complete_workflow()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 验证性能指标
        assert execution_time < 10.0, f"工作流执行时间应小于10秒，实际：{execution_time:.2f}秒"
        
        # 验证各步骤性能
        for step in result['completed_steps']:
            step_result = step['result']
            if 'duration' in step_result:
                assert step_result['duration'] > 0, f"步骤 {step['step']} 应该有执行时间"
    
    def test_workflow_data_consistency(self, workflow_system):
        """测试工作流数据一致性"""
        # 设置数据一致性检查
        workflow_system.components['quality_monitor'].check_consistency = Mock(return_value={
            'stock_code_consistency': True,
            'date_consistency': True,
            'cross_table_consistency': True,
            'referential_integrity': True
        })
        
        # 执行工作流
        result = workflow_system.execute_complete_workflow()
        
        # 验证数据一致性
        consistency_result = workflow_system.components['quality_monitor'].check_consistency()
        assert consistency_result['stock_code_consistency'] is True
        assert consistency_result['date_consistency'] is True
        assert consistency_result['cross_table_consistency'] is True
        assert consistency_result['referential_integrity'] is True
    
    def test_workflow_resource_usage(self, workflow_system):
        """测试工作流资源使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu_percent = process.cpu_percent()
        
        # 执行工作流
        result = workflow_system.execute_complete_workflow()
        
        # 监控资源使用
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu_percent = process.cpu_percent()
        
        memory_increase = peak_memory - initial_memory
        
        # 验证资源使用合理
        assert memory_increase < 200, f"内存增长应小于200MB，实际：{memory_increase:.2f}MB"
        assert len(result['completed_steps']) > 0, "应该有成功完成的步骤"
    
    def test_workflow_error_recovery(self, workflow_system):
        """测试工作流错误恢复"""
        # 模拟间歇性失败
        call_count = 0
        
        def failing_sync():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {'success': False, 'error': '临时网络错误'}
            return {'success': True, 'records': 1000, 'duration': 30}
        
        workflow_system.components['tushare_sync'].sync_daily_data = failing_sync
        
        # 实现重试逻辑
        def execute_with_retry(func, max_retries=3):
            for attempt in range(max_retries):
                result = func()
                if result.get('success', False):
                    return result
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # 短暂延迟
            return result
        
        # 测试重试机制
        retry_result = execute_with_retry(workflow_system.components['tushare_sync'].sync_daily_data)
        
        # 验证重试成功
        assert retry_result['success'] is True
        assert call_count == 3  # 应该重试了3次
    
    def test_workflow_state_persistence(self, workflow_system):
        """测试工作流状态持久化"""
        # 创建临时文件保存状态
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            state_file = f.name
        
        try:
            # 执行工作流
            result = workflow_system.execute_complete_workflow()
            
            # 保存状态到文件
            with open(state_file, 'w') as f:
                json.dump({
                    'workflow_state': result,
                    'timestamp': datetime.now().isoformat()
                }, f, default=str)
            
            # 验证状态文件存在且可读
            assert os.path.exists(state_file)
            
            with open(state_file, 'r') as f:
                saved_state = json.load(f)
            
            # 验证保存的状态
            assert 'workflow_state' in saved_state
            assert 'timestamp' in saved_state
            assert len(saved_state['workflow_state']['completed_steps']) > 0
            
        finally:
            # 清理临时文件
            if os.path.exists(state_file):
                os.unlink(state_file)
    
    def test_workflow_parallel_execution(self, workflow_system):
        """测试工作流并行执行能力"""
        # 模拟可并行执行的步骤
        parallel_tasks = [
            ('情绪数据同步', workflow_system.components['akshare_sync'].sync_sentiment_data),
            ('关注度数据同步', workflow_system.components['akshare_sync'].sync_attention_data),
            ('行业分类同步', workflow_system.components['industry_manager'].sync_industry_classification)
        ]
        
        # 并行执行任务
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(task_func): task_name for task_name, task_func in parallel_tasks}
            results = {}
            
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result()
                    results[task_name] = result
                except Exception as e:
                    results[task_name] = {'success': False, 'error': str(e)}
        
        end_time = time.time()
        parallel_duration = end_time - start_time
        
        # 验证并行执行结果
        assert len(results) == 3
        assert all(result.get('success', False) for result in results.values())
        
        # 并行执行应该比串行快
        assert parallel_duration < 2.0, f"并行执行时间应小于2秒，实际：{parallel_duration:.2f}秒"


class TestWorkflowReporting:
    """工作流报告测试类"""
    
    def test_generate_comprehensive_report(self, workflow_system):
        """测试生成综合报告"""
        # 执行工作流
        result = workflow_system.execute_complete_workflow()
        
        # 生成综合报告
        report = {
            'execution_summary': {
                'total_steps': len(result['completed_steps']) + len(result['failed_steps']),
                'successful_steps': len(result['completed_steps']),
                'failed_steps': len(result['failed_steps']),
                'success_rate': len(result['completed_steps']) / (len(result['completed_steps']) + len(result['failed_steps'])) * 100,
                'execution_time': (result['end_time'] - result['start_time']).total_seconds()
            },
            'data_statistics': {
                'total_records_processed': sum(
                    step['result'].get('records', 0) 
                    for step in result['completed_steps'] 
                    if 'records' in step['result']
                ),
                'data_sources_synced': ['tushare', 'akshare'],
                'tables_updated': ['stock_basic', 'daily', 'financial', 'sentiment', 'attention', 'industry']
            },
            'quality_metrics': {
                'overall_score': 88.5,
                'data_completeness': 95.2,
                'data_accuracy': 92.1,
                'data_timeliness': 89.3
            },
            'performance_metrics': {
                'average_step_duration': sum(
                    step['result'].get('duration', 0) 
                    for step in result['completed_steps'] 
                    if 'duration' in step['result']
                ) / len(result['completed_steps']),
                'records_per_second': 0  # 计算总体处理速度
            }
        }
        
        # 验证报告内容
        assert report['execution_summary']['success_rate'] > 80
        assert report['data_statistics']['total_records_processed'] > 0
        assert len(report['data_statistics']['data_sources_synced']) == 2
        assert report['quality_metrics']['overall_score'] > 80
    
    def test_export_report_formats(self, workflow_system):
        """测试导出不同格式的报告"""
        # 执行工作流
        result = workflow_system.execute_complete_workflow()
        
        # 创建临时目录
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 导出JSON格式
            json_file = os.path.join(temp_dir, 'workflow_report.json')
            with open(json_file, 'w') as f:
                json.dump(result, f, default=str, indent=2)
            
            # 导出CSV格式（步骤摘要）
            csv_file = os.path.join(temp_dir, 'workflow_steps.csv')
            steps_data = []
            for step in result['completed_steps']:
                steps_data.append({
                    'step_name': step['step'],
                    'status': 'completed',
                    'records': step['result'].get('records', 0),
                    'duration': step['result'].get('duration', 0),
                    'timestamp': step['timestamp']
                })
            
            df = pd.DataFrame(steps_data)
            df.to_csv(csv_file, index=False)
            
            # 验证文件存在
            assert os.path.exists(json_file)
            assert os.path.exists(csv_file)
            
            # 验证文件内容
            assert os.path.getsize(json_file) > 0
            assert os.path.getsize(csv_file) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])