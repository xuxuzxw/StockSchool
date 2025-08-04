# -*- coding: utf-8 -*-
"""
AI策略系统自动化测试框架

实现单元测试、集成测试、性能测试等自动化测试功能
"""

import json
import logging
import os
import time
import unittest
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
import psutil
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, JSON, Float
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """测试用例"""
    test_id: str
    test_name: str
    test_type: str  # unit, integration, performance, ui
    test_category: str  # api, model, strategy, system
    description: str
    test_function: str
    test_data: Dict[str, Any]
    expected_result: Any
    timeout: int = 30
    retry_count: int = 0
    is_active: bool = True
    created_at: datetime = None

@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    test_name: str
    test_type: str
    status: str  # passed, failed, skipped, error
    execution_time: float
    error_message: str = None
    actual_result: Any = None
    performance_metrics: Dict[str, Any] = None
    executed_at: datetime = None

@dataclass
class TestSuite:
    """测试套件"""
    suite_id: str
    suite_name: str
    description: str
    test_cases: List[str]  # test_id列表
    execution_order: str = 'sequential'  # sequential, parallel
    setup_function: str = None
    teardown_function: str = None
    is_active: bool = True
    created_at: datetime = None

@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    error_rate: float
    concurrent_users: int

@dataclass
class TestReport:
    """测试报告"""
    report_id: str
    report_name: str
    test_suite_id: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    coverage_percentage: float
    performance_summary: Dict[str, Any]
    generated_at: datetime

class TestFramework:
    """自动化测试框架
    
    提供单元测试、集成测试、性能测试等自动化测试功能
    """
    
    def __init__(self, database_url: str = None, api_base_url: str = None):
        from ...utils.db import get_db_manager
        self.engine = get_db_manager().engine
        self.metadata = MetaData()
        self.api_base_url = api_base_url or 'http://localhost:5000/api'
        
        # 创建数据库表
        self._create_tables()
        
        # 初始化测试环境
        self.test_data_cache = {}
        self.performance_baseline = {}
        
        # 初始化默认测试用例
        self._init_default_test_cases()
    
    def _create_tables(self):
        """创建数据库表"""
        try:
            # 测试用例表
            test_cases = Table(
                'test_cases', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('test_id', String(100), nullable=False, unique=True),
                Column('test_name', String(200), nullable=False),
                Column('test_type', String(50), nullable=False),
                Column('test_category', String(50), nullable=False),
                Column('description', Text, nullable=False),
                Column('test_function', String(200), nullable=False),
                Column('test_data', JSON, nullable=True),
                Column('expected_result', JSON, nullable=True),
                Column('timeout', Integer, default=30),
                Column('retry_count', Integer, default=0),
                Column('is_active', Boolean, default=True),
                Column('created_at', DateTime, default=datetime.now),
                Column('updated_at', DateTime, default=datetime.now)
            )
            
            # 测试结果表
            test_results = Table(
                'test_results', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('test_id', String(100), nullable=False),
                Column('test_name', String(200), nullable=False),
                Column('test_type', String(50), nullable=False),
                Column('status', String(20), nullable=False),
                Column('execution_time', Float, nullable=False),
                Column('error_message', Text, nullable=True),
                Column('actual_result', JSON, nullable=True),
                Column('performance_metrics', JSON, nullable=True),
                Column('executed_at', DateTime, default=datetime.now)
            )
            
            # 测试套件表
            test_suites = Table(
                'test_suites', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('suite_id', String(100), nullable=False, unique=True),
                Column('suite_name', String(200), nullable=False),
                Column('description', Text, nullable=False),
                Column('test_cases', JSON, nullable=False),
                Column('execution_order', String(20), default='sequential'),
                Column('setup_function', String(200), nullable=True),
                Column('teardown_function', String(200), nullable=True),
                Column('is_active', Boolean, default=True),
                Column('created_at', DateTime, default=datetime.now),
                Column('updated_at', DateTime, default=datetime.now)
            )
            
            # 测试报告表
            test_reports = Table(
                'test_reports', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('report_id', String(100), nullable=False, unique=True),
                Column('report_name', String(200), nullable=False),
                Column('test_suite_id', String(100), nullable=False),
                Column('total_tests', Integer, nullable=False),
                Column('passed_tests', Integer, nullable=False),
                Column('failed_tests', Integer, nullable=False),
                Column('skipped_tests', Integer, nullable=False),
                Column('execution_time', Float, nullable=False),
                Column('coverage_percentage', Float, default=0.0),
                Column('performance_summary', JSON, nullable=True),
                Column('generated_at', DateTime, default=datetime.now)
            )
            
            # 性能基线表
            performance_baselines = Table(
                'performance_baselines', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('test_id', String(100), nullable=False),
                Column('metric_name', String(100), nullable=False),
                Column('baseline_value', Float, nullable=False),
                Column('threshold_min', Float, nullable=True),
                Column('threshold_max', Float, nullable=True),
                Column('created_at', DateTime, default=datetime.now),
                Column('updated_at', DateTime, default=datetime.now)
            )
            
            # 测试环境配置表
            test_environments = Table(
                'test_environments', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('env_id', String(100), nullable=False, unique=True),
                Column('env_name', String(200), nullable=False),
                Column('env_type', String(50), nullable=False),  # development, staging, production
                Column('api_base_url', String(500), nullable=False),
                Column('database_url', String(500), nullable=False),
                Column('config_data', JSON, nullable=True),
                Column('is_active', Boolean, default=True),
                Column('created_at', DateTime, default=datetime.now)
            )
            
            self.metadata.create_all(self.engine)
            logger.info("测试框架数据库表创建成功")
            
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise
    
    def _init_default_test_cases(self):
        """初始化默认测试用例"""
        try:
            # API测试用例
            api_test_cases = [
                TestCase(
                    test_id='api_health_check',
                    test_name='API健康检查',
                    test_type='integration',
                    test_category='api',
                    description='检查API服务是否正常运行',
                    test_function='test_api_health',
                    test_data={'endpoint': '/health'},
                    expected_result={'status': 'ok'}
                ),
                TestCase(
                    test_id='api_model_predict',
                    test_name='模型预测API测试',
                    test_type='integration',
                    test_category='api',
                    description='测试模型预测API功能',
                    test_function='test_model_predict_api',
                    test_data={
                        'endpoint': '/models/predict',
                        'payload': {
                            'model_id': 'test_model',
                            'features': [1.0, 2.0, 3.0, 4.0, 5.0]
                        }
                    },
                    expected_result={'prediction': 'number'}
                ),
                TestCase(
                    test_id='api_backtest',
                    test_name='回测API测试',
                    test_type='integration',
                    test_category='api',
                    description='测试策略回测API功能',
                    test_function='test_backtest_api',
                    test_data={
                        'endpoint': '/backtest/run',
                        'payload': {
                            'strategy_id': 'test_strategy',
                            'start_date': '2023-01-01',
                            'end_date': '2023-12-31',
                            'initial_capital': 100000
                        }
                    },
                    expected_result={'backtest_id': 'string'}
                )
            ]
            
            # 模型测试用例
            model_test_cases = [
                TestCase(
                    test_id='model_accuracy',
                    test_name='模型准确率测试',
                    test_type='unit',
                    test_category='model',
                    description='测试模型预测准确率',
                    test_function='test_model_accuracy',
                    test_data={'model_id': 'test_model', 'test_size': 1000},
                    expected_result={'accuracy': 0.8}
                ),
                TestCase(
                    test_id='model_performance',
                    test_name='模型性能测试',
                    test_type='performance',
                    test_category='model',
                    description='测试模型预测性能',
                    test_function='test_model_performance',
                    test_data={'model_id': 'test_model', 'batch_size': 100},
                    expected_result={'avg_response_time': 0.1}
                )
            ]
            
            # 策略测试用例
            strategy_test_cases = [
                TestCase(
                    test_id='strategy_returns',
                    test_name='策略收益率测试',
                    test_type='unit',
                    test_category='strategy',
                    description='测试策略历史收益率',
                    test_function='test_strategy_returns',
                    test_data={'strategy_id': 'test_strategy'},
                    expected_result={'annual_return': 0.15}
                ),
                TestCase(
                    test_id='strategy_risk',
                    test_name='策略风险测试',
                    test_type='unit',
                    test_category='strategy',
                    description='测试策略风险指标',
                    test_function='test_strategy_risk',
                    test_data={'strategy_id': 'test_strategy'},
                    expected_result={'max_drawdown': 0.1}
                )
            ]
            
            # 系统测试用例
            system_test_cases = [
                TestCase(
                    test_id='system_load',
                    test_name='系统负载测试',
                    test_type='performance',
                    test_category='system',
                    description='测试系统在高负载下的表现',
                    test_function='test_system_load',
                    test_data={'concurrent_users': 100, 'duration': 60},
                    expected_result={'avg_response_time': 1.0}
                ),
                TestCase(
                    test_id='database_connection',
                    test_name='数据库连接测试',
                    test_type='integration',
                    test_category='system',
                    description='测试数据库连接池',
                    test_function='test_database_connection',
                    test_data={'max_connections': 20},
                    expected_result={'connection_success': True}
                )
            ]
            
            # 保存所有测试用例
            all_test_cases = api_test_cases + model_test_cases + strategy_test_cases + system_test_cases
            for test_case in all_test_cases:
                self.save_test_case(test_case)
            
            # 创建默认测试套件
            default_suites = [
                TestSuite(
                    suite_id='api_test_suite',
                    suite_name='API测试套件',
                    description='包含所有API相关的测试用例',
                    test_cases=[tc.test_id for tc in api_test_cases]
                ),
                TestSuite(
                    suite_id='model_test_suite',
                    suite_name='模型测试套件',
                    description='包含所有模型相关的测试用例',
                    test_cases=[tc.test_id for tc in model_test_cases]
                ),
                TestSuite(
                    suite_id='full_test_suite',
                    suite_name='完整测试套件',
                    description='包含所有测试用例的完整套件',
                    test_cases=[tc.test_id for tc in all_test_cases],
                    execution_order='parallel'
                )
            ]
            
            for suite in default_suites:
                self.save_test_suite(suite)
            
        except Exception as e:
            logger.error(f"初始化默认测试用例失败: {e}")
    
    def run_test_case(self, test_id: str) -> TestResult:
        """运行单个测试用例
        
        Args:
            test_id: 测试用例ID
            
        Returns:
            测试结果
        """
        try:
            # 获取测试用例
            test_case = self.get_test_case(test_id)
            if not test_case:
                raise ValueError(f"测试用例不存在: {test_id}")
            
            start_time = time.time()
            
            try:
                # 执行测试
                actual_result = self._execute_test_function(
                    test_case.test_function, 
                    test_case.test_data
                )
                
                # 验证结果
                status = self._validate_result(
                    actual_result, 
                    test_case.expected_result
                )
                
                execution_time = time.time() - start_time
                
                # 收集性能指标
                performance_metrics = None
                if test_case.test_type == 'performance':
                    performance_metrics = self._collect_performance_metrics()
                
                result = TestResult(
                    test_id=test_case.test_id,
                    test_name=test_case.test_name,
                    test_type=test_case.test_type,
                    status=status,
                    execution_time=execution_time,
                    actual_result=actual_result,
                    performance_metrics=performance_metrics,
                    executed_at=datetime.now()
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = TestResult(
                    test_id=test_case.test_id,
                    test_name=test_case.test_name,
                    test_type=test_case.test_type,
                    status='error',
                    execution_time=execution_time,
                    error_message=str(e),
                    executed_at=datetime.now()
                )
            
            # 保存测试结果
            self.save_test_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"运行测试用例失败: {e}")
            raise
    
    def run_test_suite(self, suite_id: str) -> List[TestResult]:
        """运行测试套件
        
        Args:
            suite_id: 测试套件ID
            
        Returns:
            测试结果列表
        """
        try:
            # 获取测试套件
            test_suite = self.get_test_suite(suite_id)
            if not test_suite:
                raise ValueError(f"测试套件不存在: {suite_id}")
            
            results = []
            
            # 执行setup函数
            if test_suite.setup_function:
                self._execute_setup_teardown(test_suite.setup_function)
            
            try:
                if test_suite.execution_order == 'parallel':
                    # 并行执行
                    results = self._run_tests_parallel(test_suite.test_cases)
                else:
                    # 顺序执行
                    results = self._run_tests_sequential(test_suite.test_cases)
                
            finally:
                # 执行teardown函数
                if test_suite.teardown_function:
                    self._execute_setup_teardown(test_suite.teardown_function)
            
            # 生成测试报告
            report = self._generate_test_report(suite_id, results)
            self.save_test_report(report)
            
            return results
            
        except Exception as e:
            logger.error(f"运行测试套件失败: {e}")
            raise
    
    def run_performance_test(self, test_id: str, duration: int = 60, 
                           concurrent_users: int = 10) -> TestResult:
        """运行性能测试
        
        Args:
            test_id: 测试用例ID
            duration: 测试持续时间（秒）
            concurrent_users: 并发用户数
            
        Returns:
            测试结果
        """
        try:
            test_case = self.get_test_case(test_id)
            if not test_case:
                raise ValueError(f"测试用例不存在: {test_id}")
            
            start_time = time.time()
            results = []
            errors = []
            
            def worker():
                """工作线程函数"""
                worker_start = time.time()
                while time.time() - worker_start < duration:
                    try:
                        result = self._execute_test_function(
                            test_case.test_function,
                            test_case.test_data
                        )
                        results.append({
                            'timestamp': time.time(),
                            'response_time': time.time() - worker_start,
                            'success': True
                        })
                    except Exception as e:
                        errors.append({
                            'timestamp': time.time(),
                            'error': str(e)
                        })
                    
                    time.sleep(0.1)  # 避免过度负载
            
            # 启动并发线程
            threads = []
            for _ in range(concurrent_users):
                thread = threading.Thread(target=worker)
                thread.start()
                threads.append(thread)
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            execution_time = time.time() - start_time
            
            # 计算性能指标
            total_requests = len(results) + len(errors)
            success_requests = len(results)
            error_rate = len(errors) / total_requests if total_requests > 0 else 0
            
            response_times = [r['response_time'] for r in results]
            avg_response_time = np.mean(response_times) if response_times else 0
            throughput = success_requests / execution_time if execution_time > 0 else 0
            
            # 收集系统性能指标
            system_metrics = self._collect_performance_metrics()
            
            performance_metrics = {
                'total_requests': total_requests,
                'success_requests': success_requests,
                'error_rate': error_rate,
                'avg_response_time': avg_response_time,
                'throughput': throughput,
                'concurrent_users': concurrent_users,
                'duration': duration,
                'system_metrics': system_metrics
            }
            
            result = TestResult(
                test_id=test_case.test_id,
                test_name=f"{test_case.test_name} (性能测试)",
                test_type='performance',
                status='passed' if error_rate < 0.05 else 'failed',
                execution_time=execution_time,
                performance_metrics=performance_metrics,
                executed_at=datetime.now()
            )
            
            self.save_test_result(result)
            return result
            
        except Exception as e:
            logger.error(f"运行性能测试失败: {e}")
            raise
    
    def run_ui_test(self, test_id: str, browser: str = 'chrome') -> TestResult:
        """运行UI测试
        
        Args:
            test_id: 测试用例ID
            browser: 浏览器类型
            
        Returns:
            测试结果
        """
        try:
            test_case = self.get_test_case(test_id)
            if not test_case:
                raise ValueError(f"测试用例不存在: {test_id}")
            
            # 初始化WebDriver
            driver = self._init_webdriver(browser)
            
            start_time = time.time()
            
            try:
                # 执行UI测试
                actual_result = self._execute_ui_test(driver, test_case)
                
                # 验证结果
                status = self._validate_result(
                    actual_result,
                    test_case.expected_result
                )
                
                execution_time = time.time() - start_time
                
                result = TestResult(
                    test_id=test_case.test_id,
                    test_name=test_case.test_name,
                    test_type='ui',
                    status=status,
                    execution_time=execution_time,
                    actual_result=actual_result,
                    executed_at=datetime.now()
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = TestResult(
                    test_id=test_case.test_id,
                    test_name=test_case.test_name,
                    test_type='ui',
                    status='error',
                    execution_time=execution_time,
                    error_message=str(e),
                    executed_at=datetime.now()
                )
            
            finally:
                # 关闭浏览器
                driver.quit()
            
            self.save_test_result(result)
            return result
            
        except Exception as e:
            logger.error(f"运行UI测试失败: {e}")
            raise
    
    def generate_test_report(self, suite_id: str, output_path: str = None) -> str:
        """生成测试报告
        
        Args:
            suite_id: 测试套件ID
            output_path: 输出文件路径
            
        Returns:
            报告内容
        """
        try:
            # 获取测试结果
            results = self.get_test_results_by_suite(suite_id)
            
            # 统计信息
            total_tests = len(results)
            passed_tests = len([r for r in results if r.status == 'passed'])
            failed_tests = len([r for r in results if r.status == 'failed'])
            error_tests = len([r for r in results if r.status == 'error'])
            skipped_tests = len([r for r in results if r.status == 'skipped'])
            
            total_execution_time = sum(r.execution_time for r in results)
            
            # 生成HTML报告
            report_html = self._generate_html_report(
                suite_id, results, {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'error_tests': error_tests,
                    'skipped_tests': skipped_tests,
                    'total_execution_time': total_execution_time
                }
            )
            
            # 保存报告
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_html)
            
            return report_html
            
        except Exception as e:
            logger.error(f"生成测试报告失败: {e}")
            raise
    
    def schedule_test_execution(self, suite_id: str, schedule_time: datetime,
                              repeat_interval: int = None) -> str:
        """调度测试执行
        
        Args:
            suite_id: 测试套件ID
            schedule_time: 调度时间
            repeat_interval: 重复间隔（秒）
            
        Returns:
            调度任务ID
        """
        try:
            # 这里可以集成任务调度系统（如Celery）
            # 简化实现，直接返回任务ID
            task_id = f"task_{suite_id}_{int(time.time())}"
            
            # 保存调度信息到数据库
            schedule_sql = """
            INSERT INTO test_schedules (
                task_id, suite_id, schedule_time, repeat_interval, status
            ) VALUES (
                :task_id, :suite_id, :schedule_time, :repeat_interval, 'scheduled'
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(schedule_sql), {
                    'task_id': task_id,
                    'suite_id': suite_id,
                    'schedule_time': schedule_time,
                    'repeat_interval': repeat_interval
                })
                conn.commit()
            
            logger.info(f"测试调度创建成功: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"调度测试执行失败: {e}")
            raise
    
    # 保存和获取方法
    def save_test_case(self, test_case: TestCase):
        """保存测试用例"""
        try:
            # 检查是否已存在
            existing = self._get_test_case_exists(test_case.test_id)
            if existing:
                # 更新现有测试用例
                update_sql = """
                UPDATE test_cases
                SET test_name = :test_name, test_type = :test_type,
                    test_category = :test_category, description = :description,
                    test_function = :test_function, test_data = :test_data,
                    expected_result = :expected_result, timeout = :timeout,
                    retry_count = :retry_count, is_active = :is_active,
                    updated_at = :updated_at
                WHERE test_id = :test_id
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(update_sql), {
                        'test_id': test_case.test_id,
                        'test_name': test_case.test_name,
                        'test_type': test_case.test_type,
                        'test_category': test_case.test_category,
                        'description': test_case.description,
                        'test_function': test_case.test_function,
                        'test_data': json.dumps(test_case.test_data),
                        'expected_result': json.dumps(test_case.expected_result),
                        'timeout': test_case.timeout,
                        'retry_count': test_case.retry_count,
                        'is_active': test_case.is_active,
                        'updated_at': datetime.now()
                    })
                    conn.commit()
            else:
                # 插入新测试用例
                insert_sql = """
                INSERT INTO test_cases (
                    test_id, test_name, test_type, test_category, description,
                    test_function, test_data, expected_result, timeout,
                    retry_count, is_active
                ) VALUES (
                    :test_id, :test_name, :test_type, :test_category, :description,
                    :test_function, :test_data, :expected_result, :timeout,
                    :retry_count, :is_active
                )
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(insert_sql), {
                        'test_id': test_case.test_id,
                        'test_name': test_case.test_name,
                        'test_type': test_case.test_type,
                        'test_category': test_case.test_category,
                        'description': test_case.description,
                        'test_function': test_case.test_function,
                        'test_data': json.dumps(test_case.test_data),
                        'expected_result': json.dumps(test_case.expected_result),
                        'timeout': test_case.timeout,
                        'retry_count': test_case.retry_count,
                        'is_active': test_case.is_active
                    })
                    conn.commit()
            
        except Exception as e:
            logger.error(f"保存测试用例失败: {e}")
            raise
    
    def save_test_result(self, result: TestResult):
        """保存测试结果"""
        try:
            insert_sql = """
            INSERT INTO test_results (
                test_id, test_name, test_type, status, execution_time,
                error_message, actual_result, performance_metrics
            ) VALUES (
                :test_id, :test_name, :test_type, :status, :execution_time,
                :error_message, :actual_result, :performance_metrics
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'test_id': result.test_id,
                    'test_name': result.test_name,
                    'test_type': result.test_type,
                    'status': result.status,
                    'execution_time': result.execution_time,
                    'error_message': result.error_message,
                    'actual_result': json.dumps(result.actual_result) if result.actual_result else None,
                    'performance_metrics': json.dumps(result.performance_metrics) if result.performance_metrics else None
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
            raise
    
    def save_test_suite(self, suite: TestSuite):
        """保存测试套件"""
        try:
            # 检查是否已存在
            existing = self._get_test_suite_exists(suite.suite_id)
            if existing:
                # 更新现有测试套件
                update_sql = """
                UPDATE test_suites
                SET suite_name = :suite_name, description = :description,
                    test_cases = :test_cases, execution_order = :execution_order,
                    setup_function = :setup_function, teardown_function = :teardown_function,
                    is_active = :is_active, updated_at = :updated_at
                WHERE suite_id = :suite_id
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(update_sql), {
                        'suite_id': suite.suite_id,
                        'suite_name': suite.suite_name,
                        'description': suite.description,
                        'test_cases': json.dumps(suite.test_cases),
                        'execution_order': suite.execution_order,
                        'setup_function': suite.setup_function,
                        'teardown_function': suite.teardown_function,
                        'is_active': suite.is_active,
                        'updated_at': datetime.now()
                    })
                    conn.commit()
            else:
                # 插入新测试套件
                insert_sql = """
                INSERT INTO test_suites (
                    suite_id, suite_name, description, test_cases,
                    execution_order, setup_function, teardown_function, is_active
                ) VALUES (
                    :suite_id, :suite_name, :description, :test_cases,
                    :execution_order, :setup_function, :teardown_function, :is_active
                )
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(insert_sql), {
                        'suite_id': suite.suite_id,
                        'suite_name': suite.suite_name,
                        'description': suite.description,
                        'test_cases': json.dumps(suite.test_cases),
                        'execution_order': suite.execution_order,
                        'setup_function': suite.setup_function,
                        'teardown_function': suite.teardown_function,
                        'is_active': suite.is_active
                    })
                    conn.commit()
            
        except Exception as e:
            logger.error(f"保存测试套件失败: {e}")
            raise
    
    def save_test_report(self, report: TestReport):
        """保存测试报告"""
        try:
            insert_sql = """
            INSERT INTO test_reports (
                report_id, report_name, test_suite_id, total_tests,
                passed_tests, failed_tests, skipped_tests, execution_time,
                coverage_percentage, performance_summary
            ) VALUES (
                :report_id, :report_name, :test_suite_id, :total_tests,
                :passed_tests, :failed_tests, :skipped_tests, :execution_time,
                :coverage_percentage, :performance_summary
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'report_id': report.report_id,
                    'report_name': report.report_name,
                    'test_suite_id': report.test_suite_id,
                    'total_tests': report.total_tests,
                    'passed_tests': report.passed_tests,
                    'failed_tests': report.failed_tests,
                    'skipped_tests': report.skipped_tests,
                    'execution_time': report.execution_time,
                    'coverage_percentage': report.coverage_percentage,
                    'performance_summary': json.dumps(report.performance_summary) if report.performance_summary else None
                })
                conn.commit()
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")
            raise
    
    # 获取方法
    def get_test_case(self, test_id: str) -> Optional[TestCase]:
        """获取测试用例"""
        try:
            query_sql = """
            SELECT *
            FROM test_cases
            WHERE test_id = :test_id AND is_active = true
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'test_id': test_id})
                row = result.fetchone()
            
            if row:
                return TestCase(
                    test_id=row[1],
                    test_name=row[2],
                    test_type=row[3],
                    test_category=row[4],
                    description=row[5],
                    test_function=row[6],
                    test_data=json.loads(row[7]) if row[7] else {},
                    expected_result=json.loads(row[8]) if row[8] else None,
                    timeout=row[9],
                    retry_count=row[10],
                    is_active=row[11],
                    created_at=row[12]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取测试用例失败: {e}")
            return None
    
    def get_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """获取测试套件"""
        try:
            query_sql = """
            SELECT *
            FROM test_suites
            WHERE suite_id = :suite_id AND is_active = true
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'suite_id': suite_id})
                row = result.fetchone()
            
            if row:
                return TestSuite(
                    suite_id=row[1],
                    suite_name=row[2],
                    description=row[3],
                    test_cases=json.loads(row[4]) if row[4] else [],
                    execution_order=row[5],
                    setup_function=row[6],
                    teardown_function=row[7],
                    is_active=row[8],
                    created_at=row[9]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取测试套件失败: {e}")
            return None
    
    def get_test_results_by_suite(self, suite_id: str) -> List[TestResult]:
        """获取测试套件的所有结果"""
        try:
            # 先获取测试套件中的测试用例
            suite = self.get_test_suite(suite_id)
            if not suite:
                return []
            
            # 获取这些测试用例的最新结果
            results = []
            for test_id in suite.test_cases:
                query_sql = """
                SELECT *
                FROM test_results
                WHERE test_id = :test_id
                ORDER BY executed_at DESC
                LIMIT 1
                """
                
                with self.engine.connect() as conn:
                    result = conn.execute(text(query_sql), {'test_id': test_id})
                    row = result.fetchone()
                
                if row:
                    test_result = TestResult(
                        test_id=row[1],
                        test_name=row[2],
                        test_type=row[3],
                        status=row[4],
                        execution_time=row[5],
                        error_message=row[6],
                        actual_result=json.loads(row[7]) if row[7] else None,
                        performance_metrics=json.loads(row[8]) if row[8] else None,
                        executed_at=row[9]
                    )
                    results.append(test_result)
            
            return results
            
        except Exception as e:
            logger.error(f"获取测试结果失败: {e}")
            return []
    
    # 辅助方法
    def _execute_test_function(self, function_name: str, test_data: Dict[str, Any]) -> Any:
        """执行测试函数"""
        try:
            # 根据函数名调用相应的测试方法
            if hasattr(self, function_name):
                test_method = getattr(self, function_name)
                return test_method(test_data)
            else:
                raise ValueError(f"测试函数不存在: {function_name}")
            
        except Exception as e:
            logger.error(f"执行测试函数失败: {e}")
            raise
    
    def _validate_result(self, actual: Any, expected: Any) -> str:
        """验证测试结果"""
        try:
            if expected is None:
                return 'passed'  # 没有期望结果，只要不出错就算通过
            
            if isinstance(expected, dict):
                # 验证字典类型的期望结果
                for key, expected_value in expected.items():
                    if key not in actual:
                        return 'failed'
                    
                    if isinstance(expected_value, str):
                        # 类型检查
                        if expected_value == 'string' and not isinstance(actual[key], str):
                            return 'failed'
                        elif expected_value == 'number' and not isinstance(actual[key], (int, float)):
                            return 'failed'
                        elif expected_value == 'boolean' and not isinstance(actual[key], bool):
                            return 'failed'
                    else:
                        # 值比较
                        if actual[key] != expected_value:
                            return 'failed'
            else:
                # 直接比较
                if actual != expected:
                    return 'failed'
            
            return 'passed'
            
        except Exception as e:
            logger.error(f"验证测试结果失败: {e}")
            return 'error'
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        try:
            # 获取系统性能指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available': memory.available,
                'disk_usage': disk.percent,
                'disk_free': disk.free,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"收集性能指标失败: {e}")
            return {}
    
    def _run_tests_parallel(self, test_ids: List[str]) -> List[TestResult]:
        """并行运行测试"""
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有测试任务
            future_to_test = {executor.submit(self.run_test_case, test_id): test_id for test_id in test_ids}
            
            # 收集结果
            for future in as_completed(future_to_test):
                test_id = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"并行测试失败 {test_id}: {e}")
                    # 创建错误结果
                    error_result = TestResult(
                        test_id=test_id,
                        test_name=f"测试 {test_id}",
                        test_type='unknown',
                        status='error',
                        execution_time=0,
                        error_message=str(e),
                        executed_at=datetime.now()
                    )
                    results.append(error_result)
        
        return results
    
    def _run_tests_sequential(self, test_ids: List[str]) -> List[TestResult]:
        """顺序运行测试"""
        results = []
        
        for test_id in test_ids:
            try:
                result = self.run_test_case(test_id)
                results.append(result)
            except Exception as e:
                logger.error(f"顺序测试失败 {test_id}: {e}")
                # 创建错误结果
                error_result = TestResult(
                    test_id=test_id,
                    test_name=f"测试 {test_id}",
                    test_type='unknown',
                    status='error',
                    execution_time=0,
                    error_message=str(e),
                    executed_at=datetime.now()
                )
                results.append(error_result)
        
        return results
    
    def _execute_setup_teardown(self, function_name: str):
        """执行setup/teardown函数"""
        try:
            if hasattr(self, function_name):
                setup_method = getattr(self, function_name)
                setup_method()
            else:
                logger.warning(f"Setup/Teardown函数不存在: {function_name}")
        except Exception as e:
            logger.error(f"执行Setup/Teardown函数失败: {e}")
    
    def _generate_test_report(self, suite_id: str, results: List[TestResult]) -> TestReport:
        """生成测试报告"""
        try:
            total_tests = len(results)
            passed_tests = len([r for r in results if r.status == 'passed'])
            failed_tests = len([r for r in results if r.status == 'failed'])
            skipped_tests = len([r for r in results if r.status == 'skipped'])
            
            total_execution_time = sum(r.execution_time for r in results)
            
            # 计算性能摘要
            performance_results = [r for r in results if r.performance_metrics]
            performance_summary = {}
            if performance_results:
                avg_response_time = np.mean([r.performance_metrics.get('avg_response_time', 0) for r in performance_results])
                avg_throughput = np.mean([r.performance_metrics.get('throughput', 0) for r in performance_results])
                performance_summary = {
                    'avg_response_time': avg_response_time,
                    'avg_throughput': avg_throughput,
                    'performance_test_count': len(performance_results)
                }
            
            report = TestReport(
                report_id=f"report_{suite_id}_{int(time.time())}",
                report_name=f"测试报告 - {suite_id}",
                test_suite_id=suite_id,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                execution_time=total_execution_time,
                coverage_percentage=0.0,  # 需要代码覆盖率工具支持
                performance_summary=performance_summary,
                generated_at=datetime.now()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"生成测试报告失败: {e}")
            raise
    
    def _generate_html_report(self, suite_id: str, results: List[TestResult], 
                            summary: Dict[str, Any]) -> str:
        """生成HTML测试报告"""
        try:
            html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>测试报告 - {suite_id}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .summary table {{ border-collapse: collapse; width: 100%; }}
        .summary th, .summary td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .summary th {{ background-color: #f2f2f2; }}
        .results {{ margin: 20px 0; }}
        .test-case {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ background-color: #d4edda; }}
        .failed {{ background-color: #f8d7da; }}
        .error {{ background-color: #fff3cd; }}
        .skipped {{ background-color: #e2e3e5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>测试报告</h1>
        <p>测试套件: {suite_id}</p>
        <p>生成时间: {generated_at}</p>
    </div>
    
    <div class="summary">
        <h2>测试摘要</h2>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>总测试数</td><td>{total_tests}</td></tr>
            <tr><td>通过</td><td>{passed_tests}</td></tr>
            <tr><td>失败</td><td>{failed_tests}</td></tr>
            <tr><td>错误</td><td>{error_tests}</td></tr>
            <tr><td>跳过</td><td>{skipped_tests}</td></tr>
            <tr><td>总执行时间</td><td>{total_execution_time:.2f}秒</td></tr>
            <tr><td>成功率</td><td>{success_rate:.1f}%</td></tr>
        </table>
    </div>
    
    <div class="results">
        <h2>测试结果详情</h2>
        {test_results_html}
    </div>
</body>
</html>
            """
            
            # 生成测试结果HTML
            test_results_html = ""
            for result in results:
                status_class = result.status
                error_info = f"<p><strong>错误信息:</strong> {result.error_message}</p>" if result.error_message else ""
                
                test_results_html += f"""
                <div class="test-case {status_class}">
                    <h3>{result.test_name} ({result.test_id})</h3>
                    <p><strong>状态:</strong> {result.status}</p>
                    <p><strong>类型:</strong> {result.test_type}</p>
                    <p><strong>执行时间:</strong> {result.execution_time:.3f}秒</p>
                    {error_info}
                </div>
                """
            
            # 计算成功率
            success_rate = (summary['passed_tests'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
            
            # 填充模板
            html_content = html_template.format(
                suite_id=suite_id,
                generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                total_tests=summary['total_tests'],
                passed_tests=summary['passed_tests'],
                failed_tests=summary['failed_tests'],
                error_tests=summary['error_tests'],
                skipped_tests=summary['skipped_tests'],
                total_execution_time=summary['total_execution_time'],
                success_rate=success_rate,
                test_results_html=test_results_html
            )
            
            return html_content
            
        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
            raise
    
    def _init_webdriver(self, browser: str):
        """初始化WebDriver"""
        try:
            if browser.lower() == 'chrome':
                from selenium.webdriver.chrome.options import Options
                options = Options()
                options.add_argument('--headless')  # 无头模式
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                return webdriver.Chrome(options=options)
            elif browser.lower() == 'firefox':
                from selenium.webdriver.firefox.options import Options
                options = Options()
                options.add_argument('--headless')
                return webdriver.Firefox(options=options)
            else:
                raise ValueError(f"不支持的浏览器类型: {browser}")
                
        except Exception as e:
            logger.error(f"初始化WebDriver失败: {e}")
            raise
    
    def _execute_ui_test(self, driver, test_case: TestCase) -> Dict[str, Any]:
        """执行UI测试"""
        try:
            test_data = test_case.test_data
            url = test_data.get('url', 'http://localhost:5000')
            
            # 访问页面
            driver.get(url)
            
            # 等待页面加载
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # 执行测试步骤
            steps = test_data.get('steps', [])
            results = {}
            
            for step in steps:
                action = step.get('action')
                selector = step.get('selector')
                value = step.get('value')
                
                if action == 'click':
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    element.click()
                elif action == 'input':
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    element.clear()
                    element.send_keys(value)
                elif action == 'check_text':
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    actual_text = element.text
                    results[f"text_{selector}"] = actual_text
                elif action == 'check_exists':
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    results[f"exists_{selector}"] = len(elements) > 0
            
            return results
            
        except Exception as e:
            logger.error(f"执行UI测试失败: {e}")
            raise
    
    # 检查方法
    def _get_test_case_exists(self, test_id: str) -> bool:
        """检查测试用例是否存在"""
        try:
            query_sql = "SELECT COUNT(*) FROM test_cases WHERE test_id = :test_id"
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'test_id': test_id})
                count = result.scalar()
            return count > 0
        except:
            return False
    
    def _get_test_suite_exists(self, suite_id: str) -> bool:
        """检查测试套件是否存在"""
        try:
            query_sql = "SELECT COUNT(*) FROM test_suites WHERE suite_id = :suite_id"
            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {'suite_id': suite_id})
                count = result.scalar()
            return count > 0
        except:
            return False
    
    # 具体测试方法实现
    def test_api_health(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """API健康检查测试"""
        try:
            endpoint = test_data.get('endpoint', '/health')
            url = f"{self.api_base_url}{endpoint}"
            
            response = requests.get(url, timeout=10)
            
            return {
                'status_code': response.status_code,
                'status': 'ok' if response.status_code == 200 else 'error',
                'response_time': response.elapsed.total_seconds()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def test_model_predict_api(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """模型预测API测试"""
        try:
            endpoint = test_data.get('endpoint', '/models/predict')
            payload = test_data.get('payload', {})
            url = f"{self.api_base_url}{endpoint}"
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'status_code': response.status_code,
                    'prediction': result.get('prediction', 0),
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'status_code': response.status_code,
                    'error': response.text
                }
                
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def test_backtest_api(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """回测API测试"""
        try:
            endpoint = test_data.get('endpoint', '/backtest/run')
            payload = test_data.get('payload', {})
            url = f"{self.api_base_url}{endpoint}"
            
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'status_code': response.status_code,
                    'backtest_id': result.get('backtest_id', ''),
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'status_code': response.status_code,
                    'error': response.text
                }
                
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def test_model_accuracy(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """模型准确率测试"""
        try:
            model_id = test_data.get('model_id', 'test_model')
            test_size = test_data.get('test_size', 1000)
            
            # 模拟模型准确率测试
            # 实际实现中应该加载真实模型和测试数据
            accuracy = np.random.uniform(0.7, 0.95)  # 模拟准确率
            
            return {
                'model_id': model_id,
                'accuracy': accuracy,
                'test_size': test_size,
                'passed': accuracy >= 0.8
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def test_model_performance(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """模型性能测试"""
        try:
            model_id = test_data.get('model_id', 'test_model')
            batch_size = test_data.get('batch_size', 100)
            
            # 模拟模型性能测试
            start_time = time.time()
            
            # 模拟批量预测
            for _ in range(batch_size):
                # 模拟预测计算
                time.sleep(0.001)  # 模拟计算时间
            
            total_time = time.time() - start_time
            avg_response_time = total_time / batch_size
            
            return {
                'model_id': model_id,
                'batch_size': batch_size,
                'total_time': total_time,
                'avg_response_time': avg_response_time,
                'throughput': batch_size / total_time
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def test_strategy_returns(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """策略收益率测试"""
        try:
            strategy_id = test_data.get('strategy_id', 'test_strategy')
            
            # 模拟策略收益率计算
            # 实际实现中应该从数据库获取真实的策略回测结果
            annual_return = np.random.uniform(0.05, 0.25)  # 模拟年化收益率
            sharpe_ratio = np.random.uniform(0.8, 2.0)     # 模拟夏普比率
            max_drawdown = np.random.uniform(0.05, 0.15)   # 模拟最大回撤
            
            return {
                'strategy_id': strategy_id,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'passed': annual_return >= 0.1 and max_drawdown <= 0.2
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def test_strategy_risk(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """策略风险测试"""
        try:
            strategy_id = test_data.get('strategy_id', 'test_strategy')
            
            # 模拟策略风险指标计算
            volatility = np.random.uniform(0.1, 0.3)       # 模拟波动率
            max_drawdown = np.random.uniform(0.05, 0.15)   # 模拟最大回撤
            var_95 = np.random.uniform(0.02, 0.05)         # 模拟VaR
            
            return {
                'strategy_id': strategy_id,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'risk_level': 'low' if max_drawdown < 0.1 else 'medium' if max_drawdown < 0.15 else 'high'
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def test_system_load(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """系统负载测试"""
        try:
            concurrent_users = test_data.get('concurrent_users', 100)
            duration = test_data.get('duration', 60)
            
            # 模拟系统负载测试
            start_time = time.time()
            response_times = []
            errors = 0
            
            def load_worker():
                nonlocal errors
                worker_start = time.time()
                while time.time() - worker_start < duration:
                    try:
                        # 模拟API请求
                        request_start = time.time()
                        time.sleep(np.random.uniform(0.01, 0.1))  # 模拟响应时间
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                    except:
                        errors += 1
                    time.sleep(0.1)
            
            # 启动并发线程
            threads = []
            for _ in range(min(concurrent_users, 20)):  # 限制线程数
                thread = threading.Thread(target=load_worker)
                thread.start()
                threads.append(thread)
            
            # 等待完成
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            total_requests = len(response_times) + errors
            avg_response_time = np.mean(response_times) if response_times else 0
            error_rate = errors / total_requests if total_requests > 0 else 0
            
            return {
                'concurrent_users': concurrent_users,
                'duration': duration,
                'total_requests': total_requests,
                'avg_response_time': avg_response_time,
                'error_rate': error_rate,
                'throughput': total_requests / total_time,
                'passed': avg_response_time < 1.0 and error_rate < 0.05
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def test_database_connection(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """数据库连接测试"""
        try:
            max_connections = test_data.get('max_connections', 20)
            
            # 测试数据库连接池
            connections = []
            connection_times = []
            
            for i in range(max_connections):
                try:
                    start_time = time.time()
                    conn = self.engine.connect()
                    connection_time = time.time() - start_time
                    
                    # 执行简单查询测试连接
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
                    
                    connections.append(conn)
                    connection_times.append(connection_time)
                    
                except Exception as e:
                    logger.error(f"数据库连接失败 {i}: {e}")
                    break
            
            # 关闭所有连接
            for conn in connections:
                try:
                    conn.close()
                except:
                    pass
            
            successful_connections = len(connections)
            avg_connection_time = np.mean(connection_times) if connection_times else 0
            
            return {
                'max_connections': max_connections,
                'successful_connections': successful_connections,
                'connection_success': successful_connections == max_connections,
                'avg_connection_time': avg_connection_time,
                'connection_rate': successful_connections / max_connections
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }