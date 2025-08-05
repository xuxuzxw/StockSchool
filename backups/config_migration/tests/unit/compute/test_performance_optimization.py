#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化和存储管理测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.compute.parallel_factor_calculator import (
    ParallelFactorCalculator, ResourceMonitor, TaskLoadBalancer
)
from src.compute.factor_cache import (
    FactorCache, MemoryCache, RedisCache, CacheConfig
)
from src.compute.incremental_calculator import (
    IncrementalFactorCalculator, DependencyManager, IncrementalDataManager
)
from src.compute.data_compression_archiver import (
    DataCompressionArchiver, CompressionEngine, ArchiveManager, CompressionLevel
)


class TestResourceMonitor:
    """系统资源监控器测试"""
    
    def test_get_cpu_usage(self):
        """测试CPU使用率获取"""
        monitor = ResourceMonitor()
        
        cpu_usage = monitor.get_cpu_usage()
        
        assert isinstance(cpu_usage, float)
        assert 0 <= cpu_usage <= 100
    
    def test_get_memory_usage(self):
        """测试内存使用情况获取"""
        monitor = ResourceMonitor()
        
        memory_info = monitor.get_memory_usage()
        
        assert isinstance(memory_info, dict)
        expected_keys = ['rss_mb', 'vms_mb', 'percent', 'available_mb', 'total_mb']
        for key in expected_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float))
            assert memory_info[key] >= 0
    
    def test_check_resource_availability(self):
        """测试资源可用性检查"""
        monitor = ResourceMonitor()
        
        availability = monitor.check_resource_availability()
        
        assert isinstance(availability, dict)
        expected_keys = ['memory_available', 'cpu_available', 'overall_available']
        for key in expected_keys:
            assert key in availability
            assert isinstance(availability[key], bool)


class TestTaskLoadBalancer:
    """任务负载均衡器测试"""
    
    def test_calculate_optimal_workers(self):
        """测试最优工作进程数计算"""
        balancer = TaskLoadBalancer()
        
        # 测试不同任务数量
        optimal_workers = balancer.calculate_optimal_workers(100, 'medium')
        
        assert isinstance(optimal_workers, int)
        assert optimal_workers >= 1
        assert optimal_workers <= balancer.max_workers
    
    def test_split_tasks(self):
        """测试任务分割"""
        balancer = TaskLoadBalancer()
        
        tasks = list(range(100))
        chunks = balancer.split_tasks(tasks, chunk_size=10)
        
        assert isinstance(chunks, list)
        assert len(chunks) == 10
        assert all(len(chunk) == 10 for chunk in chunks)
        
        # 测试自动计算块大小
        auto_chunks = balancer.split_tasks(tasks)
        assert isinstance(auto_chunks, list)
        assert len(auto_chunks) > 0
    
    def test_balance_workload(self):
        """测试工作负载平衡"""
        balancer = TaskLoadBalancer()
        
        tasks = list(range(20))
        worker_capacities = [1.0, 1.5, 0.5, 2.0]
        
        assignments = balancer.balance_workload(tasks, worker_capacities)
        
        assert isinstance(assignments, list)
        assert len(assignments) == len(worker_capacities)
        
        # 检查所有任务都被分配
        total_assigned = sum(len(assignment) for assignment in assignments)
        assert total_assigned == len(tasks)


class TestMemoryCache:
    """内存缓存测试"""
    
    @pytest.fixture
    def memory_cache(self):
        """创建内存缓存实例"""
        return MemoryCache(max_size=10, default_ttl=60)
    
    def test_set_and_get(self, memory_cache):
        """测试设置和获取缓存"""
        key = "test_key"
        value = {"data": "test_value"}
        
        # 设置缓存
        success = memory_cache.set(key, value)
        assert success is True
        
        # 获取缓存
        cached_value = memory_cache.get(key)
        assert cached_value == value
    
    def test_cache_expiration(self, memory_cache):
        """测试缓存过期"""
        key = "expire_key"
        value = "expire_value"
        
        # 设置短期缓存
        memory_cache.set(key, value, ttl=1)
        
        # 立即获取应该成功
        assert memory_cache.get(key) == value
        
        # 等待过期后获取应该失败
        import time
        time.sleep(2)
        assert memory_cache.get(key) is None
    
    def test_lru_eviction(self, memory_cache):
        """测试LRU淘汰策略"""
        # 填满缓存
        for i in range(10):
            memory_cache.set(f"key_{i}", f"value_{i}")
        
        # 添加新项应该淘汰最旧的
        memory_cache.set("new_key", "new_value")
        
        # 第一个键应该被淘汰
        assert memory_cache.get("key_0") is None
        assert memory_cache.get("new_key") == "new_value"
    
    def test_cache_stats(self, memory_cache):
        """测试缓存统计"""
        # 添加一些数据
        for i in range(5):
            memory_cache.set(f"key_{i}", f"value_{i}")
        
        stats = memory_cache.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_items' in stats
        assert 'max_size' in stats
        assert 'usage_ratio' in stats
        assert stats['total_items'] == 5


class TestFactorCache:
    """因子缓存测试"""
    
    @pytest.fixture
    def mock_config(self):
        """创建模拟配置"""
        config = CacheConfig()
        config.redis_host = 'localhost'  # 假设Redis不可用
        return config
    
    @pytest.fixture
    def factor_cache(self, mock_config):
        """创建因子缓存实例"""
        return FactorCache(mock_config)
    
    def test_factor_data_caching(self, factor_cache):
        """测试因子数据缓存"""
        from src.compute.factor_models import FactorValue
        
        ts_code = "000001.SZ"
        factor_name = "test_factor"
        factor_values = [
            FactorValue(date=date(2024, 1, 1), value=1.0),
            FactorValue(date=date(2024, 1, 2), value=2.0)
        ]
        
        # 设置缓存
        success = factor_cache.set_factor_data(ts_code, factor_name, factor_values)
        assert success is True
        
        # 获取缓存
        cached_values = factor_cache.get_factor_data(ts_code, factor_name)
        assert cached_values == factor_values
    
    def test_cache_statistics(self, factor_cache):
        """测试缓存统计"""
        # 执行一些缓存操作
        factor_cache.get_factor_data("000001.SZ", "test_factor")  # miss
        
        stats = factor_cache.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'requests' in stats
        assert 'memory_cache' in stats
        assert stats['requests']['misses'] >= 1


class TestDependencyManager:
    """依赖关系管理器测试"""
    
    def test_dependency_registration(self):
        """测试依赖注册"""
        from src.compute.incremental_calculator import FactorDependency, DependencyType
        
        manager = DependencyManager()
        
        # 注册新依赖
        dependency = FactorDependency(
            'test_factor', 
            DependencyType.DATA_DEPENDENCY, 
            ['close'], 
            10
        )
        manager.register_dependency(dependency)
        
        # 检查依赖是否注册成功
        registered_dep = manager.get_dependency('test_factor')
        assert registered_dep is not None
        assert registered_dep.factor_name == 'test_factor'
        assert registered_dep.lookback_days == 10
    
    def test_calculation_order(self):
        """测试计算顺序"""
        manager = DependencyManager()
        
        # 测试已注册的因子
        factor_names = ['sma_5', 'sma_20', 'macd', 'ema_12', 'ema_26']
        ordered_factors = manager.get_calculation_order(factor_names)
        
        assert isinstance(ordered_factors, list)
        assert len(ordered_factors) == len(factor_names)
        assert set(ordered_factors) == set(factor_names)
        
        # MACD应该在EMA之后
        if 'macd' in ordered_factors and 'ema_12' in ordered_factors:
            macd_idx = ordered_factors.index('macd')
            ema12_idx = ordered_factors.index('ema_12')
            ema26_idx = ordered_factors.index('ema_26')
            assert macd_idx > ema12_idx
            assert macd_idx > ema26_idx
    
    def test_required_lookback_days(self):
        """测试所需回看天数"""
        manager = DependencyManager()
        
        factor_names = ['sma_5', 'sma_20', 'rsi_14']
        max_lookback = manager.get_required_lookback_days(factor_names)
        
        assert isinstance(max_lookback, int)
        assert max_lookback >= 20  # sma_20需要20天


class TestIncrementalDataManager:
    """增量数据管理器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    def test_get_trading_dates(self, mock_engine):
        """测试获取交易日期"""
        manager = IncrementalDataManager(mock_engine)
        
        # 模拟数据库查询结果
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(trade_date=date(2024, 1, 1)),
            Mock(trade_date=date(2024, 1, 2)),
            Mock(trade_date=date(2024, 1, 3))
        ]
        
        mock_conn = Mock()
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 3)
        
        trading_dates = manager.get_trading_dates(start_date, end_date)
        
        assert isinstance(trading_dates, list)
        assert len(trading_dates) == 3


class TestCompressionEngine:
    """压缩引擎测试"""
    
    @pytest.fixture
    def compression_engine(self):
        """创建压缩引擎实例"""
        from src.compute.data_compression_archiver import DataCompressionConfig
        config = DataCompressionConfig()
        config.compression_threshold_mb = 0  # 总是压缩
        return CompressionEngine(config)
    
    def test_data_compression(self, compression_engine):
        """测试数据压缩"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'col1': range(1000),
            'col2': np.random.randn(1000),
            'col3': ['test_string'] * 1000
        })
        
        # 压缩数据
        compressed_data = compression_engine.compress_data(test_data)
        
        assert isinstance(compressed_data, bytes)
        assert compressed_data.startswith(b'GZIP:')
        
        # 解压缩数据
        decompressed_data = compression_engine.decompress_data(compressed_data)
        
        # 验证数据完整性
        pd.testing.assert_frame_equal(test_data, decompressed_data)
    
    def test_compression_ratio(self, compression_engine):
        """测试压缩比"""
        # 创建可压缩的数据
        test_data = ['repeated_string'] * 10000
        
        compression_ratio = compression_engine.get_compression_ratio(test_data)
        
        assert isinstance(compression_ratio, float)
        assert 0 < compression_ratio < 1  # 应该有压缩效果


class TestArchiveManager:
    """归档管理器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def temp_config(self):
        """创建临时配置"""
        from src.compute.data_compression_archiver import DataCompressionConfig
        
        config = DataCompressionConfig()
        
        # 使用临时目录
        temp_dir = tempfile.mkdtemp()
        config.archive_path = os.path.join(temp_dir, 'archive')
        config.backup_path = os.path.join(temp_dir, 'backup')
        
        return config
    
    def test_archive_manager_initialization(self, mock_engine, temp_config):
        """测试归档管理器初始化"""
        manager = ArchiveManager(mock_engine, temp_config)
        
        # 检查目录是否创建
        assert Path(temp_config.archive_path).exists()
        assert hasattr(manager, 'compression_engine')
    
    def test_get_archivable_data(self, mock_engine, temp_config):
        """测试获取可归档数据"""
        manager = ArchiveManager(mock_engine, temp_config)
        
        # 模拟数据库查询结果
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(
                ts_code='000001.SZ',
                factor_name='test_factor',
                record_count=1000,
                min_date=date(2023, 1, 1),
                max_date=date(2023, 12, 31),
                data_quality=0.95
            )
        ]
        
        mock_conn = Mock()
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        cutoff_date = date(2024, 1, 1)
        archivable_data = manager.get_archivable_data(cutoff_date)
        
        assert isinstance(archivable_data, list)
        if archivable_data:
            assert 'ts_code' in archivable_data[0]
            assert 'factor_name' in archivable_data[0]
            assert 'record_count' in archivable_data[0]


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])