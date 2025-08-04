"""
TimescaleDB优化器测试

测试超表创建、索引建立、数据压缩等功能
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import text

from src.data.timescale_optimizer import TimescaleOptimizer


class TimescaleTestDataBuilder:
    """测试数据构建器"""
    
    @staticmethod
    def create_hypertable_data(count=2):
        """创建超表测试数据"""
        return [
            ('stock_daily', 2, 10, True, 'default'),
            ('financial_data', 1, 5, False, 'default')
        ][:count]
    
    @staticmethod
    def create_chunk_data(table_name='stock_daily', count=1):
        """创建分块测试数据"""
        base_data = [
            ('_timescaledb_internal._hyper_1_1_chunk', '2024-01-01', '2024-02-01', True, 'default', None),
            ('_timescaledb_internal._hyper_1_2_chunk', '2024-02-01', '2024-03-01', False, 'default', None),
        ]
        return base_data[:count]
    
    @staticmethod
    def create_compression_stats(table_name='stock_daily'):
        """创建压缩统计测试数据"""
        return [
            (table_name, 10, 5, 1000000, 500000, 'localhost')
        ]


class MockConnectionBuilder:
    """模拟连接构建器"""
    
    def __init__(self):
        self.connection = Mock()
        self.reset_defaults()
    
    def reset_defaults(self):
        """重置默认行为"""
        self.connection.execute.return_value.scalar.return_value = True
        self.connection.execute.return_value.fetchall.return_value = []
        self.connection.execute.return_value.keys.return_value = []
        self.connection.commit.return_value = None
        return self
    
    def with_scalar_result(self, result):
        """设置标量查询结果"""
        self.connection.execute.return_value.scalar.return_value = result
        return self
    
    def with_fetchall_result(self, result, keys=None):
        """设置fetchall查询结果"""
        self.connection.execute.return_value.fetchall.return_value = result
        if keys:
            self.connection.execute.return_value.keys.return_value = keys
        return self
    
    def with_exception(self, exception):
        """设置异常"""
        self.connection.execute.side_effect = exception
        return self
    
    def build(self):
        """构建模拟连接"""
        return self.connection


class TestTimescaleOptimizer:
    """TimescaleDB优化器测试类"""
    
    # Test constants
    SAMPLE_TABLE_NAME = 'stock_daily'
    SAMPLE_HYPERTABLE_DATA = [
        ('stock_daily', 2, 10, True, 'default'),
        ('financial_data', 1, 5, False, 'default')
    ]
    SAMPLE_CHUNK_DATA = [
        ('_timescaledb_internal._hyper_1_1_chunk', '2024-01-01', '2024-02-01', True, 'default', None)
    ]
    
    @pytest.fixture
    def mock_db_engine(self):
        """创建模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def optimizer(self, mock_db_engine):
        """创建优化器实例"""
        with patch('src.data.timescale_optimizer.get_db_engine', return_value=mock_db_engine):
            return TimescaleOptimizer()
    
    @pytest.fixture
    def mock_connection(self):
        """创建配置好的模拟数据库连接"""
        conn = Mock()
        # 设置默认返回值
        conn.execute.return_value.scalar.return_value = True
        conn.execute.return_value.fetchall.return_value = []
        conn.execute.return_value.keys.return_value = []
        conn.commit.return_value = None
        return conn
    
    def setup_connection_context(self, optimizer, mock_connection):
        """设置数据库连接上下文管理器"""
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        optimizer.engine.connect.return_value = context_manager
        return context_manager
    
    def test_setup_timescale_extension_success(self, optimizer, mock_connection):
        """测试TimescaleDB扩展设置成功"""
        # Arrange
        mock_connection.execute.return_value.scalar.return_value = False
        self.setup_connection_context(optimizer, mock_connection)
        
        # Act
        result = optimizer.setup_timescale_extension()
        
        # Assert
        assert result is True, "TimescaleDB扩展设置应该成功"
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_setup_timescale_extension_already_exists(self, optimizer, mock_connection):
        """测试TimescaleDB扩展已存在"""
        # Arrange
        mock_connection.execute.return_value.scalar.return_value = True
        self.setup_connection_context(optimizer, mock_connection)
        
        # Act
        result = optimizer.setup_timescale_extension()
        
        # Assert
        assert result is True
    
    def test_create_hypertable_success(self, optimizer, mock_connection):
        """测试创建超表成功"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.return_value.scalar.side_effect = [False, True]  # 不是超表，表存在
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.create_hypertable(table_name)
        
        # Assert
        assert result is True
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_create_hypertable_already_exists(self, optimizer, mock_connection):
        """测试超表已存在"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.return_value.scalar.return_value = True
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.create_hypertable(table_name)
        
        # Assert
        assert result is True
    
    def test_create_hypertable_table_not_exists(self, optimizer, mock_connection):
        """测试表不存在时创建超表失败"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.return_value.scalar.side_effect = [False, False]  # 不是超表，表不存在
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.create_hypertable(table_name)
        
        # Assert
        assert result is False
    
    def test_create_hypertable_invalid_table(self, optimizer):
        """测试无效表名"""
        # Act
        result = optimizer.create_hypertable('invalid_table')
        
        # Assert
        assert result is False
    
    def test_create_hypertable_exception_handling(self, optimizer, mock_connection):
        """测试创建超表时的异常处理"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.side_effect = Exception("Database connection error")
        self.setup_connection_context(optimizer, mock_connection)
        
        # Act
        result = optimizer.create_hypertable(table_name)
        
        # Assert
        assert result is False, "异常情况下应该返回False"
    
    def test_create_hypertable_with_force_recreate(self, optimizer, mock_connection):
        """测试强制重新创建超表"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.return_value.scalar.side_effect = [True, True]  # 已是超表，表存在
        self.setup_connection_context(optimizer, mock_connection)
        
        # Act
        result = optimizer.create_hypertable(table_name, force_recreate=True)
        
        # Assert
        assert result is True, "强制重新创建应该成功"
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_create_composite_indexes_success(self, optimizer, mock_connection):
        """测试创建复合索引成功"""
        # Arrange
        table_name = 'stock_daily'
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.create_composite_indexes(table_name)
        
        # Assert
        assert result is True
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_enable_compression_success(self, optimizer, mock_connection):
        """测试启用数据压缩成功"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.return_value.scalar.return_value = False  # 压缩未启用
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.enable_compression(table_name)
        
        # Assert
        assert result is True
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_enable_compression_already_enabled(self, optimizer, mock_connection):
        """测试压缩已启用"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.return_value.scalar.return_value = True  # 压缩已启用
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.enable_compression(table_name)
        
        # Assert
        assert result is True
    
    def test_add_retention_policy_success(self, optimizer, mock_connection):
        """测试添加保留策略成功"""
        # Arrange
        table_name = 'stock_daily'
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.add_retention_policy(table_name)
        
        # Assert
        assert result is True
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_optimize_table_success(self, optimizer):
        """测试优化单个表成功"""
        # Arrange
        table_name = 'stock_daily'
        
        with patch.object(optimizer, 'create_hypertable', return_value=True), \
             patch.object(optimizer, 'create_composite_indexes', return_value=True), \
             patch.object(optimizer, 'enable_compression', return_value=True), \
             patch.object(optimizer, 'add_retention_policy', return_value=True):
            
            # Act
            result = optimizer.optimize_table(table_name)
            
            # Assert
            assert result is True
    
    def test_optimize_table_hypertable_creation_fails(self, optimizer):
        """测试超表创建失败时优化失败"""
        # Arrange
        table_name = 'stock_daily'
        
        with patch.object(optimizer, 'create_hypertable', return_value=False):
            # Act
            result = optimizer.optimize_table(table_name)
            
            # Assert
            assert result is False
    
    def test_optimize_all_tables_success(self, optimizer):
        """测试优化所有表成功"""
        # Arrange
        with patch.object(optimizer, 'setup_timescale_extension', return_value=True), \
             patch.object(optimizer, 'optimize_table', return_value=True):
            
            # Act
            results = optimizer.optimize_all_tables()
            
            # Assert
            assert len(results) == len(optimizer.hypertable_configs)
            assert all(results.values())
    
    def test_optimize_all_tables_extension_setup_fails(self, optimizer):
        """测试扩展设置失败时停止优化"""
        # Arrange
        with patch.object(optimizer, 'setup_timescale_extension', return_value=False):
            # Act
            results = optimizer.optimize_all_tables()
            
            # Assert
            assert len(results) == 0
    
    def test_get_hypertable_info_success(self, optimizer, mock_connection):
        """测试获取超表信息成功"""
        # Arrange
        mock_data = TimescaleTestDataBuilder.create_hypertable_data()
        mock_connection.execute.return_value.fetchall.return_value = mock_data
        mock_connection.execute.return_value.keys.return_value = [
            'hypertable_name', 'num_dimensions', 'num_chunks', 
            'compression_enabled', 'tablespace'
        ]
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.get_hypertable_info()
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_connection.execute.assert_called()
    
    def test_get_chunk_info_success(self, optimizer, mock_connection):
        """测试获取分块信息成功"""
        # Arrange
        table_name = 'stock_daily'
        mock_data = [
            ('_timescaledb_internal._hyper_1_1_chunk', '2024-01-01', '2024-02-01', True, 'default', None)
        ]
        mock_connection.execute.return_value.fetchall.return_value = mock_data
        mock_connection.execute.return_value.keys.return_value = [
            'chunk_name', 'range_start', 'range_end', 
            'is_compressed', 'chunk_tablespace', 'data_nodes'
        ]
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.get_chunk_info(table_name)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        mock_connection.execute.assert_called()
    
    def test_get_compression_stats_success(self, optimizer, mock_connection):
        """测试获取压缩统计信息成功"""
        # Arrange
        mock_data = [
            ('stock_daily', 10, 5, 1000000, 500000, 'localhost')
        ]
        mock_connection.execute.return_value.fetchall.return_value = mock_data
        mock_connection.execute.return_value.keys.return_value = [
            'hypertable_name', 'total_chunks', 'number_compressed_chunks',
            'before_compression_total_bytes', 'after_compression_total_bytes', 'node_name'
        ]
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.get_compression_stats()
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        mock_connection.execute.assert_called()
    
    def test_manual_compress_chunks_success(self, optimizer, mock_connection):
        """测试手动压缩分块成功"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.return_value.fetchall.return_value = [
            ('chunk1',), ('chunk2',), ('chunk3',)
        ]
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.manual_compress_chunks(table_name, '1 month')
        
        # Assert
        assert result is True
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    def test_reorder_chunks_success(self, optimizer, mock_connection):
        """测试重新排序分块成功"""
        # Arrange
        table_name = 'stock_daily'
        mock_connection.execute.return_value.fetchall.return_value = [
            ('chunk1',), ('chunk2',)
        ]
        optimizer.engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Act
        result = optimizer.reorder_chunks(table_name)
        
        # Assert
        assert result is True
        mock_connection.execute.assert_called()
        mock_connection.commit.assert_called()
    
    @pytest.mark.parametrize("table_name,expected_config", [
        ('stock_daily', {
            'time_column': 'trade_date',
            'chunk_time_interval': '1 month',
            'compression_after': '3 months',
            'retention_period': '5 years',
            'partition_column': 'ts_code'
        }),
        ('financial_data', {
            'time_column': 'end_date',
            'chunk_time_interval': '3 months',
            'compression_after': '6 months',
            'retention_period': '10 years',
            'partition_column': 'ts_code'
        }),
        ('news_sentiment', {
            'time_column': 'news_date',
            'chunk_time_interval': '1 month',
            'compression_after': '2 months',
            'retention_period': '2 years',
            'partition_column': 'ts_code'
        })
    ])
    def test_hypertable_configs(self, optimizer, table_name, expected_config):
        """测试超表配置正确性"""
        # Act & Assert
        assert table_name in optimizer.hypertable_configs
        config = optimizer.hypertable_configs[table_name]
        
        for key, expected_value in expected_config.items():
            assert config[key] == expected_value
    
    def test_table_exists_true(self, optimizer, mock_connection):
        """测试表存在检查返回True"""
        # Arrange
        mock_connection.execute.return_value.scalar.return_value = True
        
        # Act
        result = optimizer._table_exists(mock_connection, 'stock_daily')
        
        # Assert
        assert result is True
    
    def test_table_exists_false(self, optimizer, mock_connection):
        """测试表存在检查返回False"""
        # Arrange
        mock_connection.execute.return_value.scalar.return_value = False
        
        # Act
        result = optimizer._table_exists(mock_connection, 'nonexistent_table')
        
        # Assert
        assert result is False
    
    def test_table_exists_exception(self, optimizer, mock_connection):
        """测试表存在检查异常处理"""
        # Arrange
        mock_connection.execute.side_effect = Exception("Database error")
        
        # Act
        result = optimizer._table_exists(mock_connection, 'stock_daily')
        
        # Assert
        assert result is False