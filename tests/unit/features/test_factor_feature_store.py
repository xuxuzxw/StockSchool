#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子特征商店测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.features.factor_feature_store import FactorFeatureStore, FactorMetadata, FactorVersion
from src.compute.feature_store_adapter import FeatureStoreAdapter

class TestFactorMetadata:
    """因子元数据测试"""
    
    def test_factor_metadata_creation(self):
        """测试因子元数据创建"""
        metadata = FactorMetadata(
            factor_name="test_factor",
            factor_type="technical",
            category="momentum",
            description="测试因子",
            formula="TEST(close, 14)",
            parameters={"window": 14},
            data_requirements=["stock_daily"],
            update_frequency="daily",
            data_schema={"ts_code": "string", "factor_date": "date", "test_factor": "float"},
            tags=["test", "momentum"]
        )
        
        assert metadata.factor_name == "test_factor"
        assert metadata.factor_type == "technical"
        assert metadata.parameters["window"] == 14
        assert "test" in metadata.tags
    
    def test_factor_metadata_to_dict(self):
        """测试因子元数据转换为字典"""
        metadata = FactorMetadata(
            factor_name="test_factor",
            factor_type="technical",
            category="momentum",
            description="测试因子",
            formula="TEST(close, 14)",
            parameters={"window": 14},
            data_requirements=["stock_daily"],
            update_frequency="daily",
            data_schema={"ts_code": "string", "factor_date": "date", "test_factor": "float"}
        )
        
        metadata_dict = metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["factor_name"] == "test_factor"
        assert metadata_dict["parameters"]["window"] == 14

class TestFactorVersion:
    """因子版本测试"""
    
    def test_factor_version_creation(self):
        """测试因子版本创建"""
        version = FactorVersion(
            version_id="test_factor_v1_abc123",
            factor_name="test_factor",
            version="1",
            created_at=datetime.now(),
            metadata={"algorithm": "test_algorithm"}
        )
        
        assert version.version_id == "test_factor_v1_abc123"
        assert version.factor_name == "test_factor"
        assert version.version == "1"
        assert version.metadata["algorithm"] == "test_algorithm"
    
    def test_factor_version_to_dict(self):
        """测试因子版本转换为字典"""
        created_at = datetime.now()
        version = FactorVersion(
            version_id="test_factor_v1_abc123",
            factor_name="test_factor",
            version="1",
            created_at=created_at,
            metadata={"algorithm": "test_algorithm"}
        )
        
        version_dict = version.to_dict()
        
        assert isinstance(version_dict, dict)
        assert version_dict["version_id"] == "test_factor_v1_abc123"
        assert version_dict["created_at"] == created_at.isoformat()

class TestFactorFeatureStore:
    """因子特征商店测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def mock_feature_store(self, mock_engine):
        """模拟特征商店"""
        with patch('src.features.factor_feature_store.FeatureStore') as mock_fs:
            mock_fs_instance = Mock()
            mock_fs.return_value = mock_fs_instance
            
            store = FactorFeatureStore(mock_engine)
            store.feature_store = mock_fs_instance
            return store
    
    def test_register_factor(self, mock_feature_store, mock_engine):
        """测试注册因子"""
        # 模拟数据库连接
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        metadata = FactorMetadata(
            factor_name="test_factor",
            factor_type="technical",
            category="momentum",
            description="测试因子",
            formula="TEST(close, 14)",
            parameters={"window": 14},
            data_requirements=["stock_daily"],
            update_frequency="daily",
            data_schema={"ts_code": "string", "factor_date": "date", "test_factor": "float"}
        )
        
        result = mock_feature_store.register_factor(metadata)
        
        assert result == "test_factor"
        mock_conn.execute.assert_called()
    
    def test_create_factor_version(self, mock_feature_store, mock_engine):
        """测试创建因子版本"""
        # 模拟数据库连接
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        # 模拟查询结果（没有现有版本）
        mock_result = Mock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result
        
        version_id = mock_feature_store.create_factor_version(
            factor_name="test_factor",
            algorithm_code="def calculate(): pass",
            parameters={"window": 14},
            metadata={"description": "测试版本"}
        )
        
        assert version_id.startswith("test_factor_v")
        assert mock_conn.execute.call_count >= 2  # 查询 + 插入
    
    def test_store_factor_data(self, mock_feature_store):
        """测试存储因子数据"""
        # 准备测试数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'factor_date': [date(2024, 1, 1), date(2024, 1, 1)],
            'test_factor': [10.5, 15.2]
        })
        
        # 模拟特征商店存储成功
        mock_feature_store.feature_store.store_features.return_value = True
        
        # 模拟数据库操作
        with patch.object(mock_feature_store, '_record_data_lineage'), \
             patch.object(mock_feature_store, '_calculate_quality_metrics'):
            
            success = mock_feature_store.store_factor_data(
                factor_name="test_factor",
                version_id="test_factor_v1_abc123",
                data=test_data
            )
        
        assert success is True
        mock_feature_store.feature_store.store_features.assert_called_once()
    
    def test_get_factor_data(self, mock_feature_store):
        """测试获取因子数据"""
        # 准备模拟数据
        mock_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'factor_date': [date(2024, 1, 1), date(2024, 1, 1)],
            'test_factor': [10.5, 15.2],
            'version_id': ['test_factor_v1_abc123', 'test_factor_v1_abc123'],
            'created_at': [datetime.now(), datetime.now()]
        })
        
        # 模拟获取最新版本
        with patch.object(mock_feature_store, '_get_latest_version', return_value='test_factor_v1_abc123'):
            mock_feature_store.feature_store.get_features.return_value = mock_data
            
            result = mock_feature_store.get_factor_data(
                factor_name="test_factor",
                ts_codes=["000001.SZ", "000002.SZ"]
            )
        
        assert not result.empty
        assert len(result) == 2
        assert 'version_id' not in result.columns  # 应该被移除
        assert 'created_at' not in result.columns  # 应该被移除
    
    def test_get_factor_metadata(self, mock_feature_store, mock_engine):
        """测试获取因子元数据"""
        # 模拟数据库查询结果
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        mock_row = Mock()
        mock_row.factor_name = "test_factor"
        mock_row.factor_type = "technical"
        mock_row.category = "momentum"
        mock_row.description = "测试因子"
        mock_row.formula = "TEST(close, 14)"
        mock_row.parameters = '{"window": 14}'
        mock_row.data_requirements = ["stock_daily"]
        mock_row.update_frequency = "daily"
        mock_row.data_schema = '{"ts_code": "string"}'
        mock_row.tags = ["test"]
        
        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_conn.execute.return_value = mock_result
        
        metadata = mock_feature_store.get_factor_metadata("test_factor")
        
        assert metadata is not None
        assert metadata.factor_name == "test_factor"
        assert metadata.parameters["window"] == 14
    
    def test_search_factors(self, mock_feature_store, mock_engine):
        """测试搜索因子"""
        # 模拟数据库查询结果
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        mock_row = Mock()
        mock_row.factor_name = "test_factor"
        mock_row.factor_type = "technical"
        mock_row.category = "momentum"
        mock_row.description = "测试因子"
        mock_row.formula = "TEST(close, 14)"
        mock_row.parameters = '{"window": 14}'
        mock_row.data_requirements = ["stock_daily"]
        mock_row.update_frequency = "daily"
        mock_row.data_schema = '{"ts_code": "string"}'
        mock_row.tags = ["test"]
        
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_conn.execute.return_value = mock_result
        
        results = mock_feature_store.search_factors(
            query="test",
            factor_type="technical"
        )
        
        assert len(results) == 1
        assert results[0].factor_name == "test_factor"

class TestFeatureStoreAdapter:
    """特征商店适配器测试"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        return Mock()
    
    @pytest.fixture
    def mock_adapter(self, mock_engine):
        """模拟适配器"""
        with patch('src.compute.feature_store_adapter.FactorFeatureStore'), \
             patch('src.compute.feature_store_adapter.FactorEngine'), \
             patch('src.compute.feature_store_adapter.TechnicalFactorEngine'), \
             patch('src.compute.feature_store_adapter.FundamentalFactorEngine'), \
             patch('src.compute.feature_store_adapter.SentimentFactorEngine'):
            
            adapter = FeatureStoreAdapter(mock_engine)
            
            # 模拟特征商店
            adapter.feature_store = Mock()
            
            # 模拟计算引擎
            adapter.technical_engine = Mock()
            adapter.fundamental_engine = Mock()
            adapter.sentiment_engine = Mock()
            
            return adapter
    
    def test_calculate_and_store_factor(self, mock_adapter):
        """测试计算并存储因子"""
        # 模拟因子元数据
        mock_metadata = Mock()
        mock_metadata.factor_type = "technical"
        mock_metadata.parameters = {"window": 14}
        mock_adapter.feature_store.get_factor_metadata.return_value = mock_metadata
        
        # 模拟版本创建
        mock_adapter.feature_store.create_factor_version.return_value = "test_factor_v1_abc123"
        
        # 模拟计算结果
        mock_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'factor_date': [date(2024, 1, 1)],
            'test_factor': [10.5]
        })
        
        # 模拟计算方法
        mock_calc_method = Mock(return_value=mock_data)
        mock_adapter.technical_engine.calculate_test_factor = mock_calc_method
        
        # 模拟存储成功
        mock_adapter.feature_store.store_factor_data.return_value = True
        
        with patch.object(mock_adapter, '_get_algorithm_code', return_value="test_code"):
            success, version_id = mock_adapter.calculate_and_store_factor(
                factor_name="test_factor",
                ts_codes=["000001.SZ"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1)
            )
        
        assert success is True
        assert version_id == "test_factor_v1_abc123"
    
    def test_batch_calculate_and_store(self, mock_adapter):
        """测试批量计算并存储"""
        # 模拟单个因子计算成功
        with patch.object(mock_adapter, 'calculate_and_store_factor', return_value=(True, "v1")):
            results = mock_adapter.batch_calculate_and_store(
                factor_names=["factor1", "factor2"],
                ts_codes=["000001.SZ"]
            )
        
        assert len(results) == 2
        assert results["factor1"] == (True, "v1")
        assert results["factor2"] == (True, "v1")
    
    def test_compare_factor_versions(self, mock_adapter):
        """测试比较因子版本"""
        # 准备测试数据
        data1 = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'factor_date': [date(2024, 1, 1), date(2024, 1, 1)],
            'test_factor': [10.0, 15.0]
        })
        
        data2 = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'factor_date': [date(2024, 1, 1), date(2024, 1, 1)],
            'test_factor': [10.5, 15.5]
        })
        
        # 模拟数据获取
        mock_adapter.feature_store.get_factor_data.side_effect = [data1, data2]
        
        comparison = mock_adapter.compare_factor_versions(
            factor_name="test_factor",
            version_id1="v1",
            version_id2="v2"
        )
        
        assert comparison['comparison_possible'] is True
        assert comparison['data_points'] == 2
        assert 'correlation' in comparison
        assert 'mean_difference' in comparison
    
    def test_migrate_legacy_data(self, mock_adapter, mock_engine):
        """测试迁移历史数据"""
        # 模拟因子元数据
        mock_metadata = Mock()
        mock_metadata.factor_type = "technical"
        mock_adapter.feature_store.get_factor_metadata.return_value = mock_metadata
        
        # 模拟历史数据
        legacy_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'factor_date': [date(2024, 1, 1)],
            'test_factor': [10.5]
        })
        
        # 模拟数据库查询
        with patch('pandas.read_sql', return_value=legacy_data):
            # 模拟版本创建和数据存储
            mock_adapter.feature_store.create_factor_version.return_value = "legacy_v1"
            mock_adapter.feature_store.store_factor_data.return_value = True
            
            success = mock_adapter.migrate_legacy_data(
                factor_name="test_factor",
                source_table="stock_factors_technical"
            )
        
        assert success is True
        mock_adapter.feature_store.create_factor_version.assert_called_once()
        mock_adapter.feature_store.store_factor_data.assert_called_once()

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])