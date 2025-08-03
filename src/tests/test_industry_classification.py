#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申万行业分类管理器测试

测试行业分类数据同步、股票行业归属映射等功能
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

from src.data.industry_classification import IndustryClassificationManager


class TestIndustryClassificationManager:
    """申万行业分类管理器测试类"""
    
    @pytest.fixture
    def mock_engine(self):
        """模拟数据库引擎"""
        engine = Mock()
        conn = Mock()
        engine.connect.return_value.__enter__.return_value = conn
        return engine, conn
    
    @pytest.fixture
    def mock_tushare_pro(self):
        """模拟Tushare Pro API"""
        return Mock()
    
    @pytest.fixture
    def manager(self, mock_engine, mock_tushare_pro):
        """创建管理器实例"""
        engine, conn = mock_engine
        
        with patch('src.data.industry_classification.get_db_engine', return_value=engine), \
             patch('src.data.industry_classification.ts.pro_api', return_value=mock_tushare_pro), \
             patch.dict(os.environ, {'TUSHARE_TOKEN': 'test_token'}):
            
            manager = IndustryClassificationManager()
            manager.engine = engine
            manager.pro = mock_tushare_pro
            return manager
    
    def test_init_manager(self, manager):
        """测试管理器初始化"""
        assert manager is not None
        assert manager.api_limit > 0
        assert manager.retry_times > 0
        assert manager.retry_delay > 0
    
    def test_rate_limit_check(self, manager):
        """测试API调用频率限制"""
        # 重置计数器
        manager.call_count = 0
        manager.window_start = datetime.now()
        
        # 测试正常调用
        manager._rate_limit_check()
        assert manager.call_count == 1
        
        # 测试达到限制
        manager.call_count = manager.api_limit
        start_time = datetime.now()
        manager._rate_limit_check()
        end_time = datetime.now()
        
        # 应该重置计数器
        assert manager.call_count == 1
    
    def test_fetch_industry_classification(self, manager, mock_tushare_pro):
        """测试获取行业分类数据"""
        # 模拟Tushare返回数据
        mock_df = pd.DataFrame({
            'index_code': ['801010.SI', '801020.SI', '801030.SI'],
            'industry_name': ['农林牧渔', '采掘', '化工'],
            'parent_code': ['', '', '']
        })
        mock_tushare_pro.index_classify.return_value = mock_df
        
        result = manager._fetch_industry_classification('L1')
        
        assert result is not None
        assert len(result) == 3
        assert result.iloc[0]['industry_name'] == '农林牧渔'
        
        # 验证API调用参数
        mock_tushare_pro.index_classify.assert_called_with(
            level='L1',
            src='SW2021'
        )
    
    def test_standardize_industry_data(self, manager):
        """测试行业分类数据标准化"""
        # 准备测试数据
        raw_data = pd.DataFrame({
            'index_code': ['801010.SI', '801020.SI'],
            'industry_name': ['农林牧渔', '采掘'],
            'parent_code': ['', '']
        })
        
        result = manager._standardize_industry_data(raw_data, 'L1')
        
        assert len(result) == 2
        assert result[0]['industry_code'] == '801010.SI'
        assert result[0]['industry_name'] == '农林牧渔'
        assert result[0]['industry_level'] == 'L1'
        assert result[0]['source'] == 'tushare_sw2021'
    
    def test_fetch_industry_members(self, manager, mock_tushare_pro):
        """测试获取行业成分股数据"""
        # 模拟Tushare返回数据
        mock_df = pd.DataFrame({
            'con_code': ['000001.SZ', '000002.SZ', '600000.SH'],
            'in_date': ['20200101', '20200101', '20200101'],
            'out_date': [None, '20231201', None]
        })
        mock_tushare_pro.index_member.return_value = mock_df
        
        result = manager._fetch_industry_members('801010.SI')
        
        assert result is not None
        assert len(result) == 3
        assert result.iloc[0]['con_code'] == '000001.SZ'
        
        # 验证API调用参数
        mock_tushare_pro.index_member.assert_called_with(
            index_code='801010.SI'
        )
    
    def test_standardize_industry_members(self, manager):
        """测试行业成分股数据标准化"""
        # 准备测试数据
        raw_data = pd.DataFrame({
            'con_code': ['000001.SZ', '000002.SZ'],
            'in_date': ['20200101', '20200101'],
            'out_date': [None, '20231201']
        })
        
        result = manager._standardize_industry_members(raw_data, '801010.SI')
        
        assert len(result) == 2
        assert result[0]['ts_code'] == '000001.SZ'
        assert result[0]['industry_code'] == '801010.SI'
        assert result[0]['is_current'] is True  # 没有退出日期
        assert result[1]['is_current'] is False  # 有退出日期
    
    def test_get_industry_codes(self, manager, mock_engine):
        """测试获取行业代码列表"""
        engine, conn = mock_engine
        
        # 模拟数据库查询结果
        mock_result = [
            ('801010.SI',),
            ('801020.SI',),
            ('801030.SI',)
        ]
        conn.execute.return_value = mock_result
        
        codes = manager._get_industry_codes()
        
        assert len(codes) == 3
        assert codes[0] == '801010.SI'
        assert codes[1] == '801020.SI'
        assert codes[2] == '801030.SI'
    
    def test_get_stock_industry_history(self, manager, mock_engine):
        """测试查询股票行业归属历史"""
        engine, conn = mock_engine
        
        # 模拟数据库查询结果
        mock_result = [
            ('801010.SI', '农林牧渔', 'L1', datetime(2020, 1, 1).date(), None, True),
            ('801011.SI', '农业', 'L2', datetime(2020, 1, 1).date(), None, True)
        ]
        conn.execute.return_value = mock_result
        
        result = manager.get_stock_industry_history('000001.SZ', '2024-01-01')
        
        assert result is not None
        assert result['ts_code'] == '000001.SZ'
        assert len(result['industries']) == 2
        assert result['industries'][0]['industry_name'] == '农林牧渔'
        assert result['industries'][0]['industry_level'] == 'L1'
    
    def test_validate_industry_data_integrity(self, manager, mock_engine):
        """测试行业数据完整性验证"""
        engine, conn = mock_engine
        
        # 模拟数据库查询结果
        mock_results = [
            # 行业分类统计
            [('L1', 28, 0), ('L2', 104, 28), ('L3', 227, 104)],
            # 映射统计
            [(5000, 15000, 5000)],
            # 孤儿记录
            [0]
        ]
        
        conn.execute.side_effect = [
            mock_results[0],  # 第一次查询
            [mock_results[1][0]],  # 第二次查询
            Mock(scalar=Mock(return_value=mock_results[2][0]))  # 第三次查询
        ]
        
        result = manager.validate_industry_data_integrity()
        
        assert 'classification_stats' in result
        assert 'mapping_stats' in result
        assert 'orphan_mappings' in result
        assert 'quality_score' in result
        
        assert result['classification_stats']['L1']['total_count'] == 28
        assert result['mapping_stats']['mapped_stocks'] == 5000
        assert result['orphan_mappings'] == 0
        assert result['quality_score'] == 1.0  # 没有问题，满分
    
    def test_sync_industry_classification_success(self, manager, mock_tushare_pro, mock_engine):
        """测试行业分类同步成功"""
        engine, conn = mock_engine
        
        # 模拟Tushare返回数据
        mock_df = pd.DataFrame({
            'index_code': ['801010.SI'],
            'industry_name': ['农林牧渔'],
            'parent_code': ['']
        })
        mock_tushare_pro.index_classify.return_value = mock_df
        
        # 模拟数据库操作成功
        conn.execute.return_value = Mock()
        
        result = manager.sync_industry_classification('L1')
        
        assert result is True
        # 验证API调用
        mock_tushare_pro.index_classify.assert_called()
        # 验证数据库操作
        conn.execute.assert_called()
        conn.commit.assert_called()
    
    def test_sync_stock_industry_mapping_success(self, manager, mock_tushare_pro, mock_engine):
        """测试股票行业映射同步成功"""
        engine, conn = mock_engine
        
        # 模拟获取行业代码
        conn.execute.side_effect = [
            [('801010.SI',), ('801020.SI',)],  # 行业代码查询
            Mock(), Mock()  # 后续的插入操作
        ]
        
        # 模拟Tushare返回成分股数据
        mock_df = pd.DataFrame({
            'con_code': ['000001.SZ', '000002.SZ'],
            'in_date': ['20200101', '20200101'],
            'out_date': [None, None]
        })
        mock_tushare_pro.index_member.return_value = mock_df
        
        result = manager.sync_stock_industry_mapping()
        
        assert result is True
        # 验证API调用
        assert mock_tushare_pro.index_member.call_count == 2  # 两个行业
        # 验证数据库操作
        conn.commit.assert_called()
    
    def test_full_sync_success(self, manager):
        """测试完整同步成功"""
        with patch.object(manager, 'sync_industry_classification', return_value=True), \
             patch.object(manager, 'sync_stock_industry_mapping', return_value=True):
            
            result = manager.full_sync()
            
            assert result['industry_classification'] is True
            assert result['stock_industry_mapping'] is True
    
    def test_update_sync_status(self, manager, mock_engine):
        """测试更新同步状态"""
        engine, conn = mock_engine
        
        manager._update_sync_status(
            'industry_classification', 'success',
            records_processed=100,
            records_failed=0,
            duration=60
        )
        
        # 验证SQL执行
        conn.execute.assert_called()
        conn.commit.assert_called()


# 性能测试
class TestIndustryClassificationManagerPerformance:
    """申万行业分类管理器性能测试"""
    
    @pytest.mark.performance
    def test_data_standardization_performance(self):
        """测试数据标准化性能"""
        manager = IndustryClassificationManager()
        
        # 准备大量测试数据
        raw_data = pd.DataFrame({
            'index_code': [f'80{i:04d}.SI' for i in range(1000)],
            'industry_name': [f'行业{i}' for i in range(1000)],
            'parent_code': [f'80{i//10:04d}.SI' if i > 0 else '' for i in range(1000)]
        })
        
        import time
        start_time = time.time()
        
        result = manager._standardize_industry_data(raw_data, 'L3')
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 1000条数据应在1秒内完成标准化
        assert duration < 1.0, f"数据标准化性能过慢: {duration:.2f}秒"
        assert len(result) == 1000
    
    @pytest.mark.performance
    def test_industry_members_standardization_performance(self):
        """测试行业成分股标准化性能"""
        manager = IndustryClassificationManager()
        
        # 准备大量测试数据
        raw_data = pd.DataFrame({
            'con_code': [f'{i:06d}.SZ' for i in range(5000)],
            'in_date': ['20200101'] * 5000,
            'out_date': [None] * 5000
        })
        
        import time
        start_time = time.time()
        
        result = manager._standardize_industry_members(raw_data, '801010.SI')
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 5000条数据应在2秒内完成标准化
        assert duration < 2.0, f"成分股标准化性能过慢: {duration:.2f}秒"
        assert len(result) == 5000


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])