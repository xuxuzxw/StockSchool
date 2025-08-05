import os
import sys
from datetime import datetime, timedelta
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.industry_classification import IndustryClassificationManager

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申万行业分类管理器测试

测试行业分类数据同步、股票行业归属映射等功能
"""


# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestIndustryClassificationManager:
    """申万行业分类管理器测试类"""

    @pytest.fixture
    def mock_db_session(self):
        """模拟数据库会话"""
        return MagicMock()

    @pytest.fixture
    def mock_data_source(self):
        """模拟数据源"""
        source = MagicMock()
        source.get_industry_classification.return_value = pd.DataFrame(
            {"index_code": ["801010.SI", "801020.SI"], "industry_name": ["农林牧渔", "采掘"], "parent_code": ["", ""]}
        )
        source.get_industry_members.return_value = pd.DataFrame(
            {
                "con_code": ["000001.SZ", "000002.SZ"],
                "in_date": ["20200101", "20200101"],
                "out_date": [None, "20231201"],
            }
        )
        return source

    @pytest.fixture
    def manager(self, mock_db_session, mock_data_source):
        """创建管理器实例"""
        with patch(
            "src.data.industry_classification.DataSourceFactory.create_data_source", return_value=mock_data_source
        ):
            config = {"type": "tushare", "token": "test"}
            manager = IndustryClassificationManager(db_session=mock_db_session, data_source_config=config)
            # 手动将模拟的 repository 赋给 manager
            manager.repository = MagicMock()
            return manager

    def test_sync_industry_classification(self, manager, mock_data_source):
        """测试行业分类数据同步"""
        manager.sync_industry_classification()

        # 验证数据源是否被调用
        assert mock_data_source.get_industry_classification.call_count == 3  # L1, L2, L3

        # 验证 repository 是否被调用
        assert manager.repository.save_industry_classification.call_count == 3
        manager.repository.update_sync_status.assert_called_once_with("industry_classification")

    def test_standardize_industry_data(self, manager):
        """测试行业分类数据标准化"""
        raw_data = pd.DataFrame({"index_code": ["801010.SI"], "industry_name": ["农林牧渔"], "parent_code": [""]})
        standardized = manager._standardize_industry_data(raw_data, "L1")
        assert len(standardized) == 1
        assert standardized[0]["industry_level"] == "L1"

    def test_standardize_industry_members(self, manager):
        """测试行业成分股数据标准化"""
        raw_data = pd.DataFrame({"con_code": ["000001.SZ"], "in_date": ["20200101"], "out_date": [None]})
        standardized = manager._standardize_industry_members(raw_data, "801010.SI")
        assert len(standardized) == 1
        assert standardized[0]["is_current"] is True

    def test_sync_all_industry_members(self, manager, mock_data_source):
        """测试同步所有行业成员"""
        # 模拟 repository 返回行业代码
        manager.repository.get_all_industry_codes.return_value = ["801010.SI", "801020.SI"]

        manager.sync_all_industry_members()

        # 验证数据源和 repository 的调用
        assert mock_data_source.get_industry_members.call_count == 2
        assert manager.repository.save_industry_members.call_count == 2
        manager.repository.update_sync_status.assert_called_once_with("industry_members")

    def test_get_stock_industry_history(self, manager):
        """测试查询股票行业归属历史"""
        # 模拟 repository 返回数据
        manager.repository.get_stock_industry_history.return_value = [
            {"industry_name": "农林牧渔", "industry_level": "L1"}
        ]

        result = manager.get_stock_industry_history("000001.SZ", "2024-01-01")

        assert result is not None
        assert len(result["industries"]) == 1
        assert result["industries"][0]["industry_name"] == "农林牧渔"
        manager.repository.get_stock_industry_history.assert_called_with("000001.SZ", "2024-01-01")

    def test_validate_industry_data_integrity(self, manager):
        """测试行业数据完整性验证"""
        # 模拟 repository 返回数据
        manager.repository.get_industry_classification_stats.return_value = {"L1": 30, "L2": 100, "L3": 250}
        manager.repository.get_industry_members_stats.return_value = {"total_records": 50000, "current_members": 45000}

        # 执行验证
        manager.validate_industry_data_integrity()

        # 验证 repository 方法是否被调用
        manager.repository.get_industry_classification_stats.assert_called_once()
        manager.repository.get_industry_members_stats.assert_called_once()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
