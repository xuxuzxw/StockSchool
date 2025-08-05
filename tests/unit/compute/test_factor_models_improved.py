from datetime import date, datetime

from src.compute.factor_exceptions import InvalidFactorParameterError
from src.compute.factor_models_improved import (  # !/usr/bin/env python3; -*- coding: utf-8 -*-
    FactorCategory, FactorConstants, FactorMetadata, FactorMetadataFactory,
    FactorStatistics, FactorType, FactorValue, """, import, pytest,
    展示如何测试改进后的代码, 改进的因子模型单元测试)


class TestFactorMetadata:
    """因子元数据测试"""

    def test_create_valid_metadata(self):
        """测试创建有效的因子元数据"""
        metadata = FactorMetadata(
            name="rsi_14",
            description="14日相对强弱指数",
            factor_type=FactorType.TECHNICAL,
            category=FactorCategory.MOMENTUM,
            min_periods=14
        )

        assert metadata.name == "rsi_14"
        assert metadata.factor_type == FactorType.TECHNICAL
        assert metadata.category == FactorCategory.MOMENTUM
        assert metadata.min_periods == 14

    def test_invalid_name_raises_exception(self):
        """测试无效名称抛出异常"""
        with pytest.raises(InvalidFactorParameterError) as exc_info:
            FactorMetadata(
                name="",  # 空名称
                description="测试描述",
                factor_type=FactorType.TECHNICAL,
                category=FactorCategory.MOMENTUM
            )

        assert exc_info.value.parameter_name == "name"

    def test_invalid_min_periods_raises_exception(self):
        """测试无效最小周期抛出异常"""
        with pytest.raises(InvalidFactorParameterError) as exc_info:
            FactorMetadata(
                name="test_factor",
                description="测试描述",
                factor_type=FactorType.TECHNICAL,
                category=FactorCategory.MOMENTUM,
                min_periods=0  # 无效的最小周期
            )

        assert exc_info.value.parameter_name == "min_periods"

    def test_to_dict_serialization(self):
        """测试字典序列化"""
        metadata = FactorMetadata(
            name="rsi_14",
            description="14日相对强弱指数",
            factor_type=FactorType.TECHNICAL,
            category=FactorCategory.MOMENTUM,
            min_periods=14
        )

        result = metadata.to_dict()

        assert result["name"] == "rsi_14"
        assert result["factor_type"] == "technical"
        assert result["category"] == "momentum"
        assert result["min_periods"] == 14
        assert "created_at" in result
        assert "updated_at" in result


class TestFactorValue:
    """因子值测试"""

    def test_create_valid_factor_value(self):
        """测试创建有效的因子值"""
        factor_value = FactorValue(
            ts_code="000001.SZ",
            trade_date=date.today(),
            factor_name="rsi_14",
            raw_value=65.5,
            percentile_rank=0.75
        )

        assert factor_value.ts_code == "000001.SZ"
        assert factor_value.factor_name == "rsi_14"
        assert factor_value.raw_value == 65.5
        assert factor_value.percentile_rank == 0.75

    def test_invalid_percentile_rank_raises_exception(self):
        """测试无效分位数排名抛出异常"""
        with pytest.raises(InvalidFactorParameterError) as exc_info:
            FactorValue(
                ts_code="000001.SZ",
                trade_date=date.today(),
                factor_name="rsi_14",
                raw_value=65.5,
                percentile_rank=1.5  # 超出范围
            )

        assert exc_info.value.parameter_name == "percentile_rank"

    def test_empty_ts_code_raises_exception(self):
        """测试空股票代码抛出异常"""
        with pytest.raises(InvalidFactorParameterError) as exc_info:
            FactorValue(
                ts_code="",  # 空代码
                trade_date=date.today(),
                factor_name="rsi_14",
                raw_value=65.5
            )

        assert exc_info.value.parameter_name == "ts_code"


class TestFactorStatistics:
    """因子统计测试"""

    def test_calculate_statistics_with_valid_data(self):
        """测试使用有效数据计算统计信息"""
        stats = FactorStatistics(
            factor_name="rsi_14",
            calculation_date=date.today(),
            total_count=100,
            valid_count=0  # 将在calculate_statistics中更新
        )

        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats.calculate_statistics(values)

        assert stats.valid_count == 10
        assert stats.mean_value == 55.0
        assert stats.min_value == 10.0
        assert stats.max_value == 100.0
        assert stats.coverage_rate == 0.1  # 10/100
        assert 'p50' in stats.percentiles

    def test_calculate_statistics_with_nan_values(self):
        """测试包含NaN值的统计计算"""
        import numpy as np

        stats = FactorStatistics(
            factor_name="rsi_14",
            calculation_date=date.today(),
            total_count=10,
            valid_count=0
        )

        values = [10.0, np.nan, 30.0, np.nan, 50.0]
        stats.calculate_statistics(values)

        assert stats.valid_count == 3
        assert stats.mean_value == 30.0  # (10+30+50)/3
        assert stats.coverage_rate == 0.3  # 3/10

    def test_invalid_total_count_raises_exception(self):
        """测试无效总数量抛出异常"""
        with pytest.raises(InvalidFactorParameterError) as exc_info:
            FactorStatistics(
                factor_name="rsi_14",
                calculation_date=date.today(),
                total_count=-1,  # 负数
                valid_count=0
            )

        assert exc_info.value.parameter_name == "total_count"


class TestFactorMetadataFactory:
    """因子元数据工厂测试"""

    def test_create_technical_factor(self):
        """测试创建技术面因子"""
        metadata = FactorMetadataFactory.create_technical_factor(
            name="rsi_14",
            description="14日相对强弱指数",
            category=FactorCategory.MOMENTUM,
            min_periods=14
        )

        assert metadata.factor_type == FactorType.TECHNICAL
        assert metadata.category == FactorCategory.MOMENTUM
        assert metadata.name == "rsi_14"
        assert metadata.min_periods == 14

    def test_create_fundamental_factor(self):
        """测试创建基本面因子"""
        metadata = FactorMetadataFactory.create_fundamental_factor(
            name="pe_ratio",
            description="市盈率",
            category=FactorCategory.VALUATION
        )

        assert metadata.factor_type == FactorType.FUNDAMENTAL
        assert metadata.category == FactorCategory.VALUATION
        assert metadata.name == "pe_ratio"

    def test_create_sentiment_factor(self):
        """测试创建情绪面因子"""
        metadata = FactorMetadataFactory.create_sentiment_factor(
            name="attention_score",
            description="关注度评分",
            category=FactorCategory.ATTENTION
        )

        assert metadata.factor_type == FactorType.SENTIMENT
        assert metadata.category == FactorCategory.ATTENTION
        assert metadata.name == "attention_score"


class TestFactorConstants:
    """因子常量测试"""

    def test_constants_are_defined(self):
        """测试常量已定义"""
        assert FactorConstants.DEFAULT_RSI_WINDOW == 14
        assert FactorConstants.MIN_WINDOW_SIZE == 1
        assert FactorConstants.MAX_WINDOW_SIZE == 252
        assert len(FactorConstants.DEFAULT_PERCENTILES) == 7

    def test_percentiles_are_sorted(self):
        """测试分位数已排序"""
        percentiles = FactorConstants.DEFAULT_PERCENTILES
        assert percentiles == sorted(percentiles)
        assert percentiles[0] == 5
        assert percentiles[-1] == 95


# 参数化测试示例
@pytest.mark.parametrize("factor_type,category", [
    (FactorType.TECHNICAL, FactorCategory.MOMENTUM),
    (FactorType.TECHNICAL, FactorCategory.TREND),
    (FactorType.FUNDAMENTAL, FactorCategory.VALUATION),
    (FactorType.SENTIMENT, FactorCategory.ATTENTION),
])
def test_create_metadata_with_different_types(factor_type, category):
    """测试创建不同类型的因子元数据"""
    metadata = FactorMetadata(
        name=f"test_{factor_type.value}_{category.value}",
        description="测试因子",
        factor_type=factor_type,
        category=category
    )

    assert metadata.factor_type == factor_type
    assert metadata.category == category


# 性能测试示例
def test_factor_value_creation_performance():
    """测试因子值创建性能"""
    import time

    start_time = time.time()

    # 创建1000个因子值
    for i in range(1000):
        FactorValue(
            ts_code=f"00000{i % 100:02d}.SZ",
            trade_date=date.today(),
            factor_name="rsi_14",
            raw_value=float(i % 100)
        )

    end_time = time.time()
    execution_time = end_time - start_time

    # 应该在1秒内完成
    assert execution_time < 1.0, f"创建1000个因子值耗时过长: {execution_time:.3f}秒"