from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算数据模型
定义因子计算过程中使用的数据结构和模型
"""


class FactorType(Enum):
    """因子类型枚举"""

    TECHNICAL = "technical"  # 技术面因子
    FUNDAMENTAL = "fundamental"  # 基本面因子
    SENTIMENT = "sentiment"  # 情绪面因子


class FactorCategory(Enum):
    """因子分类枚举"""

    # 技术面分类
    MOMENTUM = "momentum"  # 动量类
    TREND = "trend"  # 趋势类
    VOLATILITY = "volatility"  # 波动率类
    VOLUME = "volume"  # 成交量类

    # 基本面分类
    VALUATION = "valuation"  # 估值类
    PROFITABILITY = "profitability"  # 盈利能力类
    GROWTH = "growth"  # 成长性类
    QUALITY = "quality"  # 质量类
    LEVERAGE = "leverage"  # 杠杆类

    # 情绪面分类
    ATTENTION = "attention"  # 关注度类
    FLOW = "flow"  # 资金流向类
    SENTIMENT_STRENGTH = "sentiment_strength"  # 情绪强度类
    EVENT = "event"  # 事件类


class CalculationStatus(Enum):
    """计算状态枚举"""

    PENDING = "pending"  # 待计算
    RUNNING = "running"  # 计算中
    SUCCESS = "success"  # 成功
    FAILED = "failed"  # 失败
    SKIPPED = "skipped"  # 跳过


@dataclass
class FactorMetadata:
    """因子元数据"""

    name: str  # 因子名称
    description: str  # 因子描述
    factor_type: FactorType  # 因子类型
    category: FactorCategory  # 因子分类
    formula: Optional[str] = None  # 计算公式
    parameters: Dict[str, Any] = field(default_factory=dict)  # 计算参数
    data_requirements: List[str] = field(default_factory=list)  # 数据需求
    min_periods: int = 1  # 最小计算周期
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "factor_type": self.factor_type.value,
            "category": self.category.value,
            "formula": self.formula,
            "parameters": self.parameters,
            "data_requirements": self.data_requirements,
            "min_periods": self.min_periods,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class FactorValue:
    """单个因子值"""

    ts_code: str  # 股票代码
    trade_date: date  # 交易日期
    factor_name: str  # 因子名称
    raw_value: Optional[float]  # 原始值
    standardized_value: Optional[float] = None  # 标准化值
    percentile_rank: Optional[float] = None  # 分位数排名
    industry_rank: Optional[float] = None  # 行业内排名
    is_valid: bool = True  # 是否有效
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "ts_code": self.ts_code,
            "trade_date": self.trade_date,
            "factor_name": self.factor_name,
            "raw_value": self.raw_value,
            "standardized_value": self.standardized_value,
            "percentile_rank": self.percentile_rank,
            "industry_rank": self.industry_rank,
            "is_valid": self.is_valid,
            "created_at": self.created_at,
        }


@dataclass
class FactorResult:
    """因子计算结果"""

    ts_code: str  # 股票代码
    calculation_date: datetime  # 计算时间
    factor_type: FactorType  # 因子类型
    status: CalculationStatus  # 计算状态
    factors: Dict[str, FactorValue] = field(default_factory=dict)  # 因子值字典
    error_message: Optional[str] = None  # 错误信息
    execution_time: Optional[float] = None  # 执行时间（秒）
    data_points: int = 0  # 数据点数量

    def add_factor(self, factor_value: FactorValue):
        """添加因子值"""
        self.factors[factor_value.factor_name] = factor_value

    def get_factor(self, factor_name: str) -> Optional[FactorValue]:
        """获取因子值"""
        return self.factors.get(factor_name)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if not self.factors:
            return pd.DataFrame()

        data = []
        for factor_value in self.factors.values():
            data.append(factor_value.to_dict())

        return pd.DataFrame(data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "ts_code": self.ts_code,
            "calculation_date": self.calculation_date,
            "factor_type": self.factor_type.value,
            "status": self.status.value,
            "factors": {name: factor.to_dict() for name, factor in self.factors.items()},
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "data_points": self.data_points,
        }


@dataclass
class CalculationTask:
    """计算任务"""

    task_id: str  # 任务ID
    ts_codes: List[str]  # 股票代码列表
    factor_types: List[FactorType]  # 因子类型列表
    start_date: Optional[date] = None  # 开始日期
    end_date: Optional[date] = None  # 结束日期
    priority: int = 0  # 优先级（数字越大优先级越高）
    created_at: datetime = field(default_factory=datetime.now)
    status: CalculationStatus = CalculationStatus.PENDING
    progress: float = 0.0  # 进度百分比
    results: List[FactorResult] = field(default_factory=list)

    def update_progress(self, completed: int, total: int):
        """更新进度"""
        self.progress = (completed / total) * 100 if total > 0 else 0.0

    def add_result(self, result: FactorResult):
        """添加计算结果"""
        self.results.append(result)

    def get_success_count(self) -> int:
        """获取成功计算的数量"""
        return sum(1 for result in self.results if result.status == CalculationStatus.SUCCESS)

    def get_failed_count(self) -> int:
        """获取失败计算的数量"""
        return sum(1 for result in self.results if result.status == CalculationStatus.FAILED)


@dataclass
class FactorStatistics:
    """因子统计信息"""

    factor_name: str  # 因子名称
    calculation_date: date  # 计算日期
    total_count: int  # 总数量
    valid_count: int  # 有效数量
    mean_value: Optional[float] = None  # 均值
    std_value: Optional[float] = None  # 标准差
    min_value: Optional[float] = None  # 最小值
    max_value: Optional[float] = None  # 最大值
    percentiles: Dict[str, float] = field(default_factory=dict)  # 分位数
    coverage_rate: float = 0.0  # 覆盖率

    def calculate_statistics(self, values: List[float]):
        """计算统计信息"""
        if not values:
            return

        valid_values = [v for v in values if not pd.isna(v)]
        self.valid_count = len(valid_values)
        self.coverage_rate = self.valid_count / self.total_count if self.total_count > 0 else 0.0

        if valid_values:
            self.mean_value = np.mean(valid_values)
            self.std_value = np.std(valid_values)
            self.min_value = np.min(valid_values)
            self.max_value = np.max(valid_values)

            # 计算分位数
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            for p in percentiles:
                self.percentiles[f"p{p}"] = np.percentile(valid_values, p)


@dataclass
class FactorConfig:
    """因子配置"""

    name: str  # 因子名称
    enabled: bool = True  # 是否启用
    parameters: Dict[str, Any] = field(default_factory=dict)  # 参数配置
    dependencies: List[str] = field(default_factory=list)  # 依赖的因子
    calculation_order: int = 0  # 计算顺序
    cache_enabled: bool = True  # 是否启用缓存
    parallel_enabled: bool = True  # 是否支持并行计算

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """获取参数值"""
        return self.parameters.get(key, default)

    def set_parameter(self, key: str, value: Any):
        """设置参数值"""
        self.parameters[key] = value


class FactorRegistry:
    """因子注册表"""

    def __init__(self):
        """方法描述"""
        self._configs: Dict[str, FactorConfig] = {}

    def register_factor(self, metadata: FactorMetadata, config: Optional[FactorConfig] = None):
        """注册因子"""
        self._factors[metadata.name] = metadata
        if config:
            self._configs[metadata.name] = config
        else:
            self._configs[metadata.name] = FactorConfig(name=metadata.name)

    def get_factor_metadata(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self._factors.get(name)

    def get_factor_config(self, name: str) -> Optional[FactorConfig]:
        """获取因子配置"""
        return self._configs.get(name)

    def list_factors(self, factor_type: Optional[FactorType] = None) -> List[str]:
        """列出因子名称"""
        if factor_type is None:
            return list(self._factors.keys())

        return [name for name, metadata in self._factors.items() if metadata.factor_type == factor_type]

    def get_factors_by_category(self, category: FactorCategory) -> List[str]:
        """按分类获取因子"""
        return [name for name, metadata in self._factors.items() if metadata.category == category]


# 全局因子注册表实例
factor_registry = FactorRegistry()


def create_factor_metadata(
    name: str, description: str, factor_type: FactorType, category: FactorCategory, **kwargs
) -> FactorMetadata:
    """创建因子元数据的便捷函数"""
    return FactorMetadata(name=name, description=description, factor_type=factor_type, category=category, **kwargs)


def create_factor_config(name: str, **kwargs) -> FactorConfig:
    """创建因子配置的便捷函数"""
    return FactorConfig(name=name, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("因子数据模型测试")

    # 创建因子元数据
    rsi_metadata = create_factor_metadata(
        name="rsi_14",
        description="14日相对强弱指数",
        factor_type=FactorType.TECHNICAL,
        category=FactorCategory.MOMENTUM,
        formula="RSI = 100 - (100 / (1 + RS))",
        parameters={"window": 14},
        data_requirements=["close"],
        min_periods=14,
    )

    # 创建因子配置
    rsi_config = create_factor_config(
        name="rsi_14", parameters={"window": 14, "threshold_oversold": 30, "threshold_overbought": 70}
    )

    # 注册因子
    factor_registry.register_factor(rsi_metadata, rsi_config)

    # 创建因子值
    factor_value = FactorValue(
        ts_code="000001.SZ",
        trade_date=date.today(),
        factor_name="rsi_14",
        raw_value=65.5,
        standardized_value=0.8,
        percentile_rank=0.75,
    )

    # 创建因子结果
    result = FactorResult(
        ts_code="000001.SZ",
        calculation_date=datetime.now(),
        factor_type=FactorType.TECHNICAL,
        status=CalculationStatus.SUCCESS,
    )
    result.add_factor(factor_value)

    print(f"注册的因子: {factor_registry.list_factors()}")
    print(f"技术面因子: {factor_registry.list_factors(FactorType.TECHNICAL)}")
    print(f"动量类因子: {factor_registry.get_factors_by_category(FactorCategory.MOMENTUM)}")
    print(f"因子结果: {result.to_dict()}")

    print("因子数据模型测试完成")
